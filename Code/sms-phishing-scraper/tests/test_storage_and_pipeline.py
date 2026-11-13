import sqlite3

from PIL import Image

from src import pipeline
from src.Tools import ocr, phishing_detector, storage

OLD_SCHEMA = """
CREATE TABLE results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, post_id TEXT NOT NULL UNIQUE,
    permalink TEXT, text TEXT, score INTEGER, matched_keywords TEXT, matched_categories TEXT,
    suspicious_urls TEXT, is_likely_phishing INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE images (image_key TEXT PRIMARY KEY, image_data BLOB NOT NULL, source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""


def columns(db_path, table):
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_init_db_upgrades_old_database(tmp_path):
    db_path = str(tmp_path / "old.db")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(OLD_SCHEMA)
    storage.init_db(db_path)
    assert {"posted_at", "text_source"} <= columns(db_path, "results")
    assert {"post_id", "ocr_text"} <= columns(db_path, "images")


def fake_analyze(text):
    return phishing_detector.PhishingAnalysis(
        text=text, score=95, matched_categories=["delivery"], is_likely_phishing="package" in text
    )


def test_process_post_classifies_ocr_text_and_skips_repeats(temp_db, monkeypatch):
    fetches = []

    def fake_fetch(url):
        fetches.append(url)
        return b"img-bytes", Image.new("RGB", (10, 10))

    monkeypatch.setattr(pipeline, "fetch_image", fake_fetch)
    monkeypatch.setattr(ocr, "extract_text_from_image", lambda img: "USPS: package on hold")
    monkeypatch.setattr(phishing_detector, "analyze_text", fake_analyze)

    args = ("reddit/Scams", "abc", "https://reddit.com/x", "Is this a scam?", ["https://i.redd.it/a.png"])
    hit = pipeline.process_post(*args, posted_at="2026-11-01T00:00:00+00:00")
    assert hit["text_source"] == "ocr"

    with sqlite3.connect(temp_db) as conn:
        row = conn.execute("SELECT text, text_source, posted_at FROM results").fetchone()
        image_text = conn.execute("SELECT ocr_text FROM images").fetchone()[0]
    assert row == ("USPS: package on hold", "ocr", "2026-11-01T00:00:00+00:00")
    assert image_text == "USPS: package on hold"

    # Second run: already stored, so nothing is downloaded again.
    assert pipeline.process_post(*args) is None
    assert len(fetches) == 1


def test_process_post_falls_back_to_post_text(temp_db, monkeypatch):
    monkeypatch.setattr(phishing_detector, "analyze_text", fake_analyze)
    hit = pipeline.process_post("lemmy/x", "1", "https://l/1", "your package is held", [])
    assert hit["text_source"] == "post"


def test_empty_post_is_recorded_so_it_is_not_refetched(temp_db, monkeypatch):
    monkeypatch.setattr(pipeline, "fetch_image", lambda url: None)
    assert pipeline.process_post("lemmy/x", "2", "https://l/2", "", ["https://x/a.png"]) is None
    assert storage.has_result("lemmy/x:2")


def test_export_uses_cached_ocr_text(temp_db, tmp_path, monkeypatch):
    storage.save_image("k1", b"x", "src", ocr_text="cached text")
    monkeypatch.setattr(ocr, "extract_text_from_image", lambda img: (_ for _ in ()).throw(AssertionError))
    out = tmp_path / "out.csv"
    assert ocr.export_database_images_to_csv(csv_path=str(out)) == 1
    assert "k1,cached text" in out.read_text(encoding="utf-8")
