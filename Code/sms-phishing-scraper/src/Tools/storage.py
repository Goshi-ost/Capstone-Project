"""SQLite persistence for phishing analyses and downloaded image data."""

import os
import sqlite3
from contextlib import contextmanager

from ..configs import config

SCHEMA = """
-- Store each analyzed post once, with its classification details.
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    post_id TEXT NOT NULL UNIQUE,
    permalink TEXT,
    text TEXT,
    score INTEGER,
    matched_keywords TEXT,
    matched_categories TEXT,
    suspicious_urls TEXT,
    is_likely_phishing INTEGER,
    posted_at TEXT,
    text_source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Keep downloaded image bytes keyed by their source-specific image identifier.
CREATE TABLE IF NOT EXISTS images (
    image_key TEXT PRIMARY KEY,
    image_data BLOB NOT NULL,
    source TEXT,
    post_id TEXT,
    ocr_text TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# Columns added after the first schema; existing databases get them via ALTER TABLE.
#   posted_at   - when the post was published on its platform (for freshness comparisons)
#   text_source - which text was classified: "ocr" (screenshot text), "post", or "none"
#   ocr_text    - cached OCR output so images are never OCR'd twice
_ADDED_COLUMNS = {
    "results": {"posted_at": "TEXT", "text_source": "TEXT"},
    "images": {"post_id": "TEXT", "ocr_text": "TEXT"},
}


# Yield a database connection and close it even if the caller raises an exception.
@contextmanager
def get_connection(db_path: str | None = None):
    conn = sqlite3.connect(db_path or config.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, sql_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


# Create the database directory and tables, upgrading older databases in place.
def init_db(db_path: str | None = None) -> None:
    path = db_path or config.DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with get_connection(path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()


# Check whether a post has already been analyzed, so re-runs can skip it before downloading anything.
def has_result(post_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM results WHERE post_id = ?", (post_id,)).fetchone()
    return row is not None


def save_image(
    image_key: str,
    image_data: bytes,
    source: str,
    post_id: str | None = None,
    ocr_text: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO images (image_key, image_data, source, post_id, ocr_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (image_key, image_data, source, post_id, ocr_text),
        )
        conn.commit()


# Insert a classification result unless its post ID has already been stored.
def save_result(
    source: str,
    post_id: str,
    permalink: str,
    analysis,
    posted_at: str | None = None,
    text_source: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO results
                (source, post_id, permalink, text, score, matched_keywords,
                 matched_categories, suspicious_urls, is_likely_phishing,
                 posted_at, text_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                post_id,
                permalink,
                analysis.text,
                analysis.score,
                ",".join(analysis.matched_keywords),
                ",".join(analysis.matched_categories),
                ",".join(analysis.suspicious_urls),
                int(analysis.is_likely_phishing),
                posted_at,
                text_source,
            ),
        )
        conn.commit()
