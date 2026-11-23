import csv
from types import SimpleNamespace

import pytest
from PIL import Image

import run_image_checks


def test_image_checks_gate_stages_and_write_csv(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("sms.png", "empty.png", "other.png"):
        Image.new("RGB", (4, 4), "white").save(images / name)

    class Classifier:
        def predict(self, path):
            return ("non_sms" if path.name == "other.png" else "sms", 0.9)

    class Detector:
        def __init__(self):
            self.model = SimpleNamespace(predict=lambda texts: ["spam"])

        def analyze(self, text):
            assert text == "A message, with punctuation"
            return SimpleNamespace(
                score=87, is_likely_phishing=True,
                matched_categories=["banking"], suspicious_urls=[],
            )

    ocr_results = iter(["", "A message, with punctuation"])
    monkeypatch.setattr(run_image_checks, "SmsImageClassifier", Classifier)
    monkeypatch.setattr(run_image_checks, "PhishingDetector", Detector)
    monkeypatch.setattr(
        run_image_checks.ocr, "extract_text_from_image_strict",
        lambda image: next(ocr_results),
    )
    output = tmp_path / "results.crv"

    assert run_image_checks.run(images, output) == 3
    with output.open(encoding="utf-8", newline="") as file:
        rows = {row["photo_key"]: row for row in csv.DictReader(file)}
    assert rows["other.png"]["status"] == "not_sms"
    assert rows["other.png"]["ocr_text"] == rows["other.png"]["spam_label"] == ""
    assert rows["empty.png"]["status"] == "empty_ocr"
    assert rows["empty.png"]["spam_label"] == ""
    assert rows["sms.png"]["ocr_text"] == "A message, with punctuation"
    assert rows["sms.png"]["spam_label"] == "spam"
    assert rows["sms.png"]["spam_score_percent"] == "87"
    assert rows["sms.png"]["is_likely_phishing"] == "True"
    assert rows["sms.png"]["status"] == "complete"


def test_ocr_failure_does_not_create_results(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4)).save(images / "sms.png")
    output = tmp_path / "results.crv"
    monkeypatch.setattr(
        run_image_checks, "SmsImageClassifier",
        lambda: SimpleNamespace(predict=lambda path: ("sms", 1.0)),
    )
    monkeypatch.setattr(run_image_checks, "PhishingDetector", lambda: None)

    def fail(image):
        raise RuntimeError("OCR failed")

    monkeypatch.setattr(run_image_checks.ocr, "extract_text_from_image_strict", fail)
    with pytest.raises(RuntimeError, match="OCR failed"):
        run_image_checks.run(images, output)
    assert not output.exists()


def test_existing_results_are_not_overwritten(tmp_path):
    output = tmp_path / "results.crv"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_image_checks.run(tmp_path, output)
    assert output.read_text(encoding="utf-8") == "existing"
