"""Classify test images, OCR SMS screenshots, and export spam analysis."""

import argparse
import csv
from pathlib import Path

from PIL import Image

from src.configs import config
from src.Tools import ocr
from src.Tools.phishing_detector import PhishingDetector
from src.Tools.smsImageClassifier import SmsImageClassifier


FIELDS = (
    "photo_key", "image_label", "image_confidence", "ocr_text", "spam_label",
    "spam_score_percent", "is_likely_phishing", "matched_categories",
    "suspicious_urls", "status",
)
DEFAULT_OUTPUT = Path(config.IMAGE_TEST_DIR) / "image_test_results.crv"


def run(image_dir: Path, output_path: Path) -> int:
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    images = sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ocr.IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    classifier = SmsImageClassifier()
    detector = PhishingDetector()
    rows = []
    for path in images:
        label, confidence = classifier.predict(path)
        if label not in ("sms", "non_sms"):
            raise ValueError(f"Unexpected image class {label!r} for {path}")
        row = {
            "photo_key": path.relative_to(image_dir).as_posix(),
            "image_label": label,
            "image_confidence": confidence,
            "status": "not_sms",
        }
        if label == "sms":
            with Image.open(path) as image:
                text = ocr.extract_text_from_image_strict(image.convert("RGB"))
            row["ocr_text"] = text
            row["status"] = "empty_ocr"
            if text.strip():
                analysis = detector.analyze(text)
                assert detector.model is not None
                row.update({
                    "spam_label": str(detector.model.predict([text])[0]),
                    "spam_score_percent": analysis.score,
                    "is_likely_phishing": analysis.is_likely_phishing,
                    "matched_categories": "; ".join(analysis.matched_categories),
                    "suspicious_urls": "; ".join(analysis.suspicious_urls),
                    "status": "complete",
                })
        rows.append(row)

    with output_path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="New CSV-formatted .crv output file (must not already exist)",
    )
    args = parser.parse_args()
    count = run(Path(config.IMAGE_TEST_DIR), args.output)
    print(f"Processed {count} images; results saved to {args.output}")


if __name__ == "__main__":
    main()
