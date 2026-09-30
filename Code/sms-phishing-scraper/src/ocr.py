import csv
import io
import os
import re
import sqlite3
import unicodedata
from pathlib import Path

import pytesseract
import requests
from PIL import Image

from . import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def clean_ocr_text(text: str) -> str:
    """Normalize OCR output into a single readable line."""
    normalized = unicodedata.normalize("NFKC", text)
    cleaned_lines = []
    for line in normalized.splitlines():
        line = "".join(char for char in line if char.isprintable())
        line = re.sub(r"\s+", " ", line).strip()
        if line and (not cleaned_lines or line != cleaned_lines[-1]):
            cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def is_image_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(IMAGE_EXTENSIONS)


def download_image(url: str, timeout: int = 10) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def extract_text_from_image(image: Image.Image) -> str:
    try:
        return clean_ocr_text(pytesseract.image_to_string(image))
    except Exception:
        return ""


def extract_text_from_url(url: str) -> str:
    if not is_image_url(url):
        return ""
    image = download_image(url)
    if image is None:
        return ""
    return extract_text_from_image(image)


def export_image_directory_to_csv(
    image_dir: str | None = None,
    csv_path: str | None = None,
) -> int:
    """OCR supported images in the image test folder and write their text to CSV."""
    source_dir = Path(image_dir or config.IMAGE_TEST_DIR)
    output_csv = csv_path or config.OCR_CSV_PATH
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("photo_key", "text"))
        writer.writeheader()
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    text = extract_text_from_image(image.convert("RGB"))
            except Exception:
                text = ""
            writer.writerow(
                {
                    "photo_key": image_path.relative_to(source_dir).as_posix(),
                    "text": text,
                }
            )
    return len(image_paths)


def cache_image(url: str, post_id: str) -> str | None:
    image = download_image(url)
    if image is None:
        return None
    os.makedirs(config.IMAGE_CACHE_DIR, exist_ok=True)
    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    path = os.path.join(config.IMAGE_CACHE_DIR, f"{post_id}{ext}")
    image.save(path)
    return path


def export_database_images_to_csv(
    db_path: str | None = None,
    csv_path: str | None = None,
) -> int:
    """OCR image bytes from SQLite and export each result with its image key."""
    source_db = db_path or config.DB_PATH
    output_csv = csv_path or config.OCR_CSV_PATH
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    exported = 0
    connection = sqlite3.connect(source_db)
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=("photo_key", "text"))
            writer.writeheader()
            rows = connection.execute(
                "SELECT image_key, image_data FROM images ORDER BY image_key"
            )
            for image_key, image_data in rows:
                try:
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    text = extract_text_from_image(image)
                except Exception:
                    text = ""
                writer.writerow({"photo_key": image_key, "text": text})
                exported += 1
    finally:
        connection.close()
    return exported
