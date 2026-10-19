import csv
import io
import os
import re
import sqlite3
import unicodedata
from pathlib import Path

import pytesseract
import requests
from PIL import Image, ImageFilter, ImageOps, ImageStat

from . import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


TARGET_MIN_WIDTH = 1500
MAX_SCALE = 3.0
MAX_PIXELS = 25_000_000
TESSERACT_CONFIG = "--oem 3 --psm 6"
MIN_ALNUM_RATIO = 0.4

_ASCII_PUNCT = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2022": " ",
        "\u00a0": " ",
    }
)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Prepare an SMS screenshot for Tesseract (grayscale, upscale, contrast, denoise)."""
    gray = ImageOps.grayscale(image)

    # Dark-mode chat themes: Tesseract prefers dark text on a light background.
    if ImageStat.Stat(gray).mean[0] < 128:
        gray = ImageOps.invert(gray)

    width, height = gray.size
    if width < TARGET_MIN_WIDTH:
        scale = min(TARGET_MIN_WIDTH / width, MAX_SCALE)
        if width * height * scale * scale <= MAX_PIXELS:
            gray = gray.resize(
                (round(width * scale), round(height * scale)), Image.LANCZOS
            )

    gray = ImageOps.autocontrast(gray, cutoff=1)
    return gray.filter(ImageFilter.MedianFilter(3))


def _repair_url(match: re.Match) -> str:
    return match.group(1) + re.sub(r"\s+", "", match.group(2))


def _is_noise_line(line: str) -> bool:
    visible = [c for c in line if not c.isspace()]
    if not visible:
        return True
    alnum = sum(c.isalnum() for c in visible)
    return alnum / len(visible) < MIN_ALNUM_RATIO and len(visible) < 6


_CLOCK = re.compile(r"^\d{1,2}:\d{2}(\s?[AaPp][Mm])?$")
_NETWORK = re.compile(r"^(?:[2-5]G|LTE|WIFI|WI-FI|VOLTE|\d{1,3}%)$", re.IGNORECASE)


def _is_status_bar_line(line: str) -> bool:
    """Phone status bar: a clock and/or network tokens plus short signal/battery junk."""
    tokens = line.split()
    if not tokens or len(tokens) > 8 or any(len(t) > 5 for t in tokens):
        return False
    if len(tokens) == 1 and tokens[0].isdigit():
        return True
    return bool(_CLOCK.match(tokens[0]) or any(_NETWORK.match(t) for t in tokens))


def clean_ocr_text(text: str) -> str:
    """Normalize OCR output into a single readable ASCII line."""
    normalized = unicodedata.normalize("NFKC", text).translate(_ASCII_PUNCT)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned_lines = []
    for line in normalized.splitlines():
        line = "".join(char for char in line if char.isprintable())
        line = re.sub(r"\s+", " ", line).strip()
        if _is_noise_line(line):
            continue
        if not cleaned_lines or line != cleaned_lines[-1]:
            cleaned_lines.append(line)

    while cleaned_lines[:1] and _is_status_bar_line(cleaned_lines[0]):
        cleaned_lines.pop(0)

    # Join hyphenated words split across lines, then merge lines into one.
    joined = " ".join(cleaned_lines)
    joined = re.sub(r"(?<=[a-z])- (?=[a-z])", "", joined)

    # URLs: "http : // bit . ly / x" -> "http://bit.ly/x"
    joined = re.sub(r"\bhxxp", "http", joined, flags=re.IGNORECASE)
    joined = re.sub(
        r"\b(https?)\s*:\s*/\s*/\s*", lambda m: m.group(1).lower() + "://", joined,
        flags=re.IGNORECASE,
    )
    joined = re.sub(
        r"(https?://|\bwww\s*\.\s*)([\w\-]+(?:\s*[./]\s*[\w\-]+)*)",
        _repair_url,
        joined,
        flags=re.IGNORECASE,
    )

    # Digits misread as letters between digits (phone numbers, amounts, codes).
    joined = re.sub(r"(?<=\d)[Oo](?=\d)", "0", joined)
    joined = re.sub(r"(?<=\d)[Il](?=\d)", "1", joined)

    # Punctuation spacing.
    joined = re.sub(r"\s+([,;:!?])", r"\1", joined)
    joined = re.sub(r"\s+\.(?=\s|$)", ".", joined)
    joined = re.sub(r"[|~_]{2,}|\s[|~_]\s", " ", joined)
    return re.sub(r"\s+", " ", joined).strip()


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
        prepared = preprocess_image(image)
        return clean_ocr_text(
            pytesseract.image_to_string(prepared, config=TESSERACT_CONFIG)
        )
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
