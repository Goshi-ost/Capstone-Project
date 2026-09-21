import io
import os

import pytesseract
import requests
from PIL import Image

from . import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


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
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""


def extract_text_from_url(url: str) -> str:
    if not is_image_url(url):
        return ""
    image = download_image(url)
    if image is None:
        return ""
    return extract_text_from_image(image)


def cache_image(url: str, post_id: str) -> str | None:
    image = download_image(url)
    if image is None:
        return None
    os.makedirs(config.IMAGE_CACHE_DIR, exist_ok=True)
    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    path = os.path.join(config.IMAGE_CACHE_DIR, f"{post_id}{ext}")
    image.save(path)
    return path
