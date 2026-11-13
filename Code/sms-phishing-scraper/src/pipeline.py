"""Shared per-post pipeline used by every scraper: dedupe, download, OCR, classify, store."""

import io
import time
from collections import Counter

import requests
from PIL import Image

from .configs import config
from .Tools import ocr, phishing_detector, storage

HEADERS = {"User-Agent": config.HTTP_USER_AGENT}
REQUEST_DELAY = 1.5
MAX_RETRY_WAIT = 60

# Running totals for the current process, printed by main.py.
stats: Counter = Counter()


# GET with an honest User-Agent, a polite delay, and one retry when rate-limited (HTTP 429).
def http_get(url: str, params: dict | None = None, retries: int = 1) -> requests.Response | None:
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        except requests.RequestException as exc:
            print(f"[warn] {url} failed: {exc}")
            return None
        time.sleep(REQUEST_DELAY)
        if resp.status_code == 429 and attempt < retries:
            try:
                wait = min(int(resp.headers.get("Retry-After", "10")), MAX_RETRY_WAIT)
            except ValueError:
                wait = 10
            print(f"[info] rate-limited by {url}; retrying in {wait}s")
            time.sleep(wait)
            continue
        if not resp.ok:
            print(f"[warn] {url} failed: HTTP {resp.status_code}")
            return None
        return resp
    return None


def http_get_json(url: str, params: dict | None = None):
    resp = http_get(url, params)
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError as exc:
        print(f"[warn] {url} returned invalid JSON: {exc}")
        return None


def fetch_image(url: str) -> tuple[bytes, Image.Image] | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.content, Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def process_post(
    source: str,
    post_id: str,
    permalink: str,
    text: str,
    image_urls: list[str],
    posted_at: str | None = None,
) -> dict | None:
    """Analyze one post and store the result. Returns a summary dict if it is likely phishing.

    The screenshot text (OCR) is what gets classified when there is any, because the
    classifier is trained on SMS messages; the poster's own title/caption describes the
    scam rather than being the scam. Post text is only used when no image text exists.
    """
    key = f"{source}:{post_id}"
    if storage.has_result(key):
        stats["skipped (already stored)"] += 1
        return None

    ocr_parts = []
    for i, url in enumerate(image_urls):
        fetched = fetch_image(url)
        if not fetched:
            continue
        data, image = fetched
        image_text = ocr.extract_text_from_image(image)
        storage.save_image(f"{source}/{post_id}/{i}", data, source, post_id=key, ocr_text=image_text)
        if image_text.strip():
            ocr_parts.append(image_text)

    if ocr_parts:
        analyzed, text_source = "\n".join(ocr_parts), "ocr"
    elif text and text.strip():
        analyzed, text_source = text.strip(), "post"
    else:
        analyzed, text_source = "", "none"

    if analyzed:
        analysis = phishing_detector.analyze_text(analyzed)
    else:
        # Still record the post so it is not downloaded again on the next run.
        analysis = phishing_detector.PhishingAnalysis(text="")

    storage.save_result(
        source=source,
        post_id=key,
        permalink=permalink,
        analysis=analysis,
        posted_at=posted_at,
        text_source=text_source,
    )
    stats["new posts analyzed"] += 1
    if not analysis.is_likely_phishing:
        return None
    stats["flagged as phishing"] += 1
    return {
        "source": source,
        "post_id": post_id,
        "permalink": permalink,
        "score": analysis.score,
        "matched_categories": analysis.matched_categories,
        "text_source": text_source,
    }
