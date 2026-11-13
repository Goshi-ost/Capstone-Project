"""Keyless scrapers: Reddit public JSON, Mastodon hashtag timelines, Lemmy search.

None of these need an API key or account; they only need a descriptive User-Agent
and polite request rates.
"""
import html
import io
import re
import time
import xml.etree.ElementTree as ET

import requests
from PIL import Image

from .Tools import ocr, phishing_detector, storage
from .configs import config

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
REQUEST_DELAY = 1.5

TAG_RE = re.compile(r"<[^>]+>")


def _get_json(url: str, params: dict | None = None):
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[warn] {url} failed: {exc}")
        return None


def _fetch_image(url: str) -> tuple[bytes, Image.Image] | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.content, Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _process(source: str, post_id: str, permalink: str, text: str, image_urls: list[str]) -> dict | None:
    """OCR images, store them, analyze the combined text, and save the result."""
    parts = [text]
    for i, url in enumerate(image_urls):
        fetched = _fetch_image(url)
        if not fetched:
            continue
        data, image = fetched
        storage.save_image(f"{source}/{post_id}/{i}", data, source)
        parts.append(ocr.extract_text_from_image(image))

    combined = "\n".join(p for p in parts if p and p.strip())
    if not combined.strip():
        return None

    analysis = phishing_detector.analyze_text(combined)
    storage.save_result(source=source, post_id=f"{source}:{post_id}", permalink=permalink, analysis=analysis)
    if not analysis.is_likely_phishing:
        return None
    return {
        "post_id": post_id,
        "permalink": permalink,
        "score": analysis.score,
        "matched_categories": analysis.matched_categories,
    }


def scan_reddit(subreddit: str, limit: int = 50) -> list[dict]:
    """Uses the public Atom feed, which needs no key (the .json endpoints are blocked)."""
    try:
        resp = requests.get(
            f"https://www.reddit.com/r/{subreddit}/new/.rss",
            params={"limit": min(limit, 100)},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        time.sleep(REQUEST_DELAY)
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"[warn] reddit r/{subreddit} failed: {exc}")
        return []

    ns = {"a": "http://www.w3.org/2005/Atom"}
    flagged = []
    for entry in root.findall("a:entry", ns):
        post_id = entry.findtext("a:id", "", ns).removeprefix("t3_")
        link = entry.find("a:link", ns).get("href")
        title = entry.findtext("a:title", "", ns)
        body = html.unescape(entry.findtext("a:content", "", ns))
        urls = re.findall(r'(?:href|src)="([^"]+)"', body)
        images = list(dict.fromkeys(html.unescape(u) for u in urls if ocr.is_image_url(html.unescape(u))))
        text = f"{title}\n{TAG_RE.sub(' ', body)}"
        hit = _process(f"reddit/{subreddit}", post_id, link, text, images)
        if hit:
            flagged.append(hit)
    return flagged


def scan_mastodon_tag(instance: str, tag: str, limit: int = 40) -> list[dict]:
    posts = _get_json(f"https://{instance}/api/v1/timelines/tag/{tag}", {"limit": min(limit, 40)})
    flagged = []
    for p in posts or []:
        images = [m["url"] for m in p.get("media_attachments", []) if m.get("type") == "image"]
        text = html.unescape(TAG_RE.sub(" ", p.get("content", "")))
        hit = _process(f"mastodon/{instance}/{tag}", p["id"], p["url"], text, images)
        if hit:
            flagged.append(hit)
    return flagged


def scan_lemmy(instance: str, query: str, limit: int = 40) -> list[dict]:
    data = _get_json(
        f"https://{instance}/api/v3/search",
        {"q": query, "type_": "Posts", "sort": "New", "limit": min(limit, 50)},
    )
    flagged = []
    for item in (data or {}).get("posts", []):
        p = item["post"]
        images = [u for u in (p.get("url"), p.get("thumbnail_url")) if u and ocr.is_image_url(u)]
        text = f"{p.get('name', '')}\n{p.get('body', '')}"
        hit = _process(f"lemmy/{instance}", str(p["id"]), p["ap_id"], text, images[:1])
        if hit:
            flagged.append(hit)
    return flagged


def scan_all(limit: int = 50) -> list[dict]:
    flagged = []
    for sub in config.DEFAULT_SUBREDDITS:
        flagged += scan_reddit(sub, limit)
    for instance, tag in config.MASTODON_TAGS:
        flagged += scan_mastodon_tag(instance, tag, limit)
    for instance, query in config.LEMMY_QUERIES:
        flagged += scan_lemmy(instance, query, limit)
    return flagged
