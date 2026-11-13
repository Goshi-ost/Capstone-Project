"""Keyless scrapers: Reddit public RSS feed, Mastodon hashtag timelines, Lemmy search.

None of these need an API key or account; they only need a descriptive User-Agent
and polite request rates (both handled by pipeline.http_get). When Reddit API
credentials are configured, main.py uses scrapers/reddit_scraper.py instead of the RSS feed.
"""
import html
import re
import xml.etree.ElementTree as ET

from .. import pipeline
from ..configs import config
from ..Tools import ocr

TAG_RE = re.compile(r"<[^>]+>")
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


def scan_reddit(subreddit: str, limit: int = 50) -> list[dict]:
    """Uses the public Atom feed, which needs no key (the .json endpoints are blocked)."""
    resp = pipeline.http_get(
        f"https://www.reddit.com/r/{subreddit}/new/.rss", {"limit": min(limit, 100)}
    )
    if resp is None:
        return []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        print(f"[warn] reddit r/{subreddit} returned an unreadable feed: {exc}")
        return []

    flagged = []
    for entry in root.findall("a:entry", ATOM_NS):
        post_id = entry.findtext("a:id", "", ATOM_NS).removeprefix("t3_")
        link_el = entry.find("a:link", ATOM_NS)
        link = link_el.get("href") if link_el is not None else ""
        title = entry.findtext("a:title", "", ATOM_NS)
        body = html.unescape(entry.findtext("a:content", "", ATOM_NS))
        posted_at = entry.findtext("a:published", None, ATOM_NS) or entry.findtext("a:updated", None, ATOM_NS)
        urls = re.findall(r'(?:href|src)="([^"]+)"', body)
        images = list(dict.fromkeys(html.unescape(u) for u in urls if ocr.is_image_url(html.unescape(u))))
        text = f"{title}\n{TAG_RE.sub(' ', body)}"
        hit = pipeline.process_post(f"reddit/{subreddit}", post_id, link, text, images, posted_at)
        if hit:
            flagged.append(hit)
    return flagged


def scan_mastodon_tag(instance: str, tag: str, limit: int = 40) -> list[dict]:
    posts = pipeline.http_get_json(
        f"https://{instance}/api/v1/timelines/tag/{tag}", {"limit": min(limit, 40)}
    )
    flagged = []
    for p in posts or []:
        images = [m["url"] for m in p.get("media_attachments", []) if m.get("type") == "image"]
        text = html.unescape(TAG_RE.sub(" ", p.get("content", "")))
        hit = pipeline.process_post(
            f"mastodon/{instance}/{tag}", p["id"], p["url"], text, images, p.get("created_at")
        )
        if hit:
            flagged.append(hit)
    return flagged


def scan_lemmy(instance: str, query: str, limit: int = 40) -> list[dict]:
    data = pipeline.http_get_json(
        f"https://{instance}/api/v3/search",
        {"q": query, "type_": "Posts", "sort": "New", "limit": min(limit, 50)},
    )
    flagged = []
    for item in (data or {}).get("posts", []):
        p = item["post"]
        # Prefer the full-size image; the thumbnail is only a fallback.
        images = [u for u in (p.get("url"), p.get("thumbnail_url")) if u and ocr.is_image_url(u)]
        text = f"{p.get('name', '')}\n{p.get('body', '')}"
        hit = pipeline.process_post(
            f"lemmy/{instance}", str(p["id"]), p["ap_id"], text, images[:1], p.get("published")
        )
        if hit:
            flagged.append(hit)
    return flagged


def scan_reddit_all(limit: int = 50) -> list[dict]:
    flagged = []
    for sub in config.DEFAULT_SUBREDDITS:
        flagged += scan_reddit(sub, limit)
    return flagged


def scan_fediverse(limit: int = 50) -> list[dict]:
    flagged = []
    for instance, tag in config.MASTODON_TAGS:
        flagged += scan_mastodon_tag(instance, tag, limit)
    for instance, query in config.LEMMY_QUERIES:
        flagged += scan_lemmy(instance, query, limit)
    return flagged


def scan_all(limit: int = 50) -> list[dict]:
    return scan_reddit_all(limit) + scan_fediverse(limit)
