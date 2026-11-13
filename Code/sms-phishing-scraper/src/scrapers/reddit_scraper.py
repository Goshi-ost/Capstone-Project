"""Reddit scraper using the official API through PRAW (needs REDDIT_CLIENT_ID / SECRET).

Preferred over the public RSS feed when credentials exist: higher rate limits, no
blocking, gallery support, and exact post timestamps.
"""

import html
from datetime import datetime, timezone

import praw
import prawcore

from .. import pipeline
from ..configs import config
from ..Tools import ocr


def has_credentials() -> bool:
    return bool(config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET)


def get_reddit_client() -> praw.Reddit:
    if not has_credentials():
        raise RuntimeError(
            "Missing Reddit API credentials. Copy .env.example to .env and fill in "
            "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET from https://www.reddit.com/prefs/apps"
        )
    return praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )


# Collect the direct image URL and any gallery images, without duplicates.
def _image_urls(submission) -> list[str]:
    urls = []
    url = getattr(submission, "url", "") or ""
    if ocr.is_image_url(url):
        urls.append(url)

    if getattr(submission, "is_gallery", False):
        items = (getattr(submission, "gallery_data", None) or {}).get("items", [])
        metadata = getattr(submission, "media_metadata", None) or {}
        for item in items:
            image_url = metadata.get(item.get("media_id"), {}).get("s", {}).get("u", "")
            if image_url:
                urls.append(html.unescape(image_url))
    return list(dict.fromkeys(urls))


def scan_subreddit(reddit: praw.Reddit, subreddit_name: str, limit: int = 50) -> list[dict]:
    flagged = []
    try:
        for submission in reddit.subreddit(subreddit_name).new(limit=limit):
            posted_at = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat()
            hit = pipeline.process_post(
                source=f"reddit/{subreddit_name}",
                post_id=submission.id,
                permalink=f"https://www.reddit.com{submission.permalink}",
                text=f"{submission.title or ''}\n{submission.selftext or ''}",
                image_urls=_image_urls(submission),
                posted_at=posted_at,
            )
            if hit:
                flagged.append(hit)
    except prawcore.exceptions.PrawcoreException as exc:
        print(f"[warn] reddit r/{subreddit_name} failed: {exc}")
    return flagged


def scan_subreddits(subreddit_names: list[str], limit: int = 50) -> list[dict]:
    reddit = get_reddit_client()
    results = []
    for name in subreddit_names:
        results.extend(scan_subreddit(reddit, name, limit=limit))
    return results
