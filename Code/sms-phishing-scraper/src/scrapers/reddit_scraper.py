import praw

from . import config, ocr, phishing_detector, storage


def get_reddit_client() -> praw.Reddit:
    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        raise RuntimeError(
            "Missing Reddit API credentials. Copy .env.example to .env and fill in "
            "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET from https://www.reddit.com/prefs/apps"
        )
    return praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )


def _gather_post_text(submission) -> str:
    parts = [submission.title or "", submission.selftext or ""]

    # Screenshots of scam texts are usually attached as an image URL
    if getattr(submission, "url", None) and ocr.is_image_url(submission.url):
        parts.append(ocr.extract_text_from_url(submission.url))

    # Gallery posts have multiple images
    if getattr(submission, "is_gallery", False):
        try:
            media_ids = [item["media_id"] for item in submission.gallery_data["items"]]
            for media_id in media_ids:
                meta = submission.media_metadata.get(media_id, {})
                image_url = meta.get("s", {}).get("u", "").replace("&amp;", "&")
                if image_url:
                    parts.append(ocr.extract_text_from_url(image_url))
        except Exception:
            pass

    return "\n".join(p for p in parts if p)


def scan_subreddit(reddit: praw.Reddit, subreddit_name: str, limit: int = 50) -> list[dict]:
    subreddit = reddit.subreddit(subreddit_name)
    flagged = []

    for submission in subreddit.new(limit=limit):
        text = _gather_post_text(submission)
        if not text.strip():
            continue

        analysis = phishing_detector.analyze_text(text)
        storage.save_result(
            source=f"reddit/{subreddit_name}",
            post_id=submission.id,
            permalink=f"https://reddit.com{submission.permalink}",
            analysis=analysis,
        )

        if analysis.is_likely_phishing:
            flagged.append(
                {
                    "post_id": submission.id,
                    "permalink": f"https://reddit.com{submission.permalink}",
                    "score": analysis.score,
                    "matched_categories": analysis.matched_categories,
                }
            )

    return flagged


def scan_subreddits(subreddit_names: list[str], limit: int = 50) -> list[dict]:
    reddit = get_reddit_client()
    results = []
    for name in subreddit_names:
        results.extend(scan_subreddit(reddit, name, limit=limit))
    return results
