from src import pipeline
from src.configs import config
from src.scrapers import public_scraper, reddit_scraper
from src.Tools import ocr, storage


def main() -> None:
    storage.init_db()
    limit = config.SCAN_LIMIT
    flagged = []

    if reddit_scraper.has_credentials():
        print(f"Reddit: official API, r/{', r/'.join(config.DEFAULT_SUBREDDITS)}")
        flagged += reddit_scraper.scan_subreddits(config.DEFAULT_SUBREDDITS, limit=limit)
    else:
        print("Reddit: no API credentials in .env, using the public RSS feed (may be rate-limited)")
        flagged += public_scraper.scan_reddit_all(limit)

    print(f"Mastodon tags: {config.MASTODON_TAGS}; Lemmy queries: {config.LEMMY_QUERIES}")
    flagged += public_scraper.scan_fediverse(limit)

    exported_images = ocr.export_database_images_to_csv()

    print()
    for label in ("new posts analyzed", "skipped (already stored)", "flagged as phishing"):
        print(f"{label}: {pipeline.stats[label]}")
    print(f"Exported OCR text for {exported_images} images to {config.OCR_CSV_PATH}")
    print(f"\nFlagged {len(flagged)} likely-phishing posts this run:")
    for item in flagged:
        categories = ", ".join(item["matched_categories"]) or "suspicious link"
        print(f"- [{item['score']}] {item['permalink']} ({categories}; from {item['text_source']} text)")


if __name__ == "__main__":
    main()
