from src import config, reddit_scraper, storage


def main() -> None:
    storage.init_db()
    flagged = reddit_scraper.scan_subreddits(config.DEFAULT_SUBREDDITS, limit=50)

    print(f"Scanned subreddits: {config.DEFAULT_SUBREDDITS}")
    print(f"Flagged {len(flagged)} likely-phishing posts:\n")
    for item in flagged:
        print(f"- [{item['score']}] {item['permalink']} ({', '.join(item['matched_categories'])})")


if __name__ == "__main__":
    main()
