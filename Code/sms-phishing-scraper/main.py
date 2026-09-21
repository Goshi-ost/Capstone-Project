from src import config, ocr, reddit_scraper, storage


def main() -> None:
    storage.init_db()
    flagged = reddit_scraper.scan_subreddits(config.DEFAULT_SUBREDDITS, limit=50)
    exported_images = ocr.export_database_images_to_csv()

    print(f"Scanned subreddits: {config.DEFAULT_SUBREDDITS}")
    print(f"Exported OCR text for {exported_images} images to {config.OCR_CSV_PATH}")
    print(f"Flagged {len(flagged)} likely-phishing posts:\n")
    for item in flagged:
        print(f"- [{item['score']}] {item['permalink']} ({', '.join(item['matched_categories'])})")


if __name__ == "__main__":
    main()
