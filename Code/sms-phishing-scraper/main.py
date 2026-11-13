from src import config, ocr, public_scraper, storage


def main() -> None:
    storage.init_db()
    flagged = public_scraper.scan_all(limit=50)
    exported_images = ocr.export_database_images_to_csv()

    print(f"Scanned subreddits: {config.DEFAULT_SUBREDDITS}")
    print(f"Exported OCR text for {exported_images} images to {config.OCR_CSV_PATH}")
    print(f"Flagged {len(flagged)} likely-phishing posts:\n")
    for item in flagged:
        print(f"- [{item['score']}] {item['permalink']} ({', '.join(item['matched_categories'])})")


if __name__ == "__main__":
    main()
