# SMS Phishing Scraper (Reddit)

Scrapes Reddit posts (screenshots + text) from scam-reporting subreddits, runs OCR on
attached images, and scores the combined text for common SMS phishing patterns
(urgency language, account-suspension scams, delivery scams, financial/gift-card scams,
credential-harvesting links, suspicious/shortened URLs).

This is a research/detection heuristic tool, not a production anti-fraud system —
treat flagged results as candidates for review, not definitive verdicts.

## Setup

1. Create a Reddit API app at https://www.reddit.com/prefs/apps (choose "script" type)
   to get a client ID and secret. No paid tier needed.
2. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Windows installer)
   and note its install path if it's not on your PATH.
3. Copy `.env.example` to `.env` and fill in your Reddit credentials and (if needed)
   `TESSERACT_CMD`.
4. Install dependencies:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Run

```powershell
python main.py
```

Results are stored in `data/results.db` (SQLite). Flagged posts are also printed to the console.
The OCR export is written to `src/List.crv` with `photo_key` and cleaned `text` columns.

To add images for the OCR export, store their bytes in the SQLite `images` table:

```sql
INSERT INTO images (image_key, image_data, source)
VALUES ('photo-001', readfile('path/to/photo.png'), 'example');
```

The `image_key` is copied into the CSV as `photo_key`, keeping each extracted text value
linked to the image it came from.

## Project layout

- `src/config.py` — env config, default subreddits
- `src/ocr.py` — image download, text extraction, cleanup, and SQLite-to-CSV export
- `src/phishing_detector.py` — keyword/URL heuristic scorer
- `src/reddit_scraper.py` — pulls posts from subreddits via PRAW, extracts text/images
- `src/storage.py` — SQLite persistence
- `main.py` — entry point

## Notes on X (Twitter)

X's official API v2 has a free tier but is heavily rate-limited, and unofficial
scraping tools frequently break due to anti-bot enforcement and may violate X's
Terms of Service. This project currently targets Reddit only; adding X support
later is straightforward (a `src/x_scraper.py` following the same pattern) once
you decide on an API-based approach.
