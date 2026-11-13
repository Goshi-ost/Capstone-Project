# SMS Phishing Scraper

Collects phishing SMS screenshots that people post publicly on social media, extracts
the message text with OCR, and classifies it as phishing or not, with a lure type
(delivery, banking, toll, credentials, prize, government, crypto, job, impersonation).

This is a research tool, not a production anti-fraud system. Treat flagged results as
candidates for review, not verdicts.

**Sources today:** Reddit (official API if credentials are set, else the public RSS feed),
Mastodon hashtag timelines, Lemmy search. X, Facebook and Instagram are not implemented yet.

## Setup

1. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Windows installer).
2. Copy `.env.example` to `.env` and fill it in. Reddit credentials come from
   https://www.reddit.com/prefs/apps (choose "script"). Without them, Reddit falls back
   to the RSS feed, which Reddit rate-limits heavily (HTTP 429).
3. Install dependencies:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r src/configs/requirements.txt
   ```

## Run

Run everything from this folder (`Code/sms-phishing-scraper`):

```powershell
python main.py                              # scrape, OCR, classify, store, export
python -m pytest                            # run the tests
python src/Tools/phishing_detector.py       # retrain the text model and print its evaluation
```

The first run trains the text model (a few minutes) and saves it to
`data/models/text_classifier.joblib`; later runs load it. It retrains automatically if
`Resources/spam.csv` changes.

## How a post is processed

`src/pipeline.py` handles every post the same way, whichever scraper found it:

1. **Skip** it if it is already in the database, before downloading anything.
2. **Download and OCR** its images. Images and their OCR text are stored together.
3. **Classify** the screenshot (OCR) text. The poster's own title/caption is only
   classified when there is no image text, because it usually *describes* a scam rather
   than being one. The `text_source` column records which was used.
4. **Store** the result, including the post's original publish time (`posted_at`).

### How "phishing" is decided

The ML model (TF-IDF + Naive Bayes / logistic regression, tuned with 5-fold
cross-validation, evaluated on a 10% holdout) is trained on the UCI SMS Spam Collection.
It detects *spam*, and much spam is marketing, so a message is flagged only when:

- the model says spam **and** the text matches a phishing lure category or has a
  suspicious link; **or**
- the text matches a lure category **and** has a suspicious link, whatever the model says.
  This is needed because the 2012-era training data misses modern smishing (it scores a
  typical E-ZPass toll scam at about 11% spam).

Suspicious links: URL shorteners, raw IP addresses, punycode (`xn--`) lookalike domains,
and TLDs commonly abused in smishing (`.top`, `.xyz`, `.icu`, ...). Lure categories and
their keywords are in `PHISHING_CATEGORIES` in `src/Tools/phishing_detector.py`.

**Known limitation:** the real fix is training on labeled modern phishing messages.

## Data

- `data/results.db` (SQLite): `results` (one row per post) and `images` (image bytes plus
  cached OCR text). Older databases are upgraded automatically on start.
- `src/crv-files/List.crv`: CSV export of every stored image's OCR text (`photo_key`, `text`).

## Project layout

- `main.py`: entry point
- `src/pipeline.py`: shared per-post pipeline, polite HTTP helper (honest User-Agent, 429 retry)
- `src/scrapers/reddit_scraper.py`: Reddit via the official API (PRAW)
- `src/scrapers/public_scraper.py`: keyless Reddit RSS, Mastodon and Lemmy
- `src/Tools/ocr.py`: image preprocessing, OCR, text cleanup, CSV export
- `src/Tools/phishing_detector.py`: text model, lure categories, URL checks
- `src/Tools/smsImageClassifier.py`: SMS-screenshot vs. other-image classifier (not trained or wired in yet)
- `src/Tools/storage.py`: SQLite storage
- `src/configs/config.py`: settings from `.env`, source lists, paths
- `tests/`: pytest suite

## Adding a platform

Add `src/scrapers/<platform>_scraper.py` that fetches posts and calls
`pipeline.process_post(source, post_id, permalink, text, image_urls, posted_at)` for each,
then call it from `main.py`. Dedupe, OCR, classification and storage come for free.
