import os
from dotenv import load_dotenv

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "sms-phishing-scraper")

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
if not TESSERACT_CMD and os.name == "nt":
	default_tesseract = os.path.join(
		os.environ.get("ProgramFiles", r"C:\Program Files"),
		"Tesseract-OCR",
		"tesseract.exe",
	)
	if os.path.exists(default_tesseract):
		TESSERACT_CMD = default_tesseract

# Subreddits known for people posting screenshots of scam/phishing SMS messages
DEFAULT_SUBREDDITS = ["Scams", "phishing", "IdentityTheft", "Scam"]

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.db")
IMAGE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
OCR_CSV_PATH = os.path.join(os.path.dirname(__file__), "List.crv")
