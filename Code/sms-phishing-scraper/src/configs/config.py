import os
from dotenv import load_dotenv

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "sms-phishing-scraper")

# Identify the scraper honestly; sites like Reddit block or throttle spoofed browser agents.
HTTP_USER_AGENT = os.getenv(
	"HTTP_USER_AGENT", "sms-phishing-scraper/0.2 (academic capstone research project)"
)

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

# Keyless sources: (instance, hashtag) for Mastodon, (instance, search query) for Lemmy
MASTODON_TAGS = [("mastodon.social", "smishing"), ("mastodon.social", "scam"), ("infosec.exchange", "smishing")]
LEMMY_QUERIES = [("lemmy.world", "scam text"), ("lemmy.world", "smishing")]

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(SRC_DIR, "..", "data", "results.db")
IMAGE_CACHE_DIR = os.path.join(SRC_DIR, "..", "data", "images")
IMAGE_TEST_DIR = os.path.join(SRC_DIR, "imageTest")
OCR_CSV_PATH = os.path.join(SRC_DIR, "crv-files", "List.crv")

# Posts fetched per subreddit / hashtag / search query on each run.
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "50"))
