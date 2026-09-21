import re
from dataclasses import dataclass, field

URGENCY_KEYWORDS = [
    "urgent", "immediately", "act now", "act fast", "limited time", "expires today",
    "final notice", "last chance", "time-sensitive", "right away",
]

ACCOUNT_SCAM_KEYWORDS = [
    "verify your account", "confirm your identity", "account suspended",
    "account locked", "account will be closed", "unusual activity",
    "unauthorized access", "update your payment", "billing issue",
]

DELIVERY_SCAM_KEYWORDS = [
    "package", "delivery failed", "redelivery", "customs fee", "shipping fee",
    "tracking number", "unable to deliver",
]

FINANCIAL_SCAM_KEYWORDS = [
    "gift card", "wire transfer", "bitcoin", "crypto", "irs", "tax refund",
    "social security", "bank account", "claim your prize", "you have won",
    "lottery", "refund pending",
]

CREDENTIAL_HARVEST_KEYWORDS = [
    "click here", "click the link", "log in to verify", "enter your password",
    "confirm your ssn", "enter your card number",
]

ALL_KEYWORD_GROUPS = {
    "urgency": URGENCY_KEYWORDS,
    "account_scam": ACCOUNT_SCAM_KEYWORDS,
    "delivery_scam": DELIVERY_SCAM_KEYWORDS,
    "financial_scam": FINANCIAL_SCAM_KEYWORDS,
    "credential_harvest": CREDENTIAL_HARVEST_KEYWORDS,
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
}
IP_URL_PATTERN = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")


@dataclass
class PhishingAnalysis:
    text: str
    score: int = 0
    matched_keywords: list = field(default_factory=list)
    matched_categories: list = field(default_factory=list)
    urls: list = field(default_factory=list)
    suspicious_urls: list = field(default_factory=list)
    is_likely_phishing: bool = False


def _extract_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


def _is_suspicious_url(url: str) -> bool:
    if IP_URL_PATTERN.match(url):
        return True
    stripped = url.lower().replace("https://", "").replace("http://", "").replace("www.", "")
    domain = stripped.split("/")[0]
    return domain in SHORTENER_DOMAINS


def analyze_text(text: str, threshold: int = 3) -> PhishingAnalysis:
    """Heuristic phishing scorer. Not a substitute for a trained ML model,
    but a reasonable first pass for flagging likely SMS phishing content."""
    lowered = text.lower()
    analysis = PhishingAnalysis(text=text)

    for category, keywords in ALL_KEYWORD_GROUPS.items():
        for kw in keywords:
            if kw in lowered:
                analysis.matched_keywords.append(kw)
                if category not in analysis.matched_categories:
                    analysis.matched_categories.append(category)
                analysis.score += 1

    urls = _extract_urls(text)
    analysis.urls = urls
    for url in urls:
        if _is_suspicious_url(url):
            analysis.suspicious_urls.append(url)
            analysis.score += 2

    analysis.is_likely_phishing = analysis.score >= threshold
    return analysis
