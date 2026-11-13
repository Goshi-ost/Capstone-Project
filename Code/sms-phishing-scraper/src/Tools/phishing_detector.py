"""Classify SMS text as phishing: an ML spam model plus phishing lure-type rules.

The ML model is trained on the UCI SMS Spam Collection (Resources/spam.csv), which
labels spam vs. ham. Much spam is plain marketing, so a message is only flagged as
phishing when the model says spam AND it matches a phishing lure category (delivery,
banking, toll, credentials, ...) or contains a suspicious URL. Matched categories also
give each message a type for the "categorize collected messages" analysis.
"""

import ipaddress
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import joblib

DATASET_PATH = Path(__file__).resolve().parents[4] / "Resources" / "spam.csv"
OCR_CSV_PATH = Path(__file__).resolve().parents[1] / "crv-files" / "List.crv"
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "text_classifier.joblib"
# Bump when training code changes so stale saved models are retrained.
MODEL_VERSION = 1

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "t.ly", "tiny.cc",
}
# TLDs heavily abused in smishing campaigns.
RISKY_TLDS = {
    "top", "xyz", "icu", "click", "live", "info", "vip", "cyou", "sbs", "cfd",
    "buzz", "rest", "bond", "life", "online", "site", "shop", "support",
}

# Phishing lure categories. Each pattern is matched case-insensitively on word boundaries.
PHISHING_CATEGORIES: dict[str, list[str]] = {
    "delivery": [
        r"package", r"parcel", r"shipment", r"re-?deliver\w*", r"delivery (?:attempt|failed|address)",
        r"usps", r"fedex", r"dhl", r"royal mail", r"post office", r"customs fee", r"tracking (?:number|id)",
    ],
    "banking": [
        r"bank", r"card ending", r"(?:unusual|suspicious) (?:activity|sign-?in|login)",
        r"account (?:(?:has been|is|was) )?(?:locked|suspended|frozen|restricted|on hold|disabled)",
        r"zelle", r"paypal", r"venmo", r"cash ?app", r"wire transfer", r"unauthori[sz]ed",
    ],
    "toll_fine": [
        r"tolls?", r"unpaid", r"outstanding (?:balance|amount|invoice)", r"traffic (?:fine|ticket|violation)",
        r"e-?z ?pass", r"sun ?pass", r"fastrak", r"parking (?:fine|ticket|violation)", r"late fee",
    ],
    "credentials": [
        r"verify", r"verification", r"confirm (?:your|the) (?:identity|account|details|information)",
        r"log ?in", r"password", r"one[- ]time (?:code|password|passcode)", r"otp", r"security code",
        r"update (?:your )?(?:payment|billing) (?:info|information|details|method)",
    ],
    "prize": [
        r"winner", r"you(?:'ve| have)? won", r"prize", r"reward points?", r"gift ?card",
        r"claim (?:your|now|it)", r"congratulations", r"selected (?:to|for)",
    ],
    "government": [
        r"irs", r"tax refund", r"hmrc", r"dmv", r"social security", r"medicare", r"stimulus",
        r"court (?:date|summons)", r"warrant",
    ],
    "crypto_investment": [
        r"bitcoin", r"crypto\w*", r"usdt", r"investment", r"trading (?:platform|account)",
        r"guaranteed (?:profit|returns?)",
    ],
    "job": [r"hiring", r"recruit\w*", r"remote (?:job|work|position)", r"part[- ]time", r"per (?:day|hour)", r"job offer"],
    "impersonation": [r"hi (?:mum|mom|dad)", r"(?:my )?new number", r"lost my phone", r"wrong number"],
    "urgency": [
        r"urgent", r"immediately", r"within \d+ (?:hours?|hrs|days?)", r"final (?:notice|reminder|warning)",
        r"act now", r"expires? (?:today|soon|in)", r"(?:will be|to avoid) (?:suspended|cancell?ed|closed|charged)",
    ],
}
# Urgency is a supporting signal, not a lure type: marketing spam is urgent too.
_NON_LURE_CATEGORIES = {"urgency"}
_CATEGORY_RES = {
    name: re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)
    for name, patterns in PHISHING_CATEGORIES.items()
}


@dataclass
class PhishingAnalysis:
    """Classification result, including the score and any suspicious URLs."""

    text: str
    score: int = 0
    matched_keywords: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    suspicious_urls: list[str] = field(default_factory=list)
    is_likely_phishing: bool = False


@dataclass
class EvaluationResult:
    """Holdout-set metrics and selected parameters from model tuning."""

    accuracy: float
    report: str
    confusion_matrix: list[list[int]]
    best_params: dict


# Find HTTP(S) URLs embedded in a message.
def _extract_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


# Flag link shorteners, raw-IP hosts, punycode lookalike domains, and abused TLDs.
def _is_suspicious_url(url: str) -> bool:
    host = url.lower().split("://", 1)[-1].split("/", 1)[0].split("@")[-1].split(":")[0]
    host = host.removeprefix("www.").rstrip(".")
    if host in SHORTENER_DOMAINS:
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if any(label.startswith("xn--") for label in host.split(".")):
        return True
    return host.rsplit(".", 1)[-1] in RISKY_TLDS


# Return the matched phishing categories and the keywords that triggered them.
def categorize(text: str) -> tuple[list[str], list[str]]:
    categories, keywords = [], []
    for name, pattern in _CATEGORY_RES.items():
        hits = {m.group(0).lower() for m in pattern.finditer(text)}
        if hits:
            categories.append(name)
            keywords.extend(sorted(hits))
    return categories, list(dict.fromkeys(keywords))


# Phishing = the model says spam AND there is a lure or a suspicious link, OR a lure plus a
# suspicious link regardless of the model (the 2012-era training data misses modern
# smishing such as toll and delivery scams).
def is_phishing(is_spam: bool, categories: list[str], suspicious_urls: list[str]) -> bool:
    has_lure = any(c not in _NON_LURE_CATEGORIES for c in categories)
    has_bad_link = bool(suspicious_urls)
    return (is_spam and (has_lure or has_bad_link)) or (has_lure and has_bad_link)


class PhishingDetector:
    """Tune an SMS spam classifier and retain a stratified 10% test set."""

    # Store classifier configuration and initialize model/evaluation state.
    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        random_state: int = 42,
        test_size: float = 0.1,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.random_state = random_state
        self.test_size = test_size
        self.model: Pipeline | None = None
        self.evaluation: EvaluationResult | None = None

    def _dataset_fingerprint(self) -> dict:
        stat = self.dataset_path.stat()
        return {
            "version": MODEL_VERSION,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "random_state": self.random_state,
            "test_size": self.test_size,
        }

    # Load a saved model when it matches the current dataset; otherwise train and save one.
    def load_or_train(self, model_path: Path = MODEL_PATH) -> None:
        model_path = Path(model_path)
        fingerprint = self._dataset_fingerprint()
        if model_path.exists():
            try:
                saved = joblib.load(model_path)
                if saved.get("fingerprint") == fingerprint:
                    self.model, self.evaluation = saved["model"], saved["evaluation"]
                    return
            except Exception as exc:  # corrupt file or incompatible scikit-learn version
                print(f"[warn] could not load saved model ({exc}); retraining")
        print("Training text classifier (one-time; saved for later runs)...")
        self.train_and_evaluate()
        self.save(model_path, fingerprint)

    def save(self, model_path: Path = MODEL_PATH, fingerprint: dict | None = None) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "fingerprint": fingerprint or self._dataset_fingerprint(),
                "model": self.model,
                "evaluation": self.evaluation,
            },
            model_path,
        )

    # Tune candidate text classifiers and evaluate the winner on held-out data.
    def train_and_evaluate(self) -> EvaluationResult:
        data = pd.read_csv(self.dataset_path, encoding="latin-1", usecols=[0, 1])
        data.columns = ["label", "text"]
        data = data.dropna(subset=["label", "text"])
        texts = data["text"].astype(str)
        labels = data["label"].astype(str).str.strip().str.lower()
        valid_labels = labels.isin({"ham", "spam"})
        texts, labels = texts[valid_labels], labels[valid_labels]

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(strip_accents="unicode", sublinear_tf=True)),
                ("classifier", LogisticRegression(max_iter=2000, random_state=self.random_state)),
            ]
        )
        parameter_grid = [
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [1, 2],
                "classifier": [MultinomialNB()],
                "classifier__alpha": [0.1, 0.5, 1.0],
            },
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [1, 2],
                "classifier": [LogisticRegression(max_iter=2000, random_state=self.random_state)],
                "classifier__C": [0.5, 1.0, 2.0, 4.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ]
        cross_validation = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="accuracy",
            cv=cross_validation,
            n_jobs=-1,
            refit=True,
        )
        search.fit(train_texts, train_labels)

        predictions = search.predict(test_texts)
        self.model = search.best_estimator_
        self.evaluation = EvaluationResult(
            accuracy=float(accuracy_score(test_labels, predictions)),
            report=classification_report(test_labels, predictions, labels=["ham", "spam"]),
            confusion_matrix=confusion_matrix(
                test_labels, predictions, labels=["ham", "spam"]
            ).tolist(),
            best_params=search.best_params_,
        )
        return self.evaluation

    # Classify a message: spam probability, phishing categories, and suspicious URLs.
    def analyze(self, text: str) -> PhishingAnalysis:
        if self.model is None:
            self.load_or_train()
        assert self.model is not None

        prediction = str(self.model.predict([text])[0])
        probabilities = self.model.predict_proba([text])[0]
        classes = list(self.model.classes_)
        spam_probability = float(probabilities[classes.index("spam")])
        urls = _extract_urls(text)
        suspicious_urls = [url for url in urls if _is_suspicious_url(url)]
        categories, keywords = categorize(text)
        return PhishingAnalysis(
            text=text,
            score=round(spam_probability * 100),
            matched_keywords=keywords,
            matched_categories=categories,
            urls=urls,
            suspicious_urls=suspicious_urls,
            is_likely_phishing=is_phishing(prediction == "spam", categories, suspicious_urls),
        )

    # Classify non-empty CSV text rows and analyze any URL in each photo_key field.
    def classify_csv(
        self, csv_path: Path = OCR_CSV_PATH
    ) -> list[tuple[str, PhishingAnalysis | None, PhishingAnalysis]]:
        data = pd.read_csv(csv_path, encoding="utf-8")
        if "text" not in data.columns:
            raise ValueError(f"CSV must contain a 'text' column: {csv_path}")

        predictions = []
        for _, row in data.iterrows():
            if pd.isna(row["text"]) or not str(row["text"]).strip():
                continue
            first_field = str(row.get("photo_key", ""))
            url_match = _extract_urls(first_field)
            url_analysis = self.analyze(url_match[0]) if url_match else None
            text_analysis = self.analyze(str(row["text"]))
            predictions.append((first_field, url_analysis, text_analysis))
        return predictions


_detector: PhishingDetector | None = None


# Classify text using a lazily created, shared phishing detector.
def analyze_text(text: str) -> PhishingAnalysis:
    global _detector
    if _detector is None:
        _detector = PhishingDetector()
    return _detector.analyze(text)


if __name__ == "__main__":
    detector = PhishingDetector()
    results = detector.train_and_evaluate()
    detector.save()
    print(f"Holdout accuracy: {results.accuracy:.2%}")
    print(f"Best parameters: {results.best_params}")
    print("Confusion matrix (rows: ham, spam; columns: ham, spam):")
    print(results.confusion_matrix)
    print(results.report)
    print(f"Predictions from {OCR_CSV_PATH}:")
    predictions = detector.classify_csv()
    if not predictions:
        print("No non-empty OCR text found.")
    for first_field, url_analysis, text_analysis in predictions:
        if url_analysis is None:
            print(f"[NO URL] {first_field}")
        else:
            url_label = "SPAM" if url_analysis.is_likely_phishing else "HAM"
            print(f"[URL {url_label}] {url_analysis.text}")
        text_label = "SPAM" if text_analysis.is_likely_phishing else "HAM"
        print(f"[TEXT {text_label}] {text_analysis.text}")
