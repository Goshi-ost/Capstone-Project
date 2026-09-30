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

DATASET_PATH = Path(__file__).resolve().parents[3] / "Resources" / "spam.csv"
OCR_CSV_PATH = Path(__file__).with_name("List.crv")
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"}


@dataclass
class PhishingAnalysis:
    text: str
    score: int = 0
    matched_keywords: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    suspicious_urls: list[str] = field(default_factory=list)
    is_likely_phishing: bool = False


@dataclass
class EvaluationResult:
    accuracy: float
    report: str
    confusion_matrix: list[list[int]]
    best_params: dict


def _extract_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


def _is_suspicious_url(url: str) -> bool:
    domain = url.lower().split("://", 1)[-1].split("/", 1)[0].removeprefix("www.")
    return domain in SHORTENER_DOMAINS


class PhishingDetector:
    """Tune an SMS spam classifier and retain a stratified 10% test set."""

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

    def analyze(self, text: str) -> PhishingAnalysis:
        if self.model is None:
            self.train_and_evaluate()
        assert self.model is not None

        prediction = str(self.model.predict([text])[0])
        probabilities = self.model.predict_proba([text])[0]
        classes = list(self.model.classes_)
        spam_probability = float(probabilities[classes.index("spam")])
        urls = _extract_urls(text)
        return PhishingAnalysis(
            text=text,
            score=round(spam_probability * 100),
            matched_categories=["machine_learning"] if prediction == "spam" else [],
            urls=urls,
            suspicious_urls=[url for url in urls if _is_suspicious_url(url)],
            is_likely_phishing=prediction == "spam",
        )

    def classify_csv(
        self, csv_path: Path = OCR_CSV_PATH
    ) -> list[tuple[str, PhishingAnalysis | None, PhishingAnalysis]]:
        """Classify a URL in the first column and the text column separately."""
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


def analyze_text(text: str) -> PhishingAnalysis:
    """Classify text as spam/phishing using the tuned SMS model."""
    global _detector
    if _detector is None:
        _detector = PhishingDetector()
    return _detector.analyze(text)


if __name__ == "__main__":
    detector = PhishingDetector()
    results = detector.train_and_evaluate()
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
