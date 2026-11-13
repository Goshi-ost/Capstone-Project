import pytest

from src.Tools import phishing_detector as pd_


class FakeModel:
    """Stands in for the trained pipeline so tests don't need to train anything."""

    classes_ = ["ham", "spam"]

    def __init__(self, spam: bool):
        self.spam = spam

    def predict(self, texts):
        return ["spam" if self.spam else "ham"]

    def predict_proba(self, texts):
        return [[0.1, 0.9]] if self.spam else [[0.9, 0.1]]


def make_detector(spam: bool) -> pd_.PhishingDetector:
    detector = pd_.PhishingDetector()
    detector.model = FakeModel(spam)
    return detector


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://bit.ly/x", True),
        ("http://192.168.0.1/login", True),
        ("https://xn--pypal-4ve.com/", True),
        ("https://usps-redelivery.top/track", True),
        ("https://www.usps.com/track", False),
        ("https://example.com", False),
    ],
)
def test_suspicious_urls(url, expected):
    assert pd_._is_suspicious_url(url) is expected


def test_categorize_delivery_scam():
    categories, keywords = pd_.categorize(
        "USPS: Your package could not be delivered. Redeliver within 24 hours."
    )
    assert categories == ["delivery", "urgency"]
    assert "usps" in keywords and "within 24 hours" in keywords


def test_categorize_ignores_innocent_text():
    assert pd_.categorize("Running late, see you at dinner tonight")[0] == []


def test_marketing_spam_is_not_phishing():
    analysis = make_detector(spam=True).analyze("Huge sale this weekend! Act now, 50% off everything")
    assert analysis.matched_categories == ["urgency"]
    assert not analysis.is_likely_phishing


def test_spam_with_lure_is_phishing():
    analysis = make_detector(spam=True).analyze("Your bank account has been locked. Verify at http://x.co")
    assert {"banking", "credentials"} <= set(analysis.matched_categories)
    assert analysis.is_likely_phishing
    assert analysis.score == 90


def test_spam_with_suspicious_link_only_is_phishing():
    assert make_detector(spam=True).analyze("hey check this http://bit.ly/abc").is_likely_phishing


def test_ham_with_lure_but_normal_link_is_not_phishing():
    analysis = make_detector(spam=False).analyze("Your package is at the bank https://www.usps.com")
    assert not analysis.is_likely_phishing


def test_lure_plus_suspicious_link_overrides_model():
    analysis = make_detector(spam=False).analyze("Your E-ZPass toll is unpaid: https://ezpass-pay.top")
    assert analysis.is_likely_phishing


def test_saved_model_is_reused(tmp_path, monkeypatch):
    detector = make_detector(spam=True)
    model_path = tmp_path / "model.joblib"
    detector.save(model_path)

    fresh = pd_.PhishingDetector()
    monkeypatch.setattr(fresh, "train_and_evaluate", lambda: pytest.fail("should not retrain"))
    fresh.load_or_train(model_path)
    assert isinstance(fresh.model, FakeModel)
