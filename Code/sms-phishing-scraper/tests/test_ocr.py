from src.Tools import ocr


def test_clean_repairs_spaced_out_urls():
    assert "http://bit.ly/abc" in ocr.clean_ocr_text("Visit http : // bit . ly / abc now")


def test_clean_defangs_hxxp():
    assert "http://evil.top" in ocr.clean_ocr_text("hxxp://evil.top")


def test_clean_drops_leading_status_bar():
    text = "9:41 LTE 87%\nUSPS: your package is on hold"
    assert ocr.clean_ocr_text(text) == "USPS: your package is on hold"


def test_clean_fixes_letters_inside_numbers():
    assert "1-833-763-2014" in ocr.clean_ocr_text("1-833-763-2O14")
    assert "5012" in ocr.clean_ocr_text("card ending 5O12")


def test_is_image_url_ignores_query_string():
    assert ocr.is_image_url("https://i.redd.it/abc.jpeg?width=640")
    assert not ocr.is_image_url("https://reddit.com/r/Scams/comments/abc")
