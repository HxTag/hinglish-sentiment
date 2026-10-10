from src.preprocessing import clean_text

SETTINGS = {
    "lowercase": True, "remove_urls": True, "remove_mentions": True,
    "remove_html": True, "normalize_repeated_characters": True,
    "max_repeated_characters": 2, "emoji_to_text": True,
}


def test_removes_url_mentions_and_html() -> None:
    assert clean_text("<b>Nice</b> @friend https://example.com", SETTINGS) == "nice"


def test_converts_emoji_to_words() -> None:
    assert "smiling face" in clean_text("Mast 😊", SETTINGS)


def test_normalizes_repeated_characters() -> None:
    assert clean_text("sooooo mastttt", SETTINGS) == "soo mastt"


def test_handles_empty_values() -> None:
    assert clean_text(None, SETTINGS) == ""
    assert clean_text("   ", SETTINGS) == ""
