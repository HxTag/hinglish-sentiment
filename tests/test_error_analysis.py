from src.error_analysis import categorize


def test_categories_cover_requested_heuristics() -> None:
    assert "very_short" in categorize("bakwas")
    assert "mixed_sentiment" in categorize("mast but bakwas")
    assert "possible_sarcasm_or_negation" in categorize("accha nahi hai")
    assert "heavy_spelling_variation" in categorize("sooooo bekar")
