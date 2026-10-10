import pandas as pd

from src.transformer_finetune import quick_subset, safe_model_name


def test_safe_model_name() -> None:
    assert safe_model_name("google/muril-base-cased") == "google_muril-base-cased"


def test_quick_subset_is_balanced() -> None:
    frame = pd.DataFrame({"label": ["negative"] * 3 + ["neutral"] * 3 + ["positive"] * 3})
    assert quick_subset(frame, 6, 42)["label"].value_counts().to_dict() == {"negative": 2, "neutral": 2, "positive": 2}
