import pandas as pd

from src.baseline import calculate_metrics


def test_metrics_have_expected_summary_and_classes() -> None:
    metrics = calculate_metrics(
        pd.Series(["positive", "negative", "neutral"]),
        ["positive", "negative", "neutral"],
        ["negative", "neutral", "positive"],
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert set(metrics["per_class"]) == {"negative", "neutral", "positive"}
