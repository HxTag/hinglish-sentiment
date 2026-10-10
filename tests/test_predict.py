import numpy as np

from src.predict import _ordered_probabilities


def test_prediction_output_has_valid_labels_and_distribution() -> None:
    probabilities = _ordered_probabilities(
        np.array([[0.2, 0.3, 0.5]]),
        ["negative", "neutral", "positive"],
        ["negative", "neutral", "positive"],
    )
    assert set(probabilities) == {"negative", "neutral", "positive"}
    assert sum(probabilities.values()) == 1.0
    assert max(probabilities, key=probabilities.get) == "positive"
