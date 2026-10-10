"""Export and summarize misclassified held-out examples for completed models."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.bilstm import BiLSTMSentiment, TextDataset, Vocabulary, collate_batch
from src.data_loader import load_config

# Error analysis loads PyTorch transformers only; avoid unrelated Colab
# TensorFlow/protobuf import conflicts when a transformer checkpoint is present.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

LOGGER = logging.getLogger(__name__)
POSITIVE_WORDS = {"accha", "achhi", "awesome", "best", "badiya", "great", "mast", "love", "super", "zabardast"}
NEGATIVE_WORDS = {"bakwas", "bad", "bekar", "boring", "hate", "kharab", "worst", "waste", "disappointing"}
NEGATION_WORDS = {"nahi", "nahin", "not", "na", "never", "mat"}
SARCASM_MARKERS = {"lol", "haha", "wow", "sure", "great", "wah"}


def softmax(values: np.ndarray) -> np.ndarray:
    """Stable softmax for models that expose uncalibrated decision scores."""
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def categorize(text: str) -> str:
    """Return transparent heuristic categories, not claims about ground truth."""
    tokens = text.lower().split()
    categories: list[str] = []
    if len(tokens) <= 3:
        categories.append("very_short")
    if POSITIVE_WORDS.intersection(tokens) and NEGATIVE_WORDS.intersection(tokens):
        categories.append("mixed_sentiment")
    if NEGATION_WORDS.intersection(tokens) or (SARCASM_MARKERS.intersection(tokens) and "!" in text):
        categories.append("possible_sarcasm_or_negation")
    if re.search(r"(.)\1{2,}", text.lower()) or sum(not character.isascii() for character in text) > 3:
        categories.append("heavy_spelling_variation")
    return ";".join(categories) if categories else "other"


def baseline_predictions(model_path: Path, texts: pd.Series) -> tuple[list[str], np.ndarray]:
    """Predict from a saved scikit-learn baseline bundle."""
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    features, model, labels = bundle["features"], bundle["model"], bundle["labels"]
    matrix = features.transform(texts)
    predicted = model.predict(matrix).tolist()
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
    else:
        probabilities = softmax(model.decision_function(matrix))
    ordered = np.zeros((len(texts), len(labels)))
    for source_index, label in enumerate(model.classes_):
        ordered[:, labels.index(label)] = probabilities[:, source_index]
    return predicted, ordered


def bilstm_predictions(model_path: Path, texts: pd.Series, config: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Load the best BiLSTM checkpoint and return labels plus softmax confidence."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    labels, settings = checkpoint["labels"], checkpoint["settings"]
    vocabulary = Vocabulary(checkpoint["vocabulary"])
    model = BiLSTMSentiment(len(vocabulary), int(settings["embedding_dim"]), int(settings["hidden_dim"]), int(settings["num_layers"]), float(settings["dropout"]), len(labels))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frame = pd.DataFrame({"text": texts, "label": [labels[0]] * len(texts)})
    dataset = TextDataset(frame, vocabulary, {label: index for index, label in enumerate(labels)}, int(settings["max_length"]))
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=False, collate_fn=collate_batch)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for token_ids, lengths, _ in loader:
            logits = model(token_ids, lengths)
            probabilities.append(torch.softmax(logits, dim=1).numpy())
    combined = np.concatenate(probabilities)
    return [labels[index] for index in combined.argmax(axis=1)], combined


def transformer_predictions(model_path: Path, texts: pd.Series, labels: list[str]) -> tuple[list[str], np.ndarray]:
    """Load a local extracted transformer; no remote model download is attempted."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    probabilities: list[np.ndarray] = []
    for start in range(0, len(texts), 16):
        batch = texts.iloc[start : start + 16].tolist()
        encoded = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            probabilities.append(torch.softmax(model(**encoded).logits, dim=1).numpy())
    combined = np.concatenate(probabilities)
    return [labels[index] for index in combined.argmax(axis=1)], combined


def write_analysis(model_name: str, predicted: list[str], probabilities: np.ndarray, test: pd.DataFrame, results_dir: Path) -> None:
    """Write misclassifications and a concise category summary."""
    frame = test[["text", "label"]].rename(columns={"label": "true_label"}).copy()
    frame["predicted_label"] = predicted
    frame["confidence"] = probabilities.max(axis=1)
    errors = frame[frame["true_label"] != frame["predicted_label"]].copy()
    errors["category"] = errors["text"].map(categorize)
    errors.to_csv(results_dir / f"{model_name}_misclassified.csv", index=False)
    counts = errors["category"].str.get_dummies(sep=";").sum().sort_values(ascending=False)
    summary = [
        f"Error analysis for {model_name}",
        f"Misclassified: {len(errors)} of {len(frame)} test examples ({len(errors) / len(frame):.1%}).",
        "Heuristic categories (an example may appear in more than one):",
        *(f"- {category}: {count}" for category, count in counts.items()),
        "Categories are rule-based signals for manual review, not verified causes.",
    ]
    (results_dir / f"{model_name}_error_analysis_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "linear_svm", "bilstm", "xlm-roberta-base"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    paths, labels = config["paths"], list(config["label_order"])
    models_dir, results_dir = Path(paths["models_dir"]), Path(paths["results_dir"])
    test = pd.read_csv(Path(paths["processed_dir"]) / "test.csv")
    for name in args.models:
        try:
            if name in {"logistic_regression", "linear_svm"}:
                predicted, probabilities = baseline_predictions(models_dir / f"{name}.pkl", test["text"])
            elif name == "bilstm":
                predicted, probabilities = bilstm_predictions(models_dir / "bilstm_best.pt", test["text"], config)
            else:
                predicted, probabilities = transformer_predictions(models_dir / f"transformer_{name}", test["text"], labels)
            write_analysis(name.replace("/", "_"), predicted, probabilities, test, results_dir)
            LOGGER.info("Saved error analysis for %s.", name)
        except FileNotFoundError:
            LOGGER.warning("Skipping %s because its local saved model is missing.", name)


if __name__ == "__main__":
    main()
