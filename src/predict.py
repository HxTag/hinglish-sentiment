"""Reusable single-text inference for the Streamlit app and batch predictions."""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.bilstm import BiLSTMSentiment, Vocabulary
from src.data_loader import load_config
from src.preprocessing import clean_text

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


@dataclass(frozen=True)
class Prediction:
    """A validated human-readable sentiment prediction."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str


def safe_model_name(name: str) -> str:
    """Map Hugging Face identifiers to their local artifact directory name."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def available_models(config_path: str | Path = "config.yaml") -> dict[str, Path]:
    """Return only model choices whose required local artifacts exist."""
    config = load_config(config_path)
    models_dir = Path(config["paths"]["models_dir"])
    available: dict[str, Path] = {}
    for name in ("logistic_regression", "linear_svm"):
        path = models_dir / f"{name}.pkl"
        if path.exists():
            available[name] = path
    checkpoint = models_dir / "bilstm_best.pt"
    if checkpoint.exists():
        available["bilstm"] = checkpoint
    for identifier in config["transformer"]["model_names"]:
        path = models_dir / f"transformer_{safe_model_name(identifier)}"
        if (path / "config.json").exists():
            available[identifier] = path
    return available


def _ordered_probabilities(raw: np.ndarray, source_labels: list[str], labels: list[str]) -> dict[str, float]:
    """Align probability columns with the configured canonical class order."""
    output = {label: 0.0 for label in labels}
    for index, label in enumerate(source_labels):
        if label in output:
            output[label] = float(raw[0, index])
    total = sum(output.values())
    if total <= 0:
        raise ValueError("Model returned invalid confidence scores.")
    return {label: value / total for label, value in output.items()}


def load_predictor(model_name: str, config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load one locally saved model without making any network requests."""
    config = load_config(config_path)
    available = available_models(config_path)
    if model_name not in available:
        raise FileNotFoundError(f"Model {model_name!r} is not available locally.")
    path, labels = available[model_name], list(config["label_order"])
    if model_name in {"logistic_regression", "linear_svm"}:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        return {"type": "baseline", "features": bundle["features"], "model": bundle["model"], "labels": bundle["labels"]}
    if model_name == "bilstm":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        settings = checkpoint["settings"]
        vocabulary = Vocabulary(checkpoint["vocabulary"])
        model = BiLSTMSentiment(len(vocabulary), int(settings["embedding_dim"]), int(settings["hidden_dim"]), int(settings["num_layers"]), float(settings["dropout"]), len(checkpoint["labels"]))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return {"type": "bilstm", "model": model, "vocabulary": vocabulary, "labels": checkpoint["labels"], "max_length": int(settings["max_length"])}
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    model.eval()
    return {"type": "transformer", "model": model, "tokenizer": tokenizer, "labels": labels, "max_length": int(config["transformer"]["max_length"])}


def predict_text(text: str, predictor: dict[str, Any], config_path: str | Path = "config.yaml") -> Prediction:
    """Clean text with the shared function and return a label plus confidences."""
    config = load_config(config_path)
    cleaned = clean_text(text, config["preprocessing"])
    if not cleaned:
        raise ValueError("Enter text containing at least one word after cleaning.")
    labels = list(config["label_order"])
    if predictor["type"] == "baseline":
        matrix = predictor["features"].transform([cleaned])
        model = predictor["model"]
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(matrix)
        else:
            scores = model.decision_function(matrix)
            shifted = scores - scores.max(axis=1, keepdims=True)
            raw = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        probabilities = _ordered_probabilities(raw, list(model.classes_), labels)
    elif predictor["type"] == "bilstm":
        token_ids = predictor["vocabulary"].encode(cleaned, predictor["max_length"])
        with torch.no_grad():
            raw = torch.softmax(predictor["model"](torch.tensor([token_ids]), torch.tensor([len(token_ids)])), dim=1).numpy()
        probabilities = _ordered_probabilities(raw, predictor["labels"], labels)
    else:
        encoded = predictor["tokenizer"]([cleaned], truncation=True, max_length=predictor["max_length"], padding=True, return_tensors="pt")
        with torch.no_grad():
            raw = torch.softmax(predictor["model"](**encoded).logits, dim=1).numpy()
        probabilities = _ordered_probabilities(raw, predictor["labels"], labels)
    label = max(probabilities, key=probabilities.get)
    return Prediction(label, probabilities[label], probabilities, cleaned)
