"""Train and evaluate reproducible TF-IDF sentiment baselines."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from src.data_loader import load_config

LOGGER = logging.getLogger(__name__)


def build_features(config: Mapping[str, Any]) -> FeatureUnion:
    """Combine word meaning and character spelling features for noisy Hinglish."""
    settings = config["baseline"]
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(ngram_range=tuple(settings["word_ngram_range"]), max_features=int(settings["max_features"]), sublinear_tf=True)),
        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(settings["char_ngram_range"]), max_features=int(settings["max_features"]), sublinear_tf=True)),
    ])


def build_models(seed: int) -> dict[str, Any]:
    """Return class-balanced classical models with deterministic settings."""
    return {
        # liblinear avoids a SciPy optimizer-version warning and suits this small,
        # multiclass student-project baseline.
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed, solver="liblinear"),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=seed),
    }


def calculate_metrics(y_true: pd.Series, y_pred: Any, labels: list[str]) -> dict[str, Any]:
    """Calculate only metrics derived from supplied predictions."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "per_class": {label: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])} for index, label in enumerate(labels)},
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: Any, labels: list[str], model_name: str, output_path: Path) -> None:
    """Save an annotated test-set confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set(xlabel="Predicted label", ylabel="True label", title=f"{model_name} confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_baselines(config: Mapping[str, Any]) -> pd.DataFrame:
    """Fit TF-IDF baselines and evaluate once on held-out test data."""
    paths = config["paths"]
    processed = Path(paths["processed_dir"])
    train_path, test_path = processed / "train.csv", processed / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed splits. Run the Phase 1 data loader first.")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    for name, frame in {"train": train, "test": test}.items():
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"{name}.csv must contain text and label columns.")
    labels = list(config["label_order"])
    results_dir, models_dir = Path(paths["results_dir"]), Path(paths["models_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    features = build_features(config)
    LOGGER.info("Fitting word and character TF-IDF features on %d training rows.", len(train))
    train_features = features.fit_transform(train["text"])
    test_features = features.transform(test["text"])
    rows: list[dict[str, Any]] = []
    reports: list[str] = []
    all_metrics: dict[str, Any] = {}
    for name, model in build_models(int(config["seed"])).items():
        started = time.perf_counter()
        model.fit(train_features, train["label"])
        predictions = model.predict(test_features)
        elapsed = time.perf_counter() - started
        metrics = calculate_metrics(test["label"], predictions, labels)
        metrics["training_seconds"] = elapsed
        all_metrics[name] = metrics
        rows.append({"model": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "training_seconds": elapsed})
        reports.extend([f"\n{'=' * 72}\n{name}\n{'=' * 72}\n", classification_report(test["label"], predictions, labels=labels, zero_division=0)])
        save_confusion_matrix(test["label"], predictions, labels, name.replace("_", " ").title(), results_dir / f"{name}_confusion_matrix.png")
        with (models_dir / f"{name}.pkl").open("wb") as handle:
            pickle.dump({"features": features, "model": model, "labels": labels}, handle)
        LOGGER.info("%s: accuracy=%.4f macro-F1=%.4f time=%.2fs", name, metrics["accuracy"], metrics["macro_f1"], elapsed)
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    table.to_csv(results_dir / "baseline_metrics.csv", index=False)
    (results_dir / "baseline_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (results_dir / "baseline_classification_reports.txt").write_text("".join(reports), encoding="utf-8")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    table = run_baselines(load_config(args.config))
    LOGGER.info("Saved baseline artifacts.\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
