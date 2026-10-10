"""Fine-tune configurable Hugging Face transformers for Hinglish sentiment."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data_loader import load_config

# This project uses PyTorch. Colab may contain a TensorFlow/protobuf combination
# that is unrelated to this workflow but can be imported by Transformers unless
# TensorFlow support is disabled before importing Transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

LOGGER = logging.getLogger(__name__)


def safe_model_name(name: str) -> str:
    """Make model identifiers safe as local file names."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def quick_subset(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Sample each sentiment class for a local debug run."""
    per_class = max(1, limit // frame["label"].nunique())
    samples = [
        group.sample(n=min(len(group), per_class), random_state=seed)
        for _, group in frame.groupby("label")
    ]
    return pd.concat(samples, ignore_index=True)


def run_one(model_name: str, config: dict[str, Any], quick: bool) -> dict[str, Any]:
    """Fine-tune one model using validation macro-F1 for checkpoint selection."""
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("Install Phase 4 dependencies with: python -m pip install -r requirements.txt") from exc

    gpu = torch.cuda.is_available()
    if not gpu and not quick:
        raise RuntimeError(
            "Full transformer fine-tuning is blocked on CPU because it is impractically slow. "
            "Use Colab/Kaggle GPU, or pass --quick for a small local debugging run."
        )
    settings, paths, labels = config["transformer"], config["paths"], list(config["label_order"])
    processed = Path(paths["processed_dir"])
    train, validation, test = (pd.read_csv(processed / f"{split}.csv") for split in ("train", "validation", "test"))
    if quick:
        train, validation, test = (quick_subset(frame, int(settings["quick_subset_size"]), int(config["seed"])) for frame in (train, validation, test))
        LOGGER.warning("QUICK DEBUG ONLY: metrics are not reportable.")
    label_to_id = {label: index for index, label in enumerate(labels)}
    for frame in (train, validation, test):
        frame["label_id"] = frame["label"].map(label_to_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare(frame: pd.DataFrame) -> Any:
        dataset = Dataset.from_pandas(frame[["text", "label_id"]], preserve_index=False).rename_column("label_id", "labels")
        return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=int(settings["max_length"])), batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = (prepare(frame) for frame in (train, validation, test))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label_to_id)
    class_counts = train["label"].value_counts()
    weights = torch.tensor([len(train) / (len(labels) * class_counts[label]) for label in labels], dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any) -> Any:
            targets = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.CrossEntropyLoss(weight=weights.to(outputs.logits.device))(outputs.logits, targets)
            return (loss, outputs) if return_outputs else loss

    def metrics(prediction: Any) -> dict[str, float]:
        logits, actual = prediction
        predicted = np.argmax(logits, axis=-1)
        return {"accuracy": float(accuracy_score(actual, predicted)), "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0))}

    key = safe_model_name(model_name)
    artifact_key = f"{key}_quick" if quick else key
    model_dir, results_dir = Path(paths["models_dir"]) / f"transformer_{artifact_key}", Path(paths["results_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    if not gpu:
        LOGGER.warning("CPU quick-debug run: its metrics are not reportable.")
    arguments = TrainingArguments(
        output_dir=str(model_dir), learning_rate=float(settings["learning_rate"]),
        per_device_train_batch_size=int(settings["batch_size"]), per_device_eval_batch_size=int(settings["batch_size"]),
        num_train_epochs=float(settings["epochs"]), weight_decay=float(settings["weight_decay"]),
        evaluation_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        fp16=gpu, seed=int(config["seed"]), report_to=[], save_total_limit=1,
    )
    trainer = WeightedTrainer(
        model=model, args=arguments, train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(settings["patience"]))],
    )
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    prediction = trainer.predict(test_ds)
    score = metrics((prediction.predictions, prediction.label_ids))
    score.update({"model": model_name, "training_seconds": elapsed, "quick_debug_run": quick})
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"transformer_{artifact_key}_metrics.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    actual = [labels[index] for index in prediction.label_ids]
    predicted = [labels[index] for index in np.argmax(prediction.predictions, axis=-1)]
    (results_dir / f"transformer_{artifact_key}_classification_report.txt").write_text(classification_report(actual, predicted, labels=labels, zero_division=0), encoding="utf-8")
    pd.DataFrame(confusion_matrix(actual, predicted, labels=labels), index=labels, columns=labels).to_csv(results_dir / f"transformer_{artifact_key}_confusion_matrix.csv")
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    LOGGER.info("%s: accuracy %.4f | macro-F1 %.4f | %.1fs", model_name, score["accuracy"], score["macro_f1"], elapsed)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    outcomes = [run_one(name, config, args.quick) for name in (args.models or config["transformer"]["model_names"])]
    suffix = "_quick" if args.quick else ""
    pd.DataFrame(outcomes).to_csv(Path(config["paths"]["results_dir"]) / f"transformer_metrics{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
