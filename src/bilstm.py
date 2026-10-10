"""Train a class-balanced BiLSTM sentiment classifier with early stopping."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from src.data_loader import load_config

LOGGER = logging.getLogger(__name__)
PAD_TOKEN, OOV_TOKEN = "<pad>", "<unk>"


def set_seed(seed: int) -> None:
    """Set seeds so CPU runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Vocabulary:
    """A train-only whitespace vocabulary with explicit padding and OOV IDs."""

    def __init__(self, tokens_to_id: dict[str, int]) -> None:
        self.tokens_to_id = tokens_to_id

    @classmethod
    def build(cls, texts: Iterable[str], min_frequency: int, max_size: int) -> "Vocabulary":
        counts = Counter(token for text in texts for token in text.split())
        ordered = sorted(
            (item for item in counts.items() if item[1] >= min_frequency),
            key=lambda item: (-item[1], item[0]),
        )
        tokens_to_id = {PAD_TOKEN: 0, OOV_TOKEN: 1}
        for token, _ in ordered[: max_size - len(tokens_to_id)]:
            tokens_to_id[token] = len(tokens_to_id)
        return cls(tokens_to_id)

    def encode(self, text: str, max_length: int) -> list[int]:
        return [self.tokens_to_id.get(token, 1) for token in text.split()[:max_length]] or [1]

    def __len__(self) -> int:
        return len(self.tokens_to_id)


class TextDataset(Dataset[tuple[list[int], int]]):
    """Encoded text/label rows used by the PyTorch DataLoader."""

    def __init__(self, frame: pd.DataFrame, vocabulary: Vocabulary, label_to_id: Mapping[str, int], max_length: int) -> None:
        self.examples = [
            (vocabulary.encode(text, max_length), label_to_id[label])
            for text, label in zip(frame["text"], frame["label"], strict=True)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.examples[index]


def collate_batch(batch: list[tuple[list[int], int]]) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a batch only to its longest sequence to limit unnecessary compute."""
    sequences, labels = zip(*batch, strict=True)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = torch.zeros((len(sequences), int(lengths.max())), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class BiLSTMSentiment(nn.Module):
    """Embedding + bidirectional LSTM + dense sentiment classifier."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation))


def evaluate(model: nn.Module, loader: DataLoader[Any], loss_function: nn.Module, device: torch.device, labels: list[str]) -> tuple[float, dict[str, float], list[str], list[str]]:
    """Evaluate loss and macro-F1 without updating the model."""
    model.eval()
    total_loss, count = 0.0, 0
    predictions: list[str] = []
    actual: list[str] = []
    with torch.no_grad():
        for tokens, lengths, targets in loader:
            logits = model(tokens.to(device), lengths)
            loss = loss_function(logits, targets.to(device))
            total_loss += loss.item() * len(targets)
            count += len(targets)
            predictions.extend(labels[index] for index in logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels[index] for index in targets.tolist())
    return total_loss / count, {
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_f1": float(f1_score(actual, predictions, labels=labels, average="macro", zero_division=0)),
    }, actual, predictions


def save_plots(history: dict[str, list[float]], actual: list[str], predicted: list[str], labels: list[str], results_dir: Path) -> None:
    """Save learning curves and a test-set confusion matrix."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["validation_loss"], label="validation")
    axes[0].set(title="BiLSTM loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history["train_f1"], label="train")
    axes[1].plot(history["validation_f1"], label="validation")
    axes[1].set(title="BiLSTM macro-F1", xlabel="Epoch", ylabel="Macro-F1")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(results_dir / "bilstm_training_curves.png", dpi=160)
    plt.close(figure)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set(title="BiLSTM confusion matrix", xlabel="Predicted label", ylabel="True label")
    figure.tight_layout()
    figure.savefig(results_dir / "bilstm_confusion_matrix.png", dpi=160)
    plt.close(figure)


def run_training(config: Mapping[str, Any]) -> dict[str, Any]:
    """Train with validation early stopping, then evaluate best checkpoint once."""
    set_seed(int(config["seed"]))
    settings, paths, labels = config["bilstm"], config["paths"], list(config["label_order"])
    processed = Path(paths["processed_dir"])
    train, validation, test = (pd.read_csv(processed / f"{name}.csv") for name in ("train", "validation", "test"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    vocabulary = Vocabulary.build(train["text"], int(settings["min_frequency"]), int(settings["max_vocab_size"]))
    LOGGER.info("Vocabulary has %d tokens; it was built only from training text.", len(vocabulary))
    datasets = [TextDataset(frame, vocabulary, label_to_id, int(settings["max_length"])) for frame in (train, validation, test)]
    loaders = [
        DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=index == 0, collate_fn=collate_batch)
        for index, dataset in enumerate(datasets)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training BiLSTM on %s.", device)
    model = BiLSTMSentiment(len(vocabulary), int(settings["embedding_dim"]), int(settings["hidden_dim"]), int(settings["num_layers"]), float(settings["dropout"]), len(labels)).to(device)
    counts = train["label"].value_counts()
    weights = torch.tensor([len(train) / (len(labels) * counts[label]) for label in labels], dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    history = {"train_loss": [], "validation_loss": [], "train_f1": [], "validation_f1": []}
    models_dir, results_dir = Path(paths["models_dir"]), Path(paths["results_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "bilstm_best.pt"
    best_f1, stale_epochs, started = -1.0, 0, time.perf_counter()
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        running_loss, seen = 0.0, 0
        train_actual, train_predicted = [], []
        for tokens, lengths, targets in loaders[0]:
            optimizer.zero_grad()
            logits = model(tokens.to(device), lengths)
            loss = loss_function(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)
            seen += len(targets)
            train_predicted.extend(labels[index] for index in logits.argmax(dim=1).detach().cpu().tolist())
            train_actual.extend(labels[index] for index in targets.tolist())
        train_loss = running_loss / seen
        train_f1 = float(f1_score(train_actual, train_predicted, labels=labels, average="macro", zero_division=0))
        validation_loss, validation_metrics, _, _ = evaluate(model, loaders[1], loss_function, device, labels)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_f1"].append(train_f1)
        history["validation_f1"].append(validation_metrics["macro_f1"])
        LOGGER.info("Epoch %d: train loss %.4f F1 %.4f | validation loss %.4f F1 %.4f", epoch, train_loss, train_f1, validation_loss, validation_metrics["macro_f1"])
        if validation_metrics["macro_f1"] > best_f1:
            best_f1, stale_epochs = validation_metrics["macro_f1"], 0
            torch.save({"model_state": model.state_dict(), "vocabulary": vocabulary.tokens_to_id, "labels": labels, "settings": dict(settings), "best_validation_macro_f1": best_f1}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(settings["patience"]):
                LOGGER.info("Early stopping after %d epochs without validation macro-F1 improvement.", stale_epochs)
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_metrics, actual, predicted = evaluate(model, loaders[2], loss_function, device, labels)
    elapsed = time.perf_counter() - started
    save_plots(history, actual, predicted, labels, results_dir)
    report = classification_report(actual, predicted, labels=labels, zero_division=0)
    (results_dir / "bilstm_classification_report.txt").write_text(report, encoding="utf-8")
    metrics = {"model": "bilstm", **test_metrics, "test_loss": test_loss, "best_validation_macro_f1": best_f1, "training_seconds": elapsed, "epochs_trained": len(history["train_loss"]), "vocabulary_size": len(vocabulary)}
    (results_dir / "bilstm_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Test accuracy %.4f | macro-F1 %.4f | %.1fs", test_metrics["accuracy"], test_metrics["macro_f1"], elapsed)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
