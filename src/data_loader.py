"""Load, validate, clean, and stratify a Hinglish sentiment CSV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text

LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read experiment configuration from YAML."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalise_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def load_dataset(csv_path: str | Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Load a CSV and return cleaned text, canonical labels, and label IDs."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Add a real CSV or use --sample only for smoke testing."
        )
    frame = pd.read_csv(path)
    text_col, label_col = config["columns"]["text"], config["columns"]["label"]
    missing = {text_col, label_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")
    mapping = {_normalise_label(key): value for key, value in config["label_mapping"].items()}
    output = frame[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).copy()
    output["label"] = output["label"].map(_normalise_label).map(mapping)
    unknown = output["label"].isna()
    if unknown.any():
        raise ValueError(
            "Found labels not declared in config.label_mapping; add their explicit mapping. "
            f"Rows: {output.index[unknown].tolist()[:10]}"
        )
    output["text"] = output["text"].map(lambda item: clean_text(item, config["preprocessing"]))
    output = output[output["text"].ne("")].reset_index(drop=True)
    order = list(config["label_order"])
    unsupported = set(output["label"]).difference(order)
    if unsupported:
        raise ValueError(f"Mapped labels absent from label_order: {sorted(unsupported)}")
    output["label_id"] = pd.Categorical(output["label"], categories=order).codes
    return output


def stratified_splits(frame: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Create deterministic train, validation, and test stratified splits."""
    split = config["split"]
    train_fraction, val_fraction, test_fraction = (
        float(split["train"]), float(split["validation"]), float(split["test"])
    )
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")
    try:
        train, holdout = train_test_split(
            frame, train_size=train_fraction, stratify=frame["label"], random_state=int(config["seed"])
        )
        validation, test = train_test_split(
            holdout,
            train_size=val_fraction / (val_fraction + test_fraction),
            stratify=holdout["label"],
            random_state=int(config["seed"]),
        )
    except ValueError as exc:
        raise ValueError(
            "Stratified splitting failed. Each class needs enough rows for all three splits."
        ) from exc
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "test": test.reset_index(drop=True)}


def save_splits(splits: Mapping[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Persist split CSVs and log class distributions."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(destination / f"{name}.csv", index=False)
        LOGGER.info("%s (%d rows):\n%s", name, len(split), split["label"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", help="Path to a real user-provided CSV.")
    parser.add_argument("--sample", action="store_true", help="Use ONLY the meaningless smoke-test sample.")
    args = parser.parse_args()
    if args.input and args.sample:
        parser.error("Choose either --input or --sample, not both.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    csv_path = args.input or (config["paths"]["sample_data"] if args.sample else config["paths"]["raw_data"])
    if args.sample:
        LOGGER.warning("SMOKE TEST ONLY: bundled sample results are meaningless; do not report them.")
    dataset = load_dataset(csv_path, config)
    LOGGER.info("Loaded %d non-empty rows. Overall distribution:\n%s", len(dataset), dataset["label"].value_counts().to_string())
    save_splits(stratified_splits(dataset, config), config["paths"]["processed_dir"])


if __name__ == "__main__":
    main()
