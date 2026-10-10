"""Create one measured comparison table and chart from saved model metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_config

LOGGER = logging.getLogger(__name__)


def load_results(results_dir: Path) -> pd.DataFrame:
    """Collect only saved metrics from completed model runs."""
    rows: list[dict[str, Any]] = []
    baseline_path = results_dir / "baseline_metrics.csv"
    if baseline_path.exists():
        rows.extend(pd.read_csv(baseline_path).to_dict("records"))
    bilstm_path = results_dir / "bilstm_metrics.json"
    if bilstm_path.exists():
        metrics = json.loads(bilstm_path.read_text(encoding="utf-8"))
        rows.append({key: metrics[key] for key in ("model", "accuracy", "macro_f1", "training_seconds")})
    # Per-model JSON files are authoritative; quick-debug JSON files are ignored.
    for transformer_path in results_dir.glob("transformer_*_metrics.json"):
        row = json.loads(transformer_path.read_text(encoding="utf-8"))
        if not bool(row.get("quick_debug_run", False)):
            rows.append({key: row[key] for key in ("model", "accuracy", "macro_f1", "training_seconds")})
    if not rows:
        raise FileNotFoundError("No completed metrics found in results/.")
    return pd.DataFrame(rows).drop_duplicates(subset=["model"], keep="last").sort_values("macro_f1", ascending=False)


def plot_comparison(table: pd.DataFrame, output_path: Path) -> None:
    """Visualize held-out accuracy and macro-F1; time stays in the CSV table."""
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = range(len(table))
    axis.bar([value - 0.18 for value in positions], table["accuracy"], width=0.36, label="Accuracy")
    axis.bar([value + 0.18 for value in positions], table["macro_f1"], width=0.36, label="Macro-F1")
    axis.set(xticks=list(positions), xticklabels=table["model"], ylim=(0, 1), ylabel="Score", title="Held-out test-set model comparison")
    axis.legend()
    for index, value in enumerate(table["macro_f1"]):
        axis.text(index + 0.18, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    results_dir = Path(config["paths"]["results_dir"])
    table = load_results(results_dir)
    table.to_csv(results_dir / "model_comparison.csv", index=False)
    plot_comparison(table, results_dir / "model_comparison.png")
    LOGGER.info("Saved comparison for %d completed models.\n%s", len(table), table.to_string(index=False))


if __name__ == "__main__":
    main()
