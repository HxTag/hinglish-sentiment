"""Convert labelled SentiMix Hinglish CoNLL files into the project's CSV format."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterator

LOGGER = logging.getLogger(__name__)
VALID_LABELS = {"positive", "negative", "neutral"}


def read_conll(path: str | Path, split_name: str) -> Iterator[dict[str, str]]:
    """Yield one complete tweet per SentiMix CoNLL record.

    A record starts with: meta <tab> tweet-id <tab> sentiment.
    Remaining non-empty lines provide token and language-tag columns. Joining
    the token column recreates the text while preserving its code-mixed words.
    """
    record_id: str | None = None
    label: str | None = None
    tokens: list[str] = []

    def emit() -> dict[str, str] | None:
        if record_id is None:
            return None
        if label not in VALID_LABELS:
            raise ValueError(f"Unexpected sentiment {label!r} in {path}, tweet {record_id}.")
        text = " ".join(tokens).strip()
        if not text:
            raise ValueError(f"Empty tweet content in {path}, tweet {record_id}.")
        return {"text": text, "label": label, "source_split": split_name, "source_id": record_id}

    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if not line.strip():
                row = emit()
                if row:
                    yield row
                record_id, label, tokens = None, None, []
                continue
            fields = line.split("\t")
            if fields[0] == "meta":
                if record_id is not None:
                    row = emit()
                    if row:
                        yield row
                if len(fields) < 3:
                    raise ValueError(f"Malformed meta line in {path}: {line!r}")
                record_id, label, tokens = fields[1], fields[2].strip().lower(), []
            elif record_id is None:
                raise ValueError(f"Token appeared before a meta line in {path}: {line!r}")
            else:
                tokens.append(fields[0])
    row = emit()
    if row:
        yield row


def convert(train_path: str | Path, dev_path: str | Path, output_path: str | Path, overwrite: bool) -> int:
    """Write labelled train/dev records to a CSV and return its row count."""
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists. Use --overwrite only after checking it.")
    rows = [*read_conll(train_path, "official_train"), *read_conll(dev_path, "official_dev")]
    if not rows:
        raise ValueError("No examples were found in the supplied files.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "source_split", "source_id"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Hinglish_train_14k_split_conll.txt")
    parser.add_argument("--dev", required=True, help="Hinglish_dev_3k_split_conll.txt")
    parser.add_argument("--output", default="data/raw/hinglish_sentiment.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    count = convert(args.train, args.dev, args.output, args.overwrite)
    LOGGER.info("Converted %d labelled SentiMix tweets to %s", count, args.output)


if __name__ == "__main__":
    main()
