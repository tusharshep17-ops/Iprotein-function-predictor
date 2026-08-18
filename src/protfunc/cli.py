"""Command-line interface for data fetching, training, and prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .data import DataDownloadError, fetch_uniprot_dataset
from .model import load_artifact, predict_sequences, train_model
from .validation import SequenceValidationError, parse_fasta


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protfunc",
        description="Predict top-level enzyme classes from amino-acid sequences.",
    )
    parser.add_argument("--version", action="version", version="protfunc-ml 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Download a balanced reviewed UniProtKB dataset."
    )
    fetch_parser.add_argument("--output", type=Path, default=Path("data/uniprot_ec.csv"))
    fetch_parser.add_argument("--per-class", type=_positive_int, default=500)
    fetch_parser.add_argument("--seed", type=int, default=42)
    fetch_parser.add_argument("--timeout", type=_positive_int, default=60)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the classifier.")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--model", type=Path, default=Path("models/protfunc.joblib"))
    train_parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-features", type=_positive_int, default=25_000)

    predict_parser = subparsers.add_parser("predict", help="Predict one or more sequences.")
    predict_parser.add_argument("--model", type=Path, default=Path("models/protfunc.joblib"))
    input_group = predict_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--sequence", help="One plain amino-acid sequence.")
    input_group.add_argument(
        "--fasta", type=Path, help="A FASTA file containing one or more records."
    )
    predict_parser.add_argument("--top-k", type=_positive_int, default=3)
    predict_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _run_fetch(args: argparse.Namespace) -> int:
    frame = fetch_uniprot_dataset(
        args.output,
        per_class=args.per_class,
        seed=args.seed,
        timeout=args.timeout,
    )
    counts = frame["label"].value_counts().sort_index()
    print(f"Saved {len(frame):,} unique sequences to {args.output}")
    for label, count in counts.items():
        print(f"  {label}: {count:,}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    metrics = train_model(
        args.data,
        args.model,
        args.reports_dir,
        test_size=args.test_size,
        seed=args.seed,
        max_features=args.max_features,
    )
    print(f"Saved model to {args.model}")
    print(f"Accuracy:          {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Macro F1:          {metrics['macro_f1']:.3f}")
    return 0


def _prediction_payload(
    identifiers: list[str], ranked_predictions: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    return [
        {"identifier": identifier, "predictions": predictions}
        for identifier, predictions in zip(identifiers, ranked_predictions, strict=True)
    ]


def _run_predict(args: argparse.Namespace) -> int:
    text = args.sequence if args.sequence is not None else args.fasta.read_text(encoding="utf-8")
    records = parse_fasta(text)
    artifact = load_artifact(args.model)
    ranked = predict_sequences(artifact, [record.sequence for record in records], top_k=args.top_k)
    payload = _prediction_payload([record.identifier for record in records], ranked)

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for result in payload:
        print(f"\n{result['identifier']}")
        for rank, prediction in enumerate(result["predictions"], start=1):
            print(
                f"  {rank}. {prediction['name']:<18} "
                f"{prediction['probability'] * 100:6.2f}%  ({prediction['label']})"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            return _run_fetch(args)
        if args.command == "train":
            return _run_train(args)
        if args.command == "predict":
            return _run_predict(args)
    except (DataDownloadError, SequenceValidationError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
