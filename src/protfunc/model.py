"""Training, evaluation, serialization, and inference."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .constants import EC_CLASSES
from .data import load_training_data
from .features import ProteinFeatureExtractor
from .validation import normalize_sequence

ARTIFACT_VERSION = 1


def build_pipeline(*, max_features: int = 25_000, random_state: int = 42) -> Pipeline:
    """Build the sequence-feature and classifier pipeline."""

    return Pipeline(
        steps=[
            (
                "features",
                ProteinFeatureExtractor(
                    ngram_range=(2, 4),
                    max_features=max_features,
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_model(
    data_path: str | Path,
    model_path: str | Path,
    reports_dir: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    max_features: int = 25_000,
) -> dict[str, Any]:
    """Train and evaluate the model, then save a versioned artifact and reports."""

    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")

    sequences, labels, class_counts = load_training_data(data_path)
    n_classes = labels.nunique()
    requested_test_count = max(n_classes, math.ceil(len(labels) * test_size))
    test_count = min(requested_test_count, len(labels) - n_classes)
    if test_count < n_classes:
        raise ValueError("Dataset is too small to put every class in both train and test sets.")

    X_train, X_test, y_train, y_test = train_test_split(
        sequences,
        labels,
        test_size=test_count,
        random_state=seed,
        stratify=labels,
    )

    pipeline = build_pipeline(max_features=max_features, random_state=seed)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    classes = list(pipeline.classes_)
    top_k = min(3, len(classes))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        f"top_{top_k}_accuracy": float(
            top_k_accuracy_score(y_test, probabilities, k=top_k, labels=classes)
        ),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_counts": {key: int(value) for key, value in class_counts.items()},
        "random_seed": seed,
        "test_size": test_size,
        "warning": (
            "This baseline uses a stratified random split. For publication-grade evaluation, "
            "replace it with sequence-similarity cluster splitting to control homology leakage."
        ),
    }

    report = classification_report(y_test, predictions, labels=classes, output_dict=True)
    matrix = confusion_matrix(y_test, predictions, labels=classes)
    created_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": created_at,
        "model": pipeline,
        "classes": classes,
        "class_metadata": {label: EC_CLASSES[label] for label in classes},
        "metrics": metrics,
        "training": {
            "data_path": str(data_path),
            "sklearn_version": sklearn.__version__,
            "max_features": max_features,
            "seed": seed,
        },
    }

    model_output = Path(model_path)
    report_output = Path(reports_dir)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output)

    (report_output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_output / "classification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        report_output / "confusion_matrix.csv", index_label="actual"
    )
    return metrics


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    """Load a trusted local model artifact and validate its basic schema.

    Never load an untrusted joblib/pickle file: deserialization can execute code.
    """

    artifact = joblib.load(model_path)
    required = {"artifact_version", "model", "classes", "class_metadata"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Model artifact is missing required fields.")
    if artifact["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {artifact['artifact_version']!r}; "
            f"expected {ARTIFACT_VERSION}."
        )
    return artifact


def predict_sequences(
    artifact: dict[str, Any], sequences: list[str], *, top_k: int = 3
) -> list[list[dict[str, Any]]]:
    """Return ranked predictions for each sequence."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    normalized = [normalize_sequence(sequence) for sequence in sequences]
    model = artifact["model"]
    classes = np.asarray(model.classes_)
    probabilities = model.predict_proba(normalized)
    limit = min(top_k, len(classes))
    results: list[list[dict[str, Any]]] = []

    for row in probabilities:
        order = np.argsort(row)[::-1][:limit]
        ranked: list[dict[str, Any]] = []
        for index in order:
            label = str(classes[index])
            metadata = artifact["class_metadata"].get(label, {})
            ranked.append(
                {
                    "label": label,
                    "name": metadata.get("name", label),
                    "description": metadata.get("description", ""),
                    "probability": float(row[index]),
                }
            )
        results.append(ranked)
    return results
