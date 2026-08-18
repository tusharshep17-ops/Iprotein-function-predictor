from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from protfunc.constants import CANONICAL_AMINO_ACIDS, EC_CLASSES
from protfunc.model import load_artifact, predict_sequences, train_model


def make_toy_dataset() -> pd.DataFrame:
    """Create deterministic class-specific sequences for an end-to-end smoke test."""

    rows: list[dict[str, str]] = []
    alphabet = CANONICAL_AMINO_ACIDS
    motifs = ["ACD", "FGH", "IKL", "MNP", "QRS", "TVW", "YGA"]
    for class_index, label in enumerate(EC_CLASSES):
        motif = motifs[class_index]
        for sample_index in range(8):
            body = (motif * 22)[:60]
            replacement = alphabet[(class_index + sample_index) % len(alphabet)]
            position = 5 + sample_index
            sequence = body[:position] + replacement + body[position + 1 :]
            rows.append({"sequence": sequence, "label": label})
    return pd.DataFrame(rows)


class ModelTests(unittest.TestCase):
    def test_train_save_load_predict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            model_path = root / "model.joblib"
            reports_dir = root / "reports"
            frame = make_toy_dataset()
            frame.to_csv(data_path, index=False)

            metrics = train_model(
                data_path,
                model_path,
                reports_dir,
                test_size=0.25,
                seed=7,
                max_features=1_000,
            )
            self.assertTrue(model_path.exists())
            self.assertTrue((reports_dir / "metrics.json").exists())
            self.assertIn("macro_f1", metrics)

            artifact = load_artifact(model_path)
            predictions = predict_sequences(
                artifact, [frame.iloc[0]["sequence"]], top_k=3
            )
            self.assertEqual(len(predictions[0]), 3)
            probability_sum = sum(item["probability"] for item in predictions[0])
            self.assertGreater(probability_sum, 0)
            self.assertLessEqual(probability_sum, 1 + np.finfo(float).eps)


if __name__ == "__main__":
    unittest.main()

