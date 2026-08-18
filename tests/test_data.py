from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from protfunc.data import fetch_uniprot_dataset, load_training_data, remove_conflicting_sequences


class DataTests(unittest.TestCase):
    def test_fetch_builds_balanced_csv_with_mocked_api(self) -> None:
        header = (
            "Entry\tEntry Name\tProtein names\tOrganism\tLength\tEC number\tSequence\n"
        )

        def fake_opener(url: str, timeout: int) -> tuple[str, None]:
            self.assertEqual(timeout, 11)
            query = parse_qs(urlparse(url).query)["query"][0]
            ec_class = int(query.split("ec:", maxsplit=1)[1].split(".", maxsplit=1)[0])
            records = []
            for index in range(10):
                prefix = "ACDEFGHIKLMNPQRSTVWY"
                tail = ("ACDEFGHIKLMNPQRSTVWY" * 2)[index : index + 20]
                sequence = prefix + tail + ("A" * ec_class) + ("C" * index)
                records.append(
                    f"P{ec_class:02d}{index:03d}\tENTRY_{ec_class}_{index}\tProtein "
                    f"{index}\tTest organism\t{len(sequence)}\t{ec_class}.1.1.1\t{sequence}"
                )
            return header + "\n".join(records) + "\n", None

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset.csv"
            frame = fetch_uniprot_dataset(
                output,
                per_class=10,
                seed=3,
                timeout=11,
                pause_seconds=0,
                opener=fake_opener,
            )
            self.assertTrue(output.exists())
            self.assertEqual(len(frame), 70)
            self.assertTrue((frame["label"].value_counts() == 10).all())

    def test_removes_conflicts_and_exact_duplicates(self) -> None:
        first = "ACDEFGHIKLMNPQRSTVWYACDEF"
        second = "YWVTSRQPNMLKIHGFEDCAYWVTS"
        frame = pd.DataFrame(
            {
                "sequence": [first, first, second, second],
                "label": [
                    "EC1_Oxidoreductase",
                    "EC1_Oxidoreductase",
                    "EC2_Transferase",
                    "EC3_Hydrolase",
                ],
            }
        )
        cleaned = remove_conflicting_sequences(frame)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["sequence"], first)

    def test_load_rejects_unknown_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            pd.DataFrame(
                {"sequence": ["ACDEFGHIKLMNPQRSTVWYACDEF"], "label": ["not_a_class"]}
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Unknown label"):
                load_training_data(path)


if __name__ == "__main__":
    unittest.main()
