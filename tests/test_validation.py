from __future__ import annotations

import unittest

from protfunc.validation import SequenceValidationError, normalize_sequence, parse_fasta


class ValidationTests(unittest.TestCase):
    def test_normalizes_case_and_whitespace(self) -> None:
        sequence = "acdef ghiklm\nnpqrstvwyacdef"
        self.assertEqual(normalize_sequence(sequence), "ACDEFGHIKLMNPQRSTVWYACDEF")

    def test_rejects_invalid_characters(self) -> None:
        with self.assertRaisesRegex(SequenceValidationError, "Invalid amino-acid"):
            normalize_sequence("ACDEFGHIKLMNPQRSTVWY123")

    def test_parses_multiple_fasta_records(self) -> None:
        text = ">alpha description\nACDEFGHIKLMNPQRSTVWY\n>beta\nYWVTSRQPNMLKIHGFEDCA"
        records = parse_fasta(text)
        self.assertEqual([record.identifier for record in records], ["alpha", "beta"])
        self.assertEqual(records[0].sequence, "ACDEFGHIKLMNPQRSTVWY")

    def test_plain_text_becomes_one_record(self) -> None:
        records = parse_fasta("ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(records[0].identifier, "query_1")


if __name__ == "__main__":
    unittest.main()

