"""Protein sequence parsing and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .constants import ALLOWED_AMINO_ACIDS, MAX_SEQUENCE_LENGTH, MIN_SEQUENCE_LENGTH


class SequenceValidationError(ValueError):
    """Raised when a protein sequence is missing or malformed."""


@dataclass(frozen=True)
class FastaRecord:
    """A minimal FASTA record."""

    identifier: str
    sequence: str


def normalize_sequence(sequence: str, *, enforce_length: bool = True) -> str:
    """Normalize and validate one amino-acid sequence.

    Whitespace is removed and letters are upper-cased. Digits, punctuation,
    stop symbols, and non-IUPAC amino-acid letters are rejected.
    """

    if not isinstance(sequence, str) or not sequence.strip():
        raise SequenceValidationError("Sequence is empty.")

    normalized = re.sub(r"\s+", "", sequence).upper()
    invalid = sorted(set(normalized) - ALLOWED_AMINO_ACIDS)
    if invalid:
        raise SequenceValidationError(
            "Invalid amino-acid character(s): " + ", ".join(repr(char) for char in invalid)
        )

    if enforce_length and len(normalized) < MIN_SEQUENCE_LENGTH:
        raise SequenceValidationError(
            f"Sequence is too short ({len(normalized)} aa); minimum is {MIN_SEQUENCE_LENGTH}."
        )
    if len(normalized) > MAX_SEQUENCE_LENGTH:
        raise SequenceValidationError(
            f"Sequence is too long ({len(normalized)} aa); maximum is {MAX_SEQUENCE_LENGTH}."
        )
    return normalized


def parse_fasta(text: str) -> list[FastaRecord]:
    """Parse FASTA text, or treat plain text as a single sequence."""

    if not isinstance(text, str) or not text.strip():
        raise SequenceValidationError("No sequence input was provided.")

    stripped = text.strip()
    if not stripped.startswith(">"):
        return [FastaRecord(identifier="query_1", sequence=normalize_sequence(stripped))]

    records: list[FastaRecord] = []
    identifier: str | None = None
    chunks: list[str] = []

    def finish_record() -> None:
        if identifier is None:
            return
        if not chunks:
            raise SequenceValidationError(f"FASTA record {identifier!r} has no sequence.")
        records.append(
            FastaRecord(identifier=identifier, sequence=normalize_sequence("".join(chunks)))
        )

    for line_number, raw_line in enumerate(stripped.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish_record()
            identifier = line[1:].strip().split(maxsplit=1)[0]
            if not identifier:
                raise SequenceValidationError(f"Missing FASTA identifier on line {line_number}.")
            chunks = []
        else:
            if identifier is None:
                raise SequenceValidationError(
                    f"Sequence data appeared before a FASTA header on line {line_number}."
                )
            chunks.append(line)

    finish_record()
    if not records:
        raise SequenceValidationError("No FASTA records were found.")
    return records

