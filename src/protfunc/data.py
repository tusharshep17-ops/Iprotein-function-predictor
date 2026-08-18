"""Dataset download and preparation utilities."""

from __future__ import annotations

import csv
import io
import random
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from .constants import EC_CLASSES, MAX_SEQUENCE_LENGTH, MIN_SEQUENCE_LENGTH
from .validation import SequenceValidationError, normalize_sequence

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FIELDS = "accession,id,protein_name,organism_name,length,ec,sequence"


class DataDownloadError(RuntimeError):
    """Raised when UniProt data cannot be downloaded or parsed."""


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def _open_url(url: str, timeout: int) -> tuple[str, str | None]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "rest.uniprot.org":
        raise DataDownloadError("Refusing to download data from a non-UniProt URL.")
    request = Request(
        url,
        headers={
            "Accept": "text/tab-separated-values",
            "User-Agent": "protfunc-ml/1.0 (educational research project)",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8")
            return body, response.headers.get("Link")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise DataDownloadError(f"UniProt request failed: {exc}") from exc


def _parse_uniprot_tsv(text: str, label: str, retrieved_at: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    expected = {
        "Entry",
        "Entry Name",
        "Protein names",
        "Organism",
        "Length",
        "EC number",
        "Sequence",
    }
    if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
        raise DataDownloadError(
            "Unexpected UniProt TSV schema. Received columns: " + repr(reader.fieldnames)
        )

    for row in reader:
        try:
            sequence = normalize_sequence(row["Sequence"])
        except SequenceValidationError:
            continue
        if not MIN_SEQUENCE_LENGTH <= len(sequence) <= MAX_SEQUENCE_LENGTH:
            continue
        accession = row["Entry"].strip()
        rows.append(
            {
                "accession": accession,
                "entry_name": row["Entry Name"].strip(),
                "protein_name": row["Protein names"].strip(),
                "organism": row["Organism"].strip(),
                "length": len(sequence),
                "ec_number": row["EC number"].strip(),
                "sequence": sequence,
                "label": label,
                "source_url": f"https://www.uniprot.org/uniprotkb/{accession}/entry",
                "retrieved_at": retrieved_at,
            }
        )
    return rows


def fetch_uniprot_dataset(
    output_path: str | Path,
    *,
    per_class: int = 500,
    seed: int = 42,
    timeout: int = 60,
    pause_seconds: float = 0.2,
    pool_multiplier: int = 2,
    opener: Callable[[str, int], tuple[str, str | None]] = _open_url,
) -> pd.DataFrame:
    """Download a balanced reviewed UniProtKB dataset for EC classes 1-7.

    The function queries reviewed entries, gathers a larger candidate pool,
    samples deterministically, removes duplicates and conflicting labels, and
    writes the resulting CSV.
    """

    if per_class < 10:
        raise ValueError("per_class must be at least 10.")
    if pool_multiplier < 1:
        raise ValueError("pool_multiplier must be at least 1.")

    retrieved_at = datetime.now(UTC).isoformat()
    rng = random.Random(seed)
    all_rows: list[dict[str, object]] = []

    for label, details in EC_CLASSES.items():
        ec_class = details["number"]
        query = f"(reviewed:true) AND (ec:{ec_class}.*)"
        parameters = {
            "query": query,
            "format": "tsv",
            "fields": UNIPROT_FIELDS,
            "size": 500,
        }
        next_url: str | None = f"{UNIPROT_SEARCH_URL}?{urlencode(parameters)}"
        candidates: list[dict[str, object]] = []
        target_pool = per_class * pool_multiplier

        while next_url and len(candidates) < target_pool:
            text, link_header = opener(next_url, timeout)
            candidates.extend(_parse_uniprot_tsv(text, label, retrieved_at))
            next_url = _next_link(link_header)
            if next_url and len(candidates) < target_pool and pause_seconds:
                time.sleep(pause_seconds)

        if len(candidates) < per_class:
            raise DataDownloadError(
                f"Only {len(candidates)} valid entries were returned for {label}; "
                f"requested {per_class}. Try a smaller --per-class value."
            )
        all_rows.extend(rng.sample(candidates, per_class))

    frame = pd.DataFrame(all_rows)
    frame = remove_conflicting_sequences(frame)
    frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def remove_conflicting_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate sequences and any sequence assigned to multiple labels."""

    if frame.empty:
        return frame.copy()
    conflicts = frame.groupby("sequence")["label"].nunique()
    conflicting_sequences = conflicts[conflicts > 1].index
    cleaned = frame[~frame["sequence"].isin(conflicting_sequences)]
    return cleaned.drop_duplicates(subset="sequence", keep="first").copy()


def load_training_data(path: str | Path) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Load, validate, and clean a CSV containing sequence and label columns."""

    frame = pd.read_csv(path)
    missing = {"sequence", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(sorted(missing))}")

    valid_rows: list[tuple[str, str]] = []
    for row_number, row in frame[["sequence", "label"]].iterrows():
        label = str(row["label"]).strip()
        if label not in EC_CLASSES:
            raise ValueError(f"Unknown label {label!r} on CSV row {row_number + 2}.")
        try:
            sequence = normalize_sequence(str(row["sequence"]))
        except SequenceValidationError as exc:
            raise ValueError(f"Invalid sequence on CSV row {row_number + 2}: {exc}") from exc
        valid_rows.append((sequence, label))

    cleaned = pd.DataFrame(valid_rows, columns=["sequence", "label"])
    cleaned = remove_conflicting_sequences(cleaned)
    counts = cleaned["label"].value_counts().sort_index().to_dict()
    if len(counts) < 2:
        raise ValueError("Training data must contain at least two function classes.")
    small_classes = {label: count for label, count in counts.items() if count < 4}
    if small_classes:
        raise ValueError(f"Each class needs at least 4 unique sequences; found {small_classes}.")
    return cleaned["sequence"], cleaned["label"], counts
