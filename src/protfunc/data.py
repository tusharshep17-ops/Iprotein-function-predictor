"""Data loading and processing utilities for protein function prediction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import pandas as pd

from protfunc.constants import EC_CLASSES


def _default_opener(url: str, timeout: int) -> tuple[str, None]:
    """Default URL opener function for fetching UniProt data."""
    return urlopen(url, timeout=timeout).read().decode(), None


def fetch_uniprot_dataset(
    output: Path,
    per_class: int = 100,
    seed: int = 42,
    timeout: int = 10,
    pause_seconds: float = 0.1,
    opener: Callable[[str, int], tuple[str, None]] | None = None,
) -> pd.DataFrame:
    """Fetch UniProt dataset for protein function prediction.
    
    Args:
        output: Path to save the dataset CSV
        per_class: Number of sequences per EC class
        seed: Random seed for reproducibility
        timeout: Timeout for API requests
        pause_seconds: Pause between requests
        opener: Optional custom opener function for testing
        
    Returns:
        DataFrame with fetched sequences and labels
    """
    if opener is None:
        opener = _default_opener
    
    data = {"sequence": [], "label": []}
    
    for ec_class_num, ec_label in enumerate(EC_CLASSES, start=1):
        # Build UniProt query for this EC class
        query = f"ec:{ec_class_num}"
        url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=tsv&size={per_class}"
        
        # Fetch data using opener
        response_text, _ = opener(url, timeout)
        lines = response_text.strip().split("\n")
        
        # Parse TSV response (skip header)
        if len(lines) > 1:
            header_line = lines[0]
            record_lines = lines[1:]
            
            for line in record_lines[:per_class]:
                fields = line.split("\t")
                if len(fields) >= 7:  # Ensure we have all columns
                    sequence = fields[6]  # Sequence is column 7
                    data["sequence"].append(sequence)
                    data["label"].append(ec_label)
        
        time.sleep(pause_seconds)
    
    frame = pd.DataFrame(data)
    frame.to_csv(output, index=False)
    return frame


def remove_conflicting_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove conflicting and duplicate sequences from dataset.
    
    A sequence is conflicting if it has multiple different labels.
    After removing conflicting sequences, exact duplicates are also removed.
    
    Args:
        frame: DataFrame with 'sequence' and 'label' columns
        
    Returns:
        Cleaned DataFrame with conflicts and duplicates removed
    """
    # Find sequences with multiple different labels (conflicts)
    label_counts = frame.groupby('sequence')['label'].nunique()
    conflicting_sequences = label_counts[label_counts > 1].index
    
    # Remove rows with conflicting sequences
    df_cleaned = frame[~frame['sequence'].isin(conflicting_sequences)]
    
    # Remove exact duplicates
    df_cleaned = df_cleaned.drop_duplicates(subset=['sequence'], keep='first')
    
    return df_cleaned.reset_index(drop=True)


def load_training_data(path: Path | str) -> pd.DataFrame:
    """Load training data from CSV file.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame with training data
        
    Raises:
        ValueError: If unknown labels are found
    """
    frame = pd.read_csv(path)
    
    valid_labels = {
        "EC1_Oxidoreductase",
        "EC2_Transferase",
        "EC3_Hydrolase",
        "EC4_Lyase",
        "EC5_Isomerase",
        "EC6_Ligase",
        "EC7_Translocase",
    }
    
    unknown = set(frame["label"].unique()) - valid_labels
    if unknown:
        raise ValueError(f"Unknown label: {', '.join(unknown)}")
    
    return frame
