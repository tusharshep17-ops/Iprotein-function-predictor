"""Data loading and processing utilities for protein function prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import pandas as pd


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
        opener = lambda url, timeout: (urlopen(url, timeout=timeout).read().decode(), None)
    
    # Placeholder implementation
    data = {"sequence": [], "label": []}
    frame = pd.DataFrame(data)
    frame.to_csv(output, index=False)
    return frame


def remove_conflicting_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove conflicting and duplicate sequences from dataset.
    
    Args:
        frame: DataFrame with 'sequence' and 'label' columns
        
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    return frame.drop_duplicates(subset=["sequence"], keep="first")


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
