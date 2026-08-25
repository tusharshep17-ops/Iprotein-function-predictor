"""Sequence features used by the classical machine-learning baseline."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .constants import CANONICAL_AMINO_ACIDS

HYDROPHOBIC = frozenset("AVILMFWY")
AROMATIC = frozenset("FWY")
POSITIVE = frozenset("KRH")
NEGATIVE = frozenset("DE")
POLAR = frozenset("STNQCY")


def sequence_descriptors(sequence: str) -> np.ndarray:
    """Return composition and simple physicochemical descriptors."""

    length = len(sequence)
    counts = Counter(sequence)
    denominator = max(length, 1)
    composition = [counts[aa] / denominator for aa in CANONICAL_AMINO_ACIDS]
    observed = [value / denominator for value in counts.values() if value]
    entropy = -sum(probability * math.log2(probability) for probability in observed)

    summary = [
        math.log1p(length),
        entropy,
        sum(counts[aa] for aa in HYDROPHOBIC) / denominator,
        sum(counts[aa] for aa in AROMATIC) / denominator,
        sum(counts[aa] for aa in POSITIVE) / denominator,
        sum(counts[aa] for aa in NEGATIVE) / denominator,
        sum(counts[aa] for aa in POLAR) / denominator,
        (sum(counts[aa] for aa in POSITIVE) - sum(counts[aa] for aa in NEGATIVE))
        / denominator,
    ]
    return np.asarray(composition + summary, dtype=np.float64)


class ProteinFeatureExtractor(TransformerMixin, BaseEstimator):
    """Combine character k-mer TF-IDF with interpretable sequence descriptors."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 4),
        max_features: int = 25_000,
        min_df: int = 2,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df

    @staticmethod
    def _as_sequences(values: Iterable[str]) -> list[str]:
        return [str(value) for value in values]

    def fit(self, X: Iterable[str], y: object = None) -> ProteinFeatureExtractor:
        sequences = self._as_sequences(X)
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char",
            lowercase=False,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float64,
        )
        self.vectorizer_.fit(sequences)
        descriptors = np.vstack([sequence_descriptors(sequence) for sequence in sequences])
        self.scaler_ = StandardScaler()
        self.scaler_.fit(descriptors)
        self.n_features_in_ = 1
        return self

    def transform(self, X: Iterable[str]) -> sparse.csr_matrix:
        check_is_fitted(self, ["vectorizer_", "scaler_"])
        sequences = self._as_sequences(X)
        kmers = self.vectorizer_.transform(sequences)
        descriptors = np.vstack([sequence_descriptors(sequence) for sequence in sequences])
        scaled_descriptors = sparse.csr_matrix(self.scaler_.transform(descriptors))
        return sparse.hstack([kmers, scaled_descriptors], format="csr")

