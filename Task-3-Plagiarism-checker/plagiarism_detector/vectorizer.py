"""
vectorizer.py — turns cleaned text into numbers we can actually compare

Just a thin, opinionated wrapper around sklearn's TfidfVectorizer.
I tuned the defaults so they work well out of the box for essay-length documents.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class DocumentVectorizer:
    """
    Converts a corpus of pre-processed text documents into TF-IDF vectors.

    Parameters
    ----------
    ngram_range : tuple
        The lower and upper boundary of n-gram sizes (default ``(1, 2)``
        captures both unigrams and bigrams).
    max_features : int or None
        Vocabulary cap.  ``None`` means no cap (use all terms).
    min_df : int or float
        Minimum document frequency for a term to be kept.
    max_df : float
        Maximum document frequency (removes corpus-wide common terms).
    sublinear_tf : bool
        Apply sublinear TF scaling (``1 + log(tf)``), which dampens the
        impact of very frequent terms.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int | None = 10_000,
        min_df: int | float = 1,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit the vocabulary on *documents* and return their TF-IDF matrix.

        Returns
        -------
        np.ndarray of shape ``(n_docs, n_features)``
        """
        if not documents:
            raise ValueError("Cannot fit vectorizer on an empty corpus.")

        logger.debug("Fitting TF-IDF on %d documents.", len(documents))

        # For very small corpora, max_df=0.95 can prune all terms when
        # documents are near-identical. Relax it dynamically.
        if len(documents) <= 5:
            self._vectorizer.set_params(max_df=1.0)

        matrix = self._vectorizer.fit_transform(documents).toarray()
        self._fitted = True
        return matrix

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform *documents* using a previously fitted vocabulary."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        return self._vectorizer.transform(documents).toarray()

    @property
    def vocabulary_size(self) -> int:
        """Number of features (vocabulary terms) in the fitted vectorizer."""
        if not self._fitted:
            return 0
        return len(self._vectorizer.vocabulary_)

    @property
    def feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        if not self._fitted:
            return []
        return self._vectorizer.get_feature_names_out().tolist()
