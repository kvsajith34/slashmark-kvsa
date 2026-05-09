"""
Similarity Engine
==================
Where we actually decide if two documents are suspiciously similar.

We combine three signals:
- TF-IDF cosine (word overlap)
- rapidfuzz (fuzzy string matching, great for minor edits)
- optional spaCy semantic (meaning similarity)

1. **TF-IDF Cosine Similarity** – catches word-overlap plagiarism and
   paraphrasing at the vocabulary level.
2. **Fuzzy String Matching** (rapidfuzz) – catches copy-paste with minor
   edits (typo fixes, punctuation changes).
3. **Semantic Similarity** (spaCy ``en_core_web_md``) – optional; catches
   paraphrasing that swaps synonyms while preserving meaning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimilarityResult:
    """Stores all similarity scores for a single document pair."""
    doc_a: str
    doc_b: str
    tfidf_score: float = 0.0
    fuzzy_score: float = 0.0
    semantic_score: float = 0.0
    composite_score: float = 0.0
    is_plagiarism: bool = False

    def to_dict(self) -> dict:
        return {
            "document_a": self.doc_a,
            "document_b": self.doc_b,
            "tfidf_similarity": round(self.tfidf_score, 4),
            "fuzzy_similarity": round(self.fuzzy_score, 4),
            "semantic_similarity": round(self.semantic_score, 4),
            "composite_score": round(self.composite_score, 4),
            "is_plagiarism": self.is_plagiarism,
        }


# ---------------------------------------------------------------------------
# Similarity Engine
# ---------------------------------------------------------------------------

class SimilarityEngine:
    """
    Compute pairwise similarity between documents using multiple methods.

    Parameters
    ----------
    threshold : float
        Composite score above which a pair is flagged as plagiarism
        (0.0 – 1.0, default ``0.75``).
    weights : dict
        Relative importance of each method.  Keys: ``"tfidf"``,
        ``"fuzzy"``, ``"semantic"``.  Values are normalised internally.
    use_semantic : bool
        If ``True`` and spaCy ``en_core_web_md`` is installed, add a
        semantic similarity component.  Falls back gracefully if the
        model is unavailable.
    fuzzy_method : str
        Which rapidfuzz scorer to use: ``"token_sort_ratio"`` (default),
        ``"partial_ratio"``, or ``"ratio"``.
    """

    _FUZZY_METHODS = {
        "ratio": "ratio",
        "partial_ratio": "partial_ratio",
        "token_sort_ratio": "token_sort_ratio",
    }

    def __init__(
        self,
        threshold: float = 0.75,
        weights: Optional[dict] = None,
        use_semantic: bool = False,
        fuzzy_method: str = "token_sort_ratio",
    ) -> None:
        self.threshold = threshold
        self.use_semantic = use_semantic
        self._fuzzy_method = fuzzy_method

        # Default weights
        default_weights = {"tfidf": 0.5, "fuzzy": 0.3, "semantic": 0.2}
        self._weights = {**default_weights, **(weights or {})}
        self._normalise_weights()

        self._spacy_nlp = None
        if use_semantic:
            self._load_spacy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare_pair(
        self,
        name_a: str,
        text_a: str,
        vec_a: np.ndarray,
        name_b: str,
        text_b: str,
        vec_b: np.ndarray,
    ) -> SimilarityResult:
        """
        Compute all similarity metrics for a single document pair.

        Parameters
        ----------
        name_a / name_b : str
            Display names (file paths, titles, etc.).
        text_a / text_b : str
            Raw (pre-processed) text of each document.
        vec_a / vec_b : np.ndarray
            TF-IDF vectors from :class:`DocumentVectorizer`.
        """
        tfidf = self._cosine(vec_a, vec_b)
        fuzzy = self._fuzzy(text_a, text_b)
        semantic = self._semantic(text_a, text_b) if self.use_semantic and self._spacy_nlp else 0.0

        composite = (
            self._weights["tfidf"] * tfidf
            + self._weights["fuzzy"] * fuzzy
            + self._weights["semantic"] * semantic
        )

        return SimilarityResult(
            doc_a=name_a,
            doc_b=name_b,
            tfidf_score=tfidf,
            fuzzy_score=fuzzy,
            semantic_score=semantic,
            composite_score=composite,
            is_plagiarism=composite >= self.threshold,
        )

    def compare_all(
        self,
        names: List[str],
        texts: List[str],
        vectors: np.ndarray,
    ) -> List[SimilarityResult]:
        """
        Compare every unique document pair in the corpus.

        Returns a list of :class:`SimilarityResult` objects sorted by
        composite score descending.
        """
        results: List[SimilarityResult] = []
        n = len(names)

        for i in range(n):
            for j in range(i + 1, n):
                result = self.compare_pair(
                    names[i], texts[i], vectors[i],
                    names[j], texts[j], vectors[j],
                )
                results.append(result)
                logger.debug(
                    "  %s ↔ %s  →  composite=%.3f  flagged=%s",
                    names[i], names[j], result.composite_score, result.is_plagiarism,
                )

        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Similarity methods
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        score = cosine_similarity([vec_a], [vec_b])[0][0]
        return float(np.clip(score, 0.0, 1.0))

    def _fuzzy(self, text_a: str, text_b: str) -> float:
        from rapidfuzz import fuzz
        scorer = getattr(fuzz, self._fuzzy_method, fuzz.token_sort_ratio)
        raw = scorer(text_a, text_b)          # returns 0-100
        return float(raw) / 100.0

    def _semantic(self, text_a: str, text_b: str) -> float:
        if self._spacy_nlp is None:
            return 0.0
        try:
            doc_a = self._spacy_nlp(text_a[:25_000])   # spaCy memory guard
            doc_b = self._spacy_nlp(text_b[:25_000])
            score = doc_a.similarity(doc_b)
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:  # pragma: no cover
            logger.warning("Semantic similarity failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalise_weights(self) -> None:
        total = sum(self._weights.values())
        if total == 0:
            raise ValueError("Weights must sum to a positive number.")
        self._weights = {k: v / total for k, v in self._weights.items()}

    def _load_spacy(self) -> None:
        try:
            import spacy
            self._spacy_nlp = spacy.load("en_core_web_md")
            logger.info("spaCy model 'en_core_web_md' loaded for semantic similarity.")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_md' not found. "
                "Run:  python -m spacy download en_core_web_md\n"
                "Semantic similarity will be skipped (score=0)."
            )
            self._spacy_nlp = None
        except ImportError:
            logger.warning("spaCy is not installed. Semantic similarity disabled.")
            self._spacy_nlp = None
