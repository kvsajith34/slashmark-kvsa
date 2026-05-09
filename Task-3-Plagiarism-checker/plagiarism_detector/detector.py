"""
detector.py — the heart of the whole thing

This is where everything gets wired together:
text cleaning → TF-IDF vectors → similarity scoring → flagging.

I tried to keep the public API dead simple so you can drop it into
scripts or notebooks without reading the whole source.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .preprocessor import TextPreprocessor
from .vectorizer import DocumentVectorizer
from .similarity import SimilarityEngine, SimilarityResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass (simple alternative to YAML for programmatic use)
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """
    All the knobs you can twist.

    I made the defaults pretty reasonable, but everyone’s data is different.
    Tweak threshold and weights until the results feel right for your use case.
    """
    threshold: float = 0.75
    ngram_range: tuple = (1, 2)
    max_features: Optional[int] = 10_000
    remove_stopwords: bool = True
    lemmatize: bool = True
    remove_numbers: bool = False
    use_semantic: bool = False
    weights: Dict[str, float] = field(
        default_factory=lambda: {"tfidf": 0.5, "fuzzy": 0.3, "semantic": 0.2}
    )
    fuzzy_method: str = "token_sort_ratio"
    extra_stopwords: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DetectorConfig":
        """Load config from a YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                # Convert lists → tuples where needed
                if key == "ngram_range" and isinstance(value, list):
                    value = tuple(value)
                setattr(cfg, key, value)
        return cfg


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Everything that comes out of a full run — documents, scores, flagged pairs, timing, etc."""
    documents: List[str]                         # file names / titles
    raw_texts: Dict[str, str]                    # name → original text
    processed_texts: Dict[str, str]              # name → normalised text
    similarity_results: List[SimilarityResult]   # sorted by score
    plagiarism_pairs: List[SimilarityResult]     # filtered subset
    elapsed_seconds: float
    config: DetectorConfig

    @property
    def flagged_count(self) -> int:
        return len(self.plagiarism_pairs)

    @property
    def total_pairs(self) -> int:
        return len(self.similarity_results)


class PlagiarismDetector:
    """
    End-to-end plagiarism detection pipeline.

    Usage
    -----
    ::

        detector = PlagiarismDetector()
        result = detector.check_files(["essay1.txt", "essay2.txt"])
        # or with raw text strings:
        result = detector.check_texts({"Alice": "...", "Bob": "..."})
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._preprocessor = TextPreprocessor(
            remove_stopwords=self.config.remove_stopwords,
            lemmatize=self.config.lemmatize,
            remove_numbers=self.config.remove_numbers,
            extra_stopwords=self.config.extra_stopwords,
        )
        self._vectorizer = DocumentVectorizer(
            ngram_range=self.config.ngram_range,
            max_features=self.config.max_features,
        )
        self._engine = SimilarityEngine(
            threshold=self.config.threshold,
            weights=self.config.weights,
            use_semantic=self.config.use_semantic,
            fuzzy_method=self.config.fuzzy_method,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_files(
        self,
        file_paths: List[Union[str, Path]],
        encoding: str = "utf-8",
    ) -> PipelineResult:
        """
        Run plagiarism detection on a list of text files.

        Parameters
        ----------
        file_paths : list
            Paths to ``.txt`` (or any plain-text) files.
        encoding : str
            File encoding (default UTF-8).

        Returns
        -------
        PipelineResult
        """
        texts: Dict[str, str] = {}
        for fp in file_paths:
            path = Path(fp)
            if not path.exists():
                logger.warning("File not found, skipping: %s", fp)
                continue
            try:
                texts[path.name] = path.read_text(encoding=encoding, errors="replace")
            except Exception as exc:
                logger.error("Could not read %s: %s", fp, exc)
        if not texts:
            raise FileNotFoundError("No valid files could be read.")
        return self.check_texts(texts)

    def check_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.txt",
        recursive: bool = False,
        encoding: str = "utf-8",
    ) -> PipelineResult:
        """
        Walk a folder (optionally recursive) and run detection on everything
        that matches the glob. Super handy for assignment submissions or blog archives.
        """
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        glob_fn = d.rglob if recursive else d.glob
        files = sorted(glob_fn(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {directory}."
            )

        logger.info("Found %d file(s) in '%s'.", len(files), directory)
        return self.check_files(files, encoding=encoding)

    def check_texts(self, texts: Dict[str, str]) -> PipelineResult:
        """
        Run plagiarism detection on an in-memory mapping of name → text.

        Parameters
        ----------
        texts : dict
            ``{"document_name": "document content", ...}``
        """
        if len(texts) < 2:
            raise ValueError("At least 2 documents are required for comparison.")

        start = time.perf_counter()
        names = list(texts.keys())
        raw_texts = dict(texts)

        logger.info("Preprocessing %d documents …", len(names))
        processed_list = self._preprocessor.process_batch(
            [texts[n] for n in names]
        )
        processed_texts = dict(zip(names, processed_list))

        logger.info("Vectorising with TF-IDF (n-grams=%s) …", self.config.ngram_range)
        vectors = self._vectorizer.fit_transform(processed_list)

        logger.info("Computing pairwise similarities …")
        all_results = self._engine.compare_all(names, processed_list, vectors)
        plagiarism_pairs = [r for r in all_results if r.is_plagiarism]

        elapsed = time.perf_counter() - start
        logger.info(
            "Done in %.2fs — %d/%d pairs flagged as plagiarism (threshold=%.2f).",
            elapsed, len(plagiarism_pairs), len(all_results), self.config.threshold,
        )

        return PipelineResult(
            documents=names,
            raw_texts=raw_texts,
            processed_texts=processed_texts,
            similarity_results=all_results,
            plagiarism_pairs=plagiarism_pairs,
            elapsed_seconds=elapsed,
            config=self.config,
        )
