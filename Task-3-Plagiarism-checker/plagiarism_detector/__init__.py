"""
plagiarism_detector
====================

The main package for the plagiarism detector I built.

It glues together text cleaning, TF-IDF vectors, fuzzy matching, and optional
semantic similarity so you can quickly spot copied or paraphrased work.

Quick start (the bits I use most often):

    from plagiarism_detector import PlagiarismDetector, DetectorConfig, Reporter

    config = DetectorConfig(threshold=0.70, ngram_range=(1, 3))
    detector = PlagiarismDetector(config)

    result = detector.check_directory("my_essays/")

    reporter = Reporter(output_dir="reports/")
    reporter.print_summary(result)
    reporter.generate_html(result)
"""

from .detector import PlagiarismDetector, DetectorConfig, PipelineResult
from .preprocessor import TextPreprocessor
from .vectorizer import DocumentVectorizer
from .similarity import SimilarityEngine, SimilarityResult
from .reporter import Reporter

__all__ = [
    "PlagiarismDetector",
    "DetectorConfig",
    "PipelineResult",
    "TextPreprocessor",
    "DocumentVectorizer",
    "SimilarityEngine",
    "SimilarityResult",
    "Reporter",
]

__version__ = "1.0.0"
