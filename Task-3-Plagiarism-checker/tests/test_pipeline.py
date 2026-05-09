"""
test_pipeline.py — the safety net

These tests have saved me from breaking things more times than I can count.
Run with: pytest tests/ -v
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from plagiarism_detector import (
    TextPreprocessor,
    DocumentVectorizer,
    SimilarityEngine,
    PlagiarismDetector,
    DetectorConfig,
    Reporter,
)


# ---------------------------------------------------------------------------
# TextPreprocessor
# ---------------------------------------------------------------------------

class TestTextPreprocessor:

    def setup_method(self):
        self.prep = TextPreprocessor(
            remove_stopwords=True,
            lemmatize=True,
            remove_numbers=False,
            min_token_length=2,
        )

    def test_basic_cleaning_works(self):
        result = self.prep.process("Hello, World! This is a TEST.")
        assert "hello" in result or "world" in result

    def test_url_and_email_removal(self):
        result = self.prep.process("Visit https://example.com for more info.")
        assert "example.com" not in result
        assert "http" not in result

    def test_stopword_removal(self):
        result = self.prep.process("This is a simple test of the system.")
        assert "is" not in result.split()
        assert "the" not in result.split()

    def test_empty_string(self):
        assert self.prep.process("") == ""
        assert self.prep.process("   ") == ""

    def test_batch_processing(self):
        texts = ["Hello world", "Good morning"]
        results = self.prep.process_batch(texts)
        assert len(results) == 2

    def test_number_removal(self):
        prep = TextPreprocessor(remove_numbers=True)
        result = prep.process("There are 42 apples and 100 oranges.")
        assert "42" not in result
        assert "100" not in result


# ---------------------------------------------------------------------------
# DocumentVectorizer
# ---------------------------------------------------------------------------

class TestDocumentVectorizer:

    def test_fit_transform_shape(self):
        v = DocumentVectorizer(ngram_range=(1, 1), max_features=100)
        docs = ["hello world test", "another document here", "hello again"]
        matrix = v.fit_transform(docs)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] <= 100

    def test_empty_corpus_raises(self):
        v = DocumentVectorizer()
        with pytest.raises(ValueError):
            v.fit_transform([])

    def test_transform_requires_fit(self):
        v = DocumentVectorizer()
        with pytest.raises(RuntimeError):
            v.transform(["some text"])

    def test_vocabulary_populated(self):
        v = DocumentVectorizer()
        v.fit_transform(["hello world", "foo bar"])
        assert v.vocabulary_size > 0
        assert len(v.feature_names) == v.vocabulary_size


# ---------------------------------------------------------------------------
# SimilarityEngine
# ---------------------------------------------------------------------------

class TestSimilarityEngine:

    def setup_method(self):
        self.engine = SimilarityEngine(threshold=0.75, use_semantic=False)

    def _make_vectors(self, texts):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        vecs = TfidfVectorizer().fit_transform(texts).toarray()
        return vecs

    def test_identical_documents_high_score(self):
        texts = ["the quick brown fox jumps over the lazy dog"] * 2
        vecs = self._make_vectors(texts)
        result = self.engine.compare_pair(
            "a", texts[0], vecs[0],
            "b", texts[1], vecs[1],
        )
        assert result.tfidf_score > 0.99
        assert result.composite_score > 0.75

    def test_different_documents_low_score(self):
        texts = [
            "artificial intelligence machine learning",
            "rainforest amazon biodiversity ecosystem",
        ]
        vecs = self._make_vectors(texts)
        result = self.engine.compare_pair(
            "a", texts[0], vecs[0],
            "b", texts[1], vecs[1],
        )
        assert result.tfidf_score < 0.3

    def test_compare_all_returns_correct_pair_count(self):
        texts = ["doc a text", "doc b text", "doc c text"]
        vecs = self._make_vectors(texts)
        results = self.engine.compare_all(["a", "b", "c"], texts, vecs)
        # 3 docs → 3 unique pairs
        assert len(results) == 3

    def test_results_sorted_descending(self):
        texts = [
            "hello world hello world",
            "hello world hello",
            "completely different topic here",
        ]
        vecs = self._make_vectors(texts)
        results = self.engine.compare_all(["a", "b", "c"], texts, vecs)
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_plagiarism_flag(self):
        engine = SimilarityEngine(threshold=0.5, use_semantic=False)
        text = "the exact same text repeated verbatim in both documents"
        texts = [text, text]
        vecs = self._make_vectors(texts)
        result = engine.compare_pair("a", texts[0], vecs[0], "b", texts[1], vecs[1])
        assert result.is_plagiarism is True


# ---------------------------------------------------------------------------
# PlagiarismDetector (integration)
# ---------------------------------------------------------------------------

class TestPlagiarismDetector:

    SAMPLE_DIR = Path(__file__).parent.parent / "sample_documents"

    def test_check_texts_with_identical(self):
        detector = PlagiarismDetector(DetectorConfig(threshold=0.70))
        text = "Artificial intelligence is transforming modern industries significantly."
        result = detector.check_texts({"A": text, "B": text})
        assert result.flagged_count >= 1

    def test_check_texts_minimum_document_count(self):
        detector = PlagiarismDetector()
        with pytest.raises(ValueError):
            detector.check_texts({"only_one": "just one document here"})

    def test_check_directory(self):
        if not self.SAMPLE_DIR.exists():
            pytest.skip("sample_documents directory not found")
        detector = PlagiarismDetector(DetectorConfig(threshold=0.60))
        result = detector.check_directory(self.SAMPLE_DIR)
        assert result.total_pairs > 0
        assert len(result.documents) >= 2

    def test_pipeline_result_counts(self):
        detector = PlagiarismDetector(DetectorConfig(threshold=0.50))
        texts = {"A": "hello world", "B": "hello world", "C": "something else entirely"}
        result = detector.check_texts(texts)
        # 3 docs → 3 pairs
        assert result.total_pairs == 3

    def test_missing_file_raises(self):
        detector = PlagiarismDetector()
        with pytest.raises(FileNotFoundError):
            detector.check_files(["nonexistent_file_xyz.txt"])


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class TestReporter:

    def test_html_report_created(self, tmp_path):
        detector = PlagiarismDetector(DetectorConfig(threshold=0.5))
        result = detector.check_texts({
            "doc1": "machine learning is a subset of artificial intelligence",
            "doc2": "artificial intelligence includes machine learning techniques",
        })
        reporter = Reporter(output_dir=tmp_path)
        html_path = reporter.generate_html(result, filename="test.html")
        assert html_path.exists()
        content = html_path.read_text()
        assert "Plagiarism" in content

    def test_json_report_valid(self, tmp_path):
        import json
        detector = PlagiarismDetector(DetectorConfig(threshold=0.5))
        result = detector.check_texts({
            "doc1": "hello world of natural language processing",
            "doc2": "natural language processing hello world",
        })
        reporter = Reporter(output_dir=tmp_path)
        json_path = reporter.generate_json(result, filename="test.json")
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "results" in data
        assert "summary" in data
