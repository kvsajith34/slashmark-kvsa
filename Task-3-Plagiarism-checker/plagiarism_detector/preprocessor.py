"""
preprocessor.py — the "make this text actually usable" stage

Cleans, normalizes, tokenizes, removes stopwords, and lemmatizes.
The whole thing is built to survive in environments where you can't
download NLTK data (common in CI or air-gapped boxes). It falls back
to a hand-curated stopword list and simple regex tokenization.
"""

import re
import string
import unicodedata
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hand-rolled fallback stopword list
# (a sensible subset of NLTK's). Used when the download fails or we're offline.
# ---------------------------------------------------------------------------
_FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now",
}


# ---------------------------------------------------------------------------
# Lazy NLTK bootstrap (downloads only what is missing, with graceful fallback)
# ---------------------------------------------------------------------------

def _ensure_nltk_data() -> dict:
    """
    Try to make required NLTK corpora available.

    Returns a dict of availability flags::

        {"punkt": bool, "stopwords": bool, "wordnet": bool}
    """
    import nltk
    available: dict = {"punkt": False, "stopwords": False, "wordnet": False}

    checks = [
        ("tokenizers/punkt_tab", "punkt_tab",  "punkt"),
        ("tokenizers/punkt",     "punkt",       "punkt"),
        ("corpora/stopwords",    "stopwords",   "stopwords"),
        ("corpora/wordnet",      "wordnet",     "wordnet"),
    ]

    for nltk_path, package, key in checks:
        if available[key]:          # already confirmed available
            continue
        try:
            nltk.data.find(nltk_path)
            available[key] = True
            continue
        except LookupError:
            pass
        try:
            nltk.download(package, quiet=True, raise_on_error=True)
            available[key] = True
        except Exception:
            logger.debug(
                "NLTK package '%s' unavailable — using fallback for '%s'.",
                package, key,
            )

    return available


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """
    Full text-normalization pipeline.

    Steps (all toggleable via constructor flags):
        1. Unicode normalization (NFC)
        2. Lower-casing
        3. URL / e-mail removal
        4. Punctuation stripping
        5. Number removal  (optional)
        6. Tokenisation (NLTK punkt if available, else regex split)
        7. Stop-word removal (optional; NLTK corpus or built-in fallback)
        8. Lemmatization (optional; NLTK WordNetLemmatizer or no-op fallback)
        9. Short-token filtering (min_token_length)
        10. Re-join into a clean string

    Parameters
    ----------
    remove_stopwords : bool
        Strip common English stop-words.
    lemmatize : bool
        Reduce tokens to their base form via WordNet lemmatizer.
    remove_numbers : bool
        Drop purely numeric tokens.
    min_token_length : int
        Discard tokens shorter than this many characters.
    extra_stopwords : list[str]
        Domain-specific words to add to the stop list.
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        remove_numbers: bool = False,
        min_token_length: int = 2,
        extra_stopwords: Optional[List[str]] = None,
    ) -> None:
        avail = _ensure_nltk_data()

        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_numbers = remove_numbers
        self.min_token_length = min_token_length

        # --- Stop-words ---
        if avail["stopwords"]:
            from nltk.corpus import stopwords as _sw
            self._stop_words: set = set(_sw.words("english"))
        else:
            self._stop_words = set(_FALLBACK_STOPWORDS)

        if extra_stopwords:
            self._stop_words.update(w.lower() for w in extra_stopwords)

        # --- Lemmatizer ---
        self._lemmatizer = None
        if avail["wordnet"]:
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()

        # --- Tokenizer flag ---
        self._use_nltk_tokenizer = avail["punkt"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> str:
        """Return a cleaned, normalised version of *text*."""
        if not text or not text.strip():
            return ""

        text = self._unicode_normalize(text)
        text = text.lower()
        text = self._remove_urls(text)
        text = self._strip_punctuation(text)

        tokens = self._tokenize(text)

        if self.remove_numbers:
            tokens = [t for t in tokens if not t.isnumeric()]

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]

        if self.lemmatize and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return " ".join(tokens)

    def process_batch(self, texts: List[str]) -> List[str]:
        """Normalise a list of texts."""
        return [self.process(t) for t in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unicode_normalize(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        url_pattern = re.compile(
            r"http[s]?://\S+|www\.\S+|\S+@\S+\.\S+", re.IGNORECASE
        )
        return url_pattern.sub(" ", text)

    @staticmethod
    def _strip_punctuation(text: str) -> str:
        # Replace hyphens / dashes with space to preserve word boundaries
        text = re.sub(r"[-–—]", " ", text)
        return text.translate(str.maketrans("", "", string.punctuation))

    def _tokenize(self, text: str) -> List[str]:
        if self._use_nltk_tokenizer:
            import nltk
            return nltk.word_tokenize(text)
        # Fallback: split on whitespace / non-alphanumeric boundaries
        return re.findall(r"\b[a-z0-9]+\b", text)
