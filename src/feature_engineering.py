import logging
import math
import re
import string
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
try:
    from textblob import TextBlob
except ImportError:  # pragma: no cover - optional fallback for minimal environments
    TextBlob = None


LOGGER = logging.getLogger(__name__)
_MISSING_TEXTBLOB_WARNED = False


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\b\w+\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
_CITATION_RE = re.compile(r"(\[\d+\]|\(\d{4}\)|according to|reported by|source[s]?:|study|researchers?)", re.IGNORECASE)
_ENTITY_PROXY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
_SOURCE_CUE_RE = re.compile(
    r"\b(according to|reported by|stated by|announced by|told reporters|official said|sources said)\b",
    re.IGNORECASE,
)


def _syllable_count(word: str) -> int:
    w = word.lower().strip()
    if not w:
        return 1

    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    words = _WORD_RE.findall(text)
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, len(sentences))
    syllables = sum(_syllable_count(w) for w in words)

    return float(206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllables / num_words))


def _safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def stylometric_vector(text: str) -> np.ndarray:
    words = _WORD_RE.findall(text)
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    num_chars = len(text)
    num_words = len(words)
    num_sentences = max(1, len(sentences))

    caps = sum(1 for ch in text if ch.isupper())
    letters = sum(1 for ch in text if ch.isalpha())

    punct_counts = {p: text.count(p) for p in ["!", "?", ",", ".", ":", ";", "'", '"', "-"]}

    avg_sentence_len = _safe_ratio(num_words, num_sentences)
    capitalization_ratio = _safe_ratio(caps, letters)
    punctuation_ratio = _safe_ratio(sum(punct_counts.values()), max(1, num_chars))

    flesch = flesch_reading_ease(text)

    return np.array(
        [
            avg_sentence_len,
            capitalization_ratio,
            punctuation_ratio,
            punct_counts["!"],
            punct_counts["?"],
            punct_counts[","],
            punct_counts["."],
            punct_counts[":"],
            punct_counts[";"],
            punct_counts["'"],
            punct_counts['"'],
            punct_counts["-"],
            flesch,
        ],
        dtype=np.float32,
    )


def sentiment_subjectivity_vector(text: str) -> np.ndarray:
    global _MISSING_TEXTBLOB_WARNED
    if TextBlob is None:
        if not _MISSING_TEXTBLOB_WARNED:
            LOGGER.warning("textblob is not installed; sentiment and subjectivity will default to 0.0")
            _MISSING_TEXTBLOB_WARNED = True
        return np.array([0.0, 0.0], dtype=np.float32)

    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)
    return np.array([polarity, subjectivity], dtype=np.float32)


def claim_detection_vector(text: str) -> np.ndarray:
    num_numbers = len(_NUMBER_RE.findall(text))
    entity_proxy = len(_ENTITY_PROXY_RE.findall(text))
    source_cues = len(_SOURCE_CUE_RE.findall(text))
    citations = len(_CITATION_RE.findall(text))
    return np.array([num_numbers, entity_proxy, source_cues, citations], dtype=np.float32)


@dataclass
class FeatureArtifacts:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    scaler: StandardScaler


class HybridFeatureBuilder:
    def __init__(
        self,
        word_max_features: int = 250000,
        char_max_features: int = 250000,
    ) -> None:
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            stop_words="english",
            sublinear_tf=True,
            max_features=word_max_features,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=char_max_features,
        )
        self.scaler = StandardScaler(with_mean=False)

    def _as_series(self, texts) -> pd.Series:
        if isinstance(texts, pd.Series):
            return texts
        return pd.Series(list(texts))

    def _dense_features(self, texts: pd.Series) -> np.ndarray:
        feats = []
        for t in texts:
            text = str(t)
            base = stylometric_vector(text)
            sentiment = sentiment_subjectivity_vector(text)
            claim = claim_detection_vector(text)
            feats.append(np.concatenate([base, sentiment, claim], axis=0))

        dense = np.vstack(feats).astype(np.float32)
        return dense

    def fit_transform(self, texts: pd.Series) -> sparse.csr_matrix:
        text_values = self._as_series(texts).fillna("").astype(str)
        word = self.word_vectorizer.fit_transform(text_values)
        char = self.char_vectorizer.fit_transform(text_values)
        dense = self._dense_features(text_values)
        dense_scaled = self.scaler.fit_transform(dense)
        dense_sparse = sparse.csr_matrix(dense_scaled)
        matrix = sparse.hstack([word, char, dense_sparse], format="csr")
        LOGGER.info("Hybrid feature matrix shape=%s", matrix.shape)
        return matrix

    def transform(self, texts: pd.Series) -> sparse.csr_matrix:
        text_values = self._as_series(texts).fillna("").astype(str)
        word = self.word_vectorizer.transform(text_values)
        char = self.char_vectorizer.transform(text_values)
        dense = self._dense_features(text_values)
        dense_scaled = self.scaler.transform(dense)
        dense_sparse = sparse.csr_matrix(dense_scaled)
        return sparse.hstack([word, char, dense_sparse], format="csr")

    def export(self) -> FeatureArtifacts:
        return FeatureArtifacts(
            word_vectorizer=self.word_vectorizer,
            char_vectorizer=self.char_vectorizer,
            scaler=self.scaler,
        )

    @classmethod
    def from_artifacts(cls, artifacts: FeatureArtifacts) -> "HybridFeatureBuilder":
        obj = cls()
        obj.word_vectorizer = artifacts.word_vectorizer
        obj.char_vectorizer = artifacts.char_vectorizer
        obj.scaler = artifacts.scaler
        return obj


def feature_names_suffix() -> list[str]:
    return [
        "avg_sentence_len",
        "capitalization_ratio",
        "punctuation_ratio",
        "count_exclamation",
        "count_question",
        "count_comma",
        "count_period",
        "count_colon",
        "count_semicolon",
        "count_apostrophe",
        "count_quote",
        "count_dash",
        "flesch_reading_ease",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "count_numbers",
        "count_named_entities_proxy",
        "count_source_cues",
        "count_citations",
    ]
