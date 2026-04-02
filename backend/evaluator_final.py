"""
Tamil Descriptive Answer Evaluation Engine  ·  v5.0
=====================================================
Evaluates student Tamil answers using:
  - Multilingual semantic similarity  (LaBSE — Language-Agnostic BERT Sentence Embedding)
  - Concept coverage scoring          (concept groups with Tamil/English synonym support)
  - Weighted keyword importance       (per-keyword weight + partial SequenceMatcher credit)
  - Contradiction detection           (cross-encoder NLI model)
  - Dynamic length handling           (marks-aware — no penalty for concise correct answers)
  - New final scoring formula         (0.50 semantic + 0.25 keyword + 0.25 concept)
  - 0.5-step rounding                 (6.24→6, 6.25→6.5, 6.75→7)
  - Enriched explainability breakdown (all sub-scores + feedback list)
  - Graded topic relevance            (irrelevant / partial-relevance / normal)
  - Model-answer embedding cache      (no repeated encoder calls per batch)
  - Accuracy metrics                  (MAE, RMSE, avg predicted vs expected)

Changes in v5.0 (over v4.0):
  • Semantic model upgraded from paraphrase-multilingual-MiniLM-L12-v2 to
    LaBSE (sentence-transformers/LaBSE) for superior Tamil + multilingual quality.
  • Concept coverage scoring — concept groups (list[list[str]]) with fuzzy
    Tamil/English synonym matching; returns concept_score on 0–10 scale.
  • Weighted keywords — dataset keywords now support float weights
    (plain strings default to weight 1.0).  Normalised to 0–10 scale.
  • Contradiction detection — cross-encoder/nli-MiniLM2-L6-H768 (CPU-friendly);
    outputs contradiction_score [0,1]; heavy penalty when detected.
  • Dynamic length handling — expected word count inferred from max_marks
    instead of comparing to model answer length; no penalty for concise answers.
  • New scoring formula:
      base = 0.50 * semantic + 0.25 * keyword + 0.25 * concept
      base *= (1 - contradiction_score * 0.5)
      + length adjustment → final_score → scaled to max_marks
  • 0.5-step rounding: round_half_step() applied to every per-question score.
  • evaluate_final() now accepts optional max_marks + concepts arguments.
  • Returned dict extended with concept_score, contradiction_score,
    length_adjustment, raw_score, rounded_score, feedback (list).
  • Fully backward-compatible with all v4.x callers (old keys still present).
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from typing import Any, Union

import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  ·  edit here; no logic changes needed elsewhere
# ──────────────────────────────────────────────────────────────────────────────

# ── v5.0 Scoring Formula Weights ──────────────────────────────────────────────
# base = WEIGHT_SEMANTIC * semantic + WEIGHT_KEYWORD * keyword + WEIGHT_CONCEPT * concept
WEIGHT_SEMANTIC: float = 0.50
WEIGHT_KEYWORD: float  = 0.25
WEIGHT_CONCEPT: float  = 0.25

# Legacy dynamic weight constants (kept for backward-compat / run_evaluation banner)
SEMANTIC_WEIGHT: float = 0.7
KEYWORD_WEIGHT: float  = 0.3
DYNAMIC_KW_THRESHOLD: int   = 8
DYNAMIC_SEMANTIC_WEIGHT: float = 0.6
DYNAMIC_KEYWORD_WEIGHT: float  = 0.4

# Minimum word count in a student answer before it is graded normally
MIN_VALID_WORDS: int = 3

# ── Relevance thresholds ───────────────────────────────────────────────────────
IRRELEVANCE_THRESHOLD: float       = 0.20
PARTIAL_RELEVANCE_THRESHOLD: float = 0.35
PARTIAL_RELEVANCE_CAP: float       = 2.0   # max marks (out of 10) for partial answer

MAX_MARKS: int = 10

# ── v5.0 Semantic model (LaBSE — superior multilingual + Tamil quality) ────────
MODEL_NAME: str = "sentence-transformers/LaBSE"

# ── v5.0 Contradiction detection model (CPU-friendly NLI cross-encoder) ────────
NLI_MODEL_NAME: str = "cross-encoder/nli-MiniLM2-L6-H768"
# Contradiction label returned by the cross-encoder
CONTRADICTION_LABEL: str = "contradiction"
# If contradiction_score exceeds this → apply heavy penalty
CONTRADICTION_THRESHOLD: float = 0.5

# ── Fuzzy thresholds ───────────────────────────────────────────────────────────
FUZZY_THRESHOLD: float     = 0.75   # ≥ this ratio → "matched" keyword
PARTIAL_KW_THRESHOLD: float = 0.40  # ≥ this ratio → partial credit (< FUZZY)
CONCEPT_FUZZY_THRESHOLD: float = 0.70  # fuzzy match threshold for concept synonyms

# ── Dynamic length handling (v5.0) — inferred from max_marks ──────────────────
# Expected word count = base_words_per_mark * max_marks
# Only penalise when student word count < LENGTH_INSUFFICIENT_RATIO * expected
BASE_WORDS_PER_MARK: float = 15.0        # ~15 words per mark allotted
LENGTH_INSUFFICIENT_RATIO: float = 0.35  # below 35 % of expected → penalty
LENGTH_PENALTY_VALUE: float = 0.85       # multiplier applied when insufficient

# Feedback thresholds
FEEDBACK_LOW_SEMANTIC: float        = 6.0
FEEDBACK_LOW_LENGTH_RATIO: float    = 0.4
FEEDBACK_MANY_MISSING_RATIO: float  = 0.5
FEEDBACK_LOW_CONFIDENCE: float      = 0.45

# Keyword type alias for the public API
KeywordInput = Union[str, dict[str, Any]]

# ── Module-level singletons ────────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_nli_model: CrossEncoder | None = None
_embedding_cache: dict[str, np.ndarray] = {}   # model-answer embedding cache

import logging as _logging
_log = _logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def get_model() -> SentenceTransformer:
    """
    Return the cached LaBSE SentenceTransformer model, loading it on first call.

    LaBSE (Language-Agnostic BERT Sentence Embedding) significantly outperforms
    paraphrase-multilingual-MiniLM-L12-v2 for Tamil, producing higher-quality
    cross-lingual sentence embeddings.  The model is kept in the module-level
    ``_model`` variable so it is initialised only once per process.

    Returns:
        Loaded SentenceTransformer instance (LaBSE).
    """
    global _model
    if _model is None:
        _log.info("Loading SentenceTransformer model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        _log.info("LaBSE model loaded successfully.")
    return _model


def get_nli_model() -> CrossEncoder:
    """
    Return the cached NLI CrossEncoder model, loading it on first call.

    Uses cross-encoder/nli-MiniLM2-L6-H768 — a lightweight but accurate
    NLI model that runs on CPU without significant latency.

    Returns:
        Loaded CrossEncoder instance for contradiction detection.
    """
    global _nli_model
    if _nli_model is None:
        _log.info("Loading NLI CrossEncoder model: %s", NLI_MODEL_NAME)
        _nli_model = CrossEncoder(NLI_MODEL_NAME)
        _log.info("NLI model loaded successfully.")
    return _nli_model


# ──────────────────────────────────────────────────────────────────────────────
# 2. Text Pre-processing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: Any) -> str:
    """
    Normalise raw Tamil (or mixed) text before evaluation.

    Steps applied in order:
      1. Unicode NFC normalisation  – collapses equivalent code-point sequences.
      2. Collapse whitespace / newlines  – multiple spaces/tabs/newlines → single space.
      3. Strip leading / trailing whitespace.
      4. Remove common OCR artefacts  – repeated dots, pipe characters, underscores
         that OCR engines insert as fill characters in answer sheets.
      5. Lowercase conversion  – safe for ASCII portions; Tamil script is caseless
         so this only affects any mixed-script content.

    This function is intentionally conservative: it does **not** strip Tamil
    punctuation (``।``, ``?``, ``!``) because those tokens are meaningful for
    semantic encoding.

    Args:
        text: Raw input string (student answer, model answer, or keyword).
              Non-string input is coerced to ``""`` gracefully.

    Returns:
        Cleaned, normalised string.
    """
    if not isinstance(text, str) or not text:
        return ""

    # Step 1 – Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)

    # Step 2 & 3 – collapse all whitespace variants into a single space
    text = re.sub(r"[\s\u00A0\u200B]+", " ", text).strip()

    # Step 4 – remove OCR fill artefacts
    text = re.sub(r"[|_]{2,}", " ", text)   # repeated pipes / underscores
    text = re.sub(r"\.{3,}", " ", text)     # ellipsis-style dots (…)

    # Step 5 – lowercase (affects only ASCII / mixed content)
    text = text.lower()

    # Final cleanup pass after substitutions
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────────────────────────────────────────────────────────────
# 3. Keyword Normalisation  (backward-compatible)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_keywords(
    keywords: list[KeywordInput],
) -> list[dict[str, Any]]:
    """
    Normalise a mixed keyword list into a uniform list of dicts.

    Accepts **both** formats, even mixed within the same list:
      - Plain string: ``"சொல்"``                  → ``{"word": "சொல்", "weight": 1}``
      - Weighted dict: ``{"word": "சொல்", "weight": 2}`` → unchanged

    Args:
        keywords: Raw keyword list from dataset or API caller.

    Returns:
        List of ``{"word": str, "weight": float}`` dicts.
        Weight values are clamped to ≥ 0.01 to prevent division-by-zero.
    """
    normalised: list[dict[str, Any]] = []
    for kw in keywords:
        if isinstance(kw, str):
            normalised.append({"word": kw, "weight": 1.0})
        elif isinstance(kw, dict):
            word = str(kw.get("word", "")).strip()
            weight = float(kw.get("weight", 1.0))
            normalised.append({"word": word, "weight": max(0.01, weight)})
    return normalised


# ──────────────────────────────────────────────────────────────────────────────
# 4. Model-Answer Embedding Cache  (NEW in v4.0)
# ──────────────────────────────────────────────────────────────────────────────

def get_model_embedding_cached(text: str) -> np.ndarray:
    """
    Return the normalised embedding for ``text``, using a module-level cache.

    On the first call for a given text the embedding is computed via the
    sentence-transformer model and stored in ``_embedding_cache``.  Subsequent
    calls with the same text return the cached array immediately, avoiding
    redundant encoder passes during batch evaluation.

    The cache is keyed by the exact (pre-processed) text string.  It persists
    for the lifetime of the process and is shared across all callers.

    Args:
        text: Pre-processed text to encode (model answer or student answer).

    Returns:
        1-D normalised numpy array (float32) of the sentence embedding.
    """
    global _embedding_cache
    if text not in _embedding_cache:
        transformer = get_model()
        embedding: np.ndarray = transformer.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        _embedding_cache[text] = embedding
    return _embedding_cache[text]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Semantic Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_semantic(
    student_answer: str,
    model_answer: str,
) -> tuple[float, float]:
    """
    Compute semantic similarity between student and model answers.

    Uses ``paraphrase-multilingual-MiniLM-L12-v2`` to produce sentence
    embeddings, then calculates cosine similarity.  Both texts should be
    pre-processed with :func:`preprocess_text` before calling this function.

    The model-answer embedding is retrieved from :func:`get_model_embedding_cached`
    so repeated batch evaluations against the same model answer do not
    re-encode it (performance improvement, NEW in v4.0).

    Args:
        student_answer: Pre-processed student Tamil answer.
        model_answer:   Pre-processed reference Tamil answer.

    Returns:
        Tuple of:
          - ``semantic_score``  : float in [0.0, 10.0]
          - ``confidence``      : float in [0.0,  1.0]  (raw cosine similarity)
    """
    # Student embedding — not cached (every student answer is unique)
    transformer = get_model()
    student_emb: np.ndarray = transformer.encode(
        student_answer,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Model-answer embedding — retrieved from cache (NEW v4.0)
    model_emb: np.ndarray = get_model_embedding_cached(model_answer)

    raw_similarity: float = float(
        cosine_similarity(
            student_emb.reshape(1, -1),
            model_emb.reshape(1, -1),
        )[0][0]
    )

    confidence: float = max(0.0, min(1.0, raw_similarity))
    semantic_score: float = round(confidence * MAX_MARKS, 2)

    return semantic_score, confidence


# ──────────────────────────────────────────────────────────────────────────────
# 6. Graded Topic Relevance Detection  (upgraded in v4.0)
# ──────────────────────────────────────────────────────────────────────────────

def is_irrelevant(confidence: float) -> bool:
    """
    Return True when the answer is completely off-topic.

    An answer is considered irrelevant when its cosine similarity to the model
    answer falls below ``IRRELEVANCE_THRESHOLD`` (default 0.20).

    See also :func:`is_partially_relevant` for the intermediate band.

    Args:
        confidence: Raw cosine similarity score in [0.0, 1.0].

    Returns:
        ``True``  if the answer should be treated as completely off-topic.
        ``False`` otherwise.
    """
    return confidence < IRRELEVANCE_THRESHOLD


def is_partially_relevant(confidence: float) -> bool:
    """
    Return True when the answer shows weak but non-zero topic alignment.

    The partial-relevance band is:
        ``IRRELEVANCE_THRESHOLD`` ≤ confidence < ``PARTIAL_RELEVANCE_THRESHOLD``
        (default 0.20 ≤ conf < 0.35)

    In this band the answer is not completely off-topic but lacks meaningful
    alignment with the expected content.  Scoring continues but final marks
    are capped at ``PARTIAL_RELEVANCE_CAP`` (default 2.0).

    Args:
        confidence: Raw cosine similarity score in [0.0, 1.0].

    Returns:
        ``True``  if the answer is in the partial-relevance band.
        ``False`` if the answer is either irrelevant or normally relevant.
    """
    return IRRELEVANCE_THRESHOLD <= confidence < PARTIAL_RELEVANCE_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# 7. Partial Keyword Matching  (NEW in v4.0)
# ──────────────────────────────────────────────────────────────────────────────

def _fuzzy_keyword_ratio(keyword: str, text: str) -> float:
    """
    Return the best SequenceMatcher ratio for ``keyword`` against ``text``.

    Strategy:
      1. If the keyword is an exact substring, return 1.0 immediately.
      2. Slide a window of size kw_len (±1) across the text words and track
         the maximum ratio seen.

    Args:
        keyword: Pre-processed keyword string.
        text:    Pre-processed student answer text.

    Returns:
        Best similarity ratio in [0.0, 1.0].
    """
    if keyword in text:
        return 1.0

    kw_words = keyword.split()
    text_words = text.split()
    kw_len = len(kw_words)

    if kw_len == 0 or not text_words:
        return 0.0

    best: float = 0.0
    for window_size in (kw_len, max(1, kw_len - 1), kw_len + 1):
        for start in range(len(text_words) - window_size + 1):
            window = " ".join(text_words[start: start + window_size])
            ratio = SequenceMatcher(None, keyword, window).ratio()
            if ratio > best:
                best = ratio

    return best


def _fuzzy_keyword_in_text(keyword: str, text: str, threshold: float) -> bool:
    """
    Return True if ``keyword`` matches any sub-sequence in ``text`` with a
    SequenceMatcher ratio ≥ ``threshold``.

    Kept for backward-compatibility; delegates to :func:`_fuzzy_keyword_ratio`.

    Args:
        keyword:   The keyword to search for (pre-processed).
        text:      The student answer text (pre-processed).
        threshold: Minimum similarity ratio (0–1) to count as a match.

    Returns:
        True if the keyword is considered present, False otherwise.
    """
    return _fuzzy_keyword_ratio(keyword, text) >= threshold


def compute_partial_keyword_score(
    student_answer: str,
    keywords: list[KeywordInput],
    fuzzy_threshold: float = FUZZY_THRESHOLD,
    partial_threshold: float = PARTIAL_KW_THRESHOLD,
) -> tuple[float, list[str], list[str]]:
    """
    Score keyword coverage using **partial credit** (NEW in v4.0).

    Instead of a binary matched/not-matched decision, each keyword earns a
    fractional score proportional to its best SequenceMatcher similarity ratio:

    - ratio ≥ ``fuzzy_threshold``   → full credit  (keyword added to *matched*)
    - ratio ≥ ``partial_threshold`` → partial credit (keyword added to *missing*,
                                      but partial credit is still awarded)
    - ratio <  ``partial_threshold`` → no credit    (keyword added to *missing*)

    The weighted sum of similarity ratios is divided by the total weight to
    obtain a score in [0, 1], then scaled to ``MAX_MARKS``.

    Args:
        student_answer:    Pre-processed student answer text.
        keywords:          List of keywords (strings or weighted dicts).
        fuzzy_threshold:   Ratio at or above which a keyword is "matched".
        partial_threshold: Ratio below which no credit is given at all.

    Returns:
        Tuple of:
          - ``keyword_score``    : float in [0.0, 10.0]
          - ``matched_keywords`` : list of keyword words with ratio ≥ fuzzy_threshold
          - ``missing_keywords`` : list of keyword words with ratio < fuzzy_threshold
    """
    if not keywords:
        return 0.0, [], []

    norm_kws = _normalise_keywords(keywords)
    total_weight: float = sum(kw["weight"] for kw in norm_kws)

    weighted_score: float = 0.0
    matched: list[str] = []
    missing: list[str] = []

    for kw in norm_kws:
        word_clean = preprocess_text(kw["word"])
        ratio = _fuzzy_keyword_ratio(word_clean, student_answer)

        if ratio >= fuzzy_threshold:
            # Full credit — counts as "matched" for UI display
            weighted_score += kw["weight"] * ratio
            matched.append(kw["word"])
        elif ratio >= partial_threshold:
            # Partial credit — keyword still listed as "missing" for feedback
            weighted_score += kw["weight"] * ratio
            missing.append(kw["word"])
        else:
            # No credit
            missing.append(kw["word"])

    normalised_ratio: float = weighted_score / total_weight if total_weight > 0 else 0.0
    keyword_score: float = round(normalised_ratio * MAX_MARKS, 2)

    return keyword_score, matched, missing


def evaluate_keywords(
    student_answer: str,
    keywords: list[KeywordInput],
    threshold: float = FUZZY_THRESHOLD,
) -> tuple[float, list[str], list[str]]:
    """
    Score keyword coverage and return the score, matched keywords, and missing
    keywords.

    In v4.0 this is a thin wrapper around :func:`compute_partial_keyword_score`
    so that partial credit is always applied.  The ``threshold`` parameter still
    controls the boundary between "matched" and "missing" labels.

    Backward compatibility:
      - Plain string lists (``["kw1", "kw2"]``) are fully supported.
      - Mixed lists (some dicts, some strings) are also normalised.
      - Return signature is unchanged: ``(score, matched, missing)``.

    Args:
        student_answer: Pre-processed student answer.
        keywords:       List of keywords — plain strings or dicts with
                        ``{"word": str, "weight": float}``.
        threshold:      Min ratio to classify a keyword as "matched" (default
                        ``FUZZY_THRESHOLD``).  Partial credit is still awarded
                        for ratios between ``PARTIAL_KW_THRESHOLD`` and this
                        value.

    Returns:
        Tuple of:
          - ``keyword_score``    : float in [0.0, 10.0]
          - ``matched_keywords`` : list[str] — keywords considered present
          - ``missing_keywords`` : list[str] — keywords considered absent
    """
    return compute_partial_keyword_score(
        student_answer,
        keywords,
        fuzzy_threshold=threshold,
        partial_threshold=PARTIAL_KW_THRESHOLD,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. Dynamic Weight Selection
# ──────────────────────────────────────────────────────────────────────────────

def get_dynamic_weights(keywords: list[KeywordInput]) -> tuple[float, float]:
    """
    Select semantic and keyword weights based on question complexity.

    Rules (configurable via module-level constants):
      - If the number of keywords exceeds ``DYNAMIC_KW_THRESHOLD`` (default 8):
        → Use higher keyword weight (``DYNAMIC_KEYWORD_WEIGHT``, default 0.4)
          and lower semantic weight (``DYNAMIC_SEMANTIC_WEIGHT``, default 0.6)
      - Otherwise: use the default weights (0.7 semantic / 0.3 keyword).

    Both returned values always sum to exactly 1.0.

    Args:
        keywords: Raw keyword list (may be strings or dicts).

    Returns:
        Tuple of ``(semantic_weight, keyword_weight)``, each in (0, 1).
    """
    if len(keywords) > DYNAMIC_KW_THRESHOLD:
        return DYNAMIC_SEMANTIC_WEIGHT, DYNAMIC_KEYWORD_WEIGHT
    return SEMANTIC_WEIGHT, KEYWORD_WEIGHT


# ──────────────────────────────────────────────────────────────────────────────
# 9. Dynamic Length Handling  (v5.0 — marks-aware, no verbosity reward)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_length_factor(
    student_answer: str,
    model_answer: str,
    max_marks: int = MAX_MARKS,
) -> float:
    """
    Compute a length adjustment factor in [0.0, 1.0] based on max_marks.

    v5.0 logic:
      - Expected word count is inferred from max_marks (not model answer length).
        expected_words = BASE_WORDS_PER_MARK * max_marks
      - Only penalise when student word count < LENGTH_INSUFFICIENT_RATIO *
        expected_words.  (default: < 35 % of expected)
      - No reward for verbosity — factor capped at 1.0.
      - Concise but complete answers are protected: factor = 1.0 unless
        the answer is extremely short for the marks allocated.

    Penalty formula when insufficient:
        factor = student_words / (LENGTH_INSUFFICIENT_RATIO * expected_words)
        clamped to [0.0, 1.0]

    Args:
        student_answer: Pre-processed student answer.
        model_answer:   Pre-processed model answer (kept for compat; not used).
        max_marks:      Marks allocated to the question (default: MAX_MARKS=10).

    Returns:
        float in [0.0, 1.0].  1.0 means no length penalty.
    """
    student_words = len(student_answer.split()) if student_answer.strip() else 0

    if student_words == 0:
        return 0.0

    expected_words = BASE_WORDS_PER_MARK * max(1, max_marks)
    threshold_words = LENGTH_INSUFFICIENT_RATIO * expected_words

    if student_words >= threshold_words:
        return 1.0  # adequate or longer — no penalty

    # Proportional penalty for very short answers
    factor = student_words / threshold_words
    return max(0.0, min(1.0, factor))


# ──────────────────────────────────────────────────────────────────────────────
# 10. Confidence-Based Score Adjustment  (kept from v4.0)
# ──────────────────────────────────────────────────────────────────────────────

def apply_confidence_adjustment(marks: float, confidence: float) -> float:
    """
    Apply a mild confidence-based modulation to final marks.

    Formula::

        adjusted = marks * (0.8 + 0.2 * confidence)

    Args:
        marks:      Raw final marks before confidence adjustment, in [0, MAX_MARKS].
        confidence: Cosine-similarity confidence score in [0.0, 1.0].

    Returns:
        Adjusted marks, clamped to [0.0, MAX_MARKS].
    """
    adjusted = marks * (0.8 + 0.2 * confidence)
    return max(0.0, min(float(MAX_MARKS), adjusted))


# ──────────────────────────────────────────────────────────────────────────────
# 10b. Concept Coverage Scoring  (NEW in v5.0)
# ──────────────────────────────────────────────────────────────────────────────

def compute_concept_score(
    student_answer: str,
    concepts: list[list[str]],
    fuzzy_threshold: float = CONCEPT_FUZZY_THRESHOLD,
) -> tuple[float, list[str], list[str]]:
    """
    Score concept coverage across concept groups.

    Each concept group is a list of synonyms / alternate forms (Tamil + English).
    A concept is considered "covered" if any synonym in the group fuzzy-matches
    the student answer above ``fuzzy_threshold``.

    Score = (covered_concepts / total_concepts) * MAX_MARKS

    Args:
        student_answer:  Pre-processed student answer text.
        concepts:        list[list[str]] — each inner list is a synonym group.
        fuzzy_threshold: Min SequenceMatcher ratio to accept a synonym match.

    Returns:
        Tuple of:
          - ``concept_score``   : float in [0.0, 10.0]
          - ``covered``         : list of representative synonyms that were found
          - ``missed``          : list of representative synonyms not found
    """
    if not concepts:
        return 0.0, [], []

    covered: list[str] = []
    missed: list[str] = []

    for group in concepts:
        if not group:
            continue
        representative = group[0]  # first synonym is used for display / feedback
        group_matched = False
        for synonym in group:
            syn_clean = preprocess_text(synonym)
            if syn_clean and _fuzzy_keyword_ratio(syn_clean, student_answer) >= fuzzy_threshold:
                group_matched = True
                break
        if group_matched:
            covered.append(representative)
        else:
            missed.append(representative)

    total = len(concepts)
    concept_score = round((len(covered) / total) * MAX_MARKS, 2) if total > 0 else 0.0
    return concept_score, covered, missed


# ──────────────────────────────────────────────────────────────────────────────
# 10c. Contradiction Detection  (NEW in v5.0)
# ──────────────────────────────────────────────────────────────────────────────

def detect_contradiction(
    student_answer: str,
    model_answer: str,
) -> float:
    """
    Detect whether the student answer contradicts the model answer.

    Uses the cross-encoder/nli-MiniLM2-L6-H768 NLI model to predict
    entailment / neutral / contradiction label scores.

    Returns:
        contradiction_score: float in [0.0, 1.0].
        Higher = stronger contradiction detected.
        Returns 0.0 gracefully if model unavailable or texts are too short.
    """
    if not student_answer.strip() or not model_answer.strip():
        return 0.0

    try:
        nli = get_nli_model()
        scores = nli.predict(
            [(model_answer, student_answer)],
            apply_softmax=True,
        )
        # CrossEncoder with nli-MiniLM2-L6-H768 returns [contradiction, entailment, neutral]
        # Label order: contradiction=0, entailment=1, neutral=2
        contradiction_score = float(scores[0][0])
        return max(0.0, min(1.0, contradiction_score))
    except Exception as exc:
        _log.warning("Contradiction detection failed (non-fatal): %s", exc)
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 10d. 0.5-Step Rounding  (NEW in v5.0)
# ──────────────────────────────────────────────────────────────────────────────

def round_half_step(value: float) -> float:
    """
    Round a score to the nearest integer or 0.5 step.

    Examples:
        6.24 → 6.0
        6.25 → 6.5
        6.74 → 6.5
        6.75 → 7.0

    Args:
        value: Raw float score.

    Returns:
        Score rounded to nearest 0.5 step.
    """
    return round(value * 2) / 2


# ──────────────────────────────────────────────────────────────────────────────
# 11. Improved Feedback Generation  (v5.0 — returns list of observations)
# ──────────────────────────────────────────────────────────────────────────────

def generate_feedback(
    semantic_score: float,
    keyword_score: float,
    missing_keywords: list[str],
    length_factor: float,
    is_empty: bool,
    is_irrelevant_answer: bool = False,
    is_partially_relevant_answer: bool = False,
    total_keyword_count: int = 0,
    confidence: float = 1.0,
    concept_score: float = 10.0,
    missed_concepts: list[str] | None = None,
    contradiction_score: float = 0.0,
) -> list[str]:
    """
    Generate a list of teacher-like feedback observations for a student answer.

    v5.0 returns a list[str] (each item is one observation) instead of a
    joined string.  Callers that need the old single-string format can do
    ``"; ".join(feedback_list)``.

    Args:
        semantic_score:               float [0, 10] from evaluate_semantic.
        keyword_score:                float [0, 10] from evaluate_keywords.
        missing_keywords:             List of unmatched keywords.
        length_factor:                float [0, 1] from calculate_length_factor.
        is_empty:                     True when the student answer is empty.
        is_irrelevant_answer:         True when the answer is completely off-topic.
        is_partially_relevant_answer: True when the answer is only weakly on-topic.
        total_keyword_count:          Total number of keywords evaluated.
        confidence:                   Raw cosine similarity.
        concept_score:                float [0, 10] concept coverage score.
        missed_concepts:              Concept groups not covered by the answer.
        contradiction_score:          float [0, 1] contradiction probability.

    Returns:
        List of feedback strings (may be empty list if all good).
    """
    missed_concepts = missed_concepts or []
    feedback: list[str] = []

    # Priority 1 – empty
    if is_empty:
        return ["No answer provided. Please attempt the question."]

    # Priority 2 – completely irrelevant
    if is_irrelevant_answer:
        return ["Answer is not relevant to the question."]

    # Priority 3 – partially relevant
    if is_partially_relevant_answer:
        return [
            "Answer is partially relevant but lacks alignment with the question. "
            "Please revise your answer to address the question more directly."
        ]

    # Contradiction detected
    if contradiction_score >= CONTRADICTION_THRESHOLD:
        feedback.append(
            f"Contradiction detected: your answer appears to conflict with the "
            f"expected answer (score={contradiction_score:.2f}). Please review."
        )

    # Length check
    if length_factor < LENGTH_INSUFFICIENT_RATIO:
        feedback.append(
            "Answer is too short for the marks allocated — please elaborate with more detail."
        )

    # Semantic clarity
    if semantic_score < FEEDBACK_LOW_SEMANTIC:
        feedback.append(
            "Answer lacks conceptual clarity — review the topic and rephrase."
        )

    # Low confidence
    if confidence < FEEDBACK_LOW_CONFIDENCE:
        feedback.append(
            "Evaluation confidence is low; answer may be unclear or ambiguous."
        )

    # Missed concepts
    if missed_concepts:
        shown = missed_concepts[:4]
        suffix = " and more" if len(missed_concepts) > 4 else ""
        feedback.append(f"Missed concept(s): {', '.join(shown)}{suffix}.")

    # Missing keywords
    if missing_keywords:
        shown = missing_keywords[:5]
        suffix = " and more" if len(missing_keywords) > 5 else ""
        missing_str = ", ".join(shown) + suffix
        many_missing = (
            total_keyword_count > 0
            and (len(missing_keywords) / total_keyword_count) >= FEEDBACK_MANY_MISSING_RATIO
        )
        if many_missing:
            feedback.append(
                f"Several key concepts are missing from your answer: {missing_str}. "
                "Please revise and include these essential ideas."
            )
        else:
            feedback.append(f"Missing key concepts: {missing_str}.")

    if not feedback:
        feedback.append("Good answer — covers the key concepts well.")

    return feedback


# ──────────────────────────────────────────────────────────────────────────────
# 12. Hybrid Final Score  (v5.0 — new formula + concepts + contradiction)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_final(
    student_answer: str,
    model_answer: str,
    keywords: list[KeywordInput],
    concepts: list[list[str]] | None = None,
    max_marks: int = MAX_MARKS,
) -> dict[str, Any]:
    """
    Compute the full hybrid evaluation result for one student answer.

    Pipeline (v5.0):
      1.  Pre-process both texts.
      2.  Empty / invalid answer fast-path — return zero marks immediately.
      3.  Compute semantic score + confidence (LaBSE embeddings, cached).
      4.  Graded relevance check (unchanged from v4.0).
      5.  Compute weighted keyword score.
      6.  Compute concept coverage score (NEW v5.0).
      7.  Detect contradiction (NEW v5.0).
      8.  New scoring formula:
            base = 0.50 * semantic + 0.25 * keyword + 0.25 * concept
            base *= (1 - contradiction_score * 0.5)
      9.  Dynamic length adjustment (marks-aware, NEW v5.0).
      10. Apply partial-relevance cap if applicable.
      11. Scale to max_marks.
      12. Apply 0.5-step rounding.
      13. Generate enriched feedback list.

    Args:
        student_answer: Raw student Tamil answer.
        model_answer:   Raw reference Tamil answer.
        keywords:       List of keywords — plain strings or dicts
                        ``{"word": str, "weight": float}``.
        concepts:       Optional list of concept groups (list[list[str]]).
                        Each inner list contains synonyms for one concept.
        max_marks:      Marks allocated to this question (default 10).

    Returns:
        Dictionary with keys (v5.0):
          - ``student_answer_clean``  : pre-processed student text
          - ``semantic_score``        : float [0, 10]
          - ``keyword_score``         : float [0, 10]
          - ``concept_score``         : float [0, 10]
          - ``contradiction_score``   : float [0, 1]
          - ``raw_score``             : float [0, 10]  (before length + cap)
          - ``length_adjustment``     : float (multiplier applied)
          - ``final_marks``           : float [0, 10]  (out of 10, for compat)
          - ``final_score``           : float (scaled to max_marks)
          - ``rounded_score``         : float (0.5-step rounded, ≤ max_marks)
          - ``confidence``            : float [0, 1]
          - ``matched_keywords``      : list[str]
          - ``missing_keywords``      : list[str]
          - ``covered_concepts``      : list[str]
          - ``missed_concepts``       : list[str]
          - ``feedback``              : list[str] — teacher-style observations
          - ``length_factor``         : float [0, 1]
          - ``is_irrelevant``         : bool
          - ``is_partially_relevant`` : bool
          - ``semantic_weight``       : float  (kept for compat)
          - ``keyword_weight``        : float  (kept for compat)
    """
    concepts = concepts or []

    # ── Step 1: Pre-processing ─────────────────────────────────────────────
    clean_student = preprocess_text(student_answer)
    clean_model   = preprocess_text(model_answer)

    all_kw_words: list[str] = [
        kw["word"] if isinstance(kw, dict) else str(kw) for kw in keywords
    ]
    all_concept_reps: list[str] = [grp[0] for grp in concepts if grp]

    # ── Step 2: Empty / very-short answer fast-path ────────────────────────
    student_word_count = len(clean_student.split()) if clean_student.strip() else 0
    is_empty = student_word_count < MIN_VALID_WORDS

    def _zero_result(fb: list[str]) -> dict[str, Any]:
        return {
            "student_answer_clean":  clean_student,
            "semantic_score":        0.0,
            "keyword_score":         0.0,
            "concept_score":         0.0,
            "contradiction_score":   0.0,
            "raw_score":             0.0,
            "length_adjustment":     0.0,
            "final_marks":           0.0,
            "final_score":           0.0,
            "rounded_score":         0.0,
            "confidence":            0.0,
            "matched_keywords":      [],
            "missing_keywords":      all_kw_words,
            "covered_concepts":      [],
            "missed_concepts":       all_concept_reps,
            "feedback":              fb,
            "length_factor":         0.0,
            "semantic_weight":       WEIGHT_SEMANTIC,
            "keyword_weight":        WEIGHT_KEYWORD,
            "is_irrelevant":         False,
            "is_partially_relevant": False,
        }

    if is_empty:
        fb = generate_feedback(
            semantic_score=0.0, keyword_score=0.0,
            missing_keywords=all_kw_words, length_factor=0.0,
            is_empty=True, total_keyword_count=len(all_kw_words),
        )
        return _zero_result(fb)

    # ── Step 3: Semantic evaluation (LaBSE, cached embeddings) ────────────
    semantic_score, confidence = evaluate_semantic(clean_student, clean_model)

    # ── Step 4a: Completely irrelevant ────────────────────────────────────
    if is_irrelevant(confidence):
        fb = generate_feedback(
            semantic_score=semantic_score, keyword_score=0.0,
            missing_keywords=all_kw_words, length_factor=0.0,
            is_empty=False, is_irrelevant_answer=True,
            total_keyword_count=len(all_kw_words), confidence=confidence,
        )
        result = _zero_result(fb)
        result.update({
            "semantic_score": semantic_score,
            "confidence":     round(confidence, 4),
            "is_irrelevant":  True,
            "semantic_weight": WEIGHT_SEMANTIC,
            "keyword_weight":  WEIGHT_KEYWORD,
        })
        return result

    # ── Step 4b: Partially relevant ────────────────────────────────────────
    partially_relevant = is_partially_relevant(confidence)

    # ── Step 5: Weighted keyword evaluation ───────────────────────────────
    keyword_score, matched_keywords, missing_keywords = evaluate_keywords(
        clean_student, keywords
    )

    # ── Step 6: Concept coverage (NEW v5.0) ───────────────────────────────
    concept_score, covered_concepts, missed_concepts = compute_concept_score(
        clean_student, concepts
    )

    # ── Step 7: Contradiction detection (NEW v5.0) ────────────────────────
    contradiction_score = detect_contradiction(clean_student, clean_model)

    # ── Step 8: New scoring formula (NEW v5.0) ────────────────────────────
    raw_score: float = (
        WEIGHT_SEMANTIC * semantic_score
        + WEIGHT_KEYWORD * keyword_score
        + WEIGHT_CONCEPT * concept_score
    )
    # Contradiction penalty
    raw_score *= (1.0 - contradiction_score * 0.5)
    raw_score = round(max(0.0, min(float(MAX_MARKS), raw_score)), 4)

    # ── Step 9: Dynamic length adjustment (marks-aware, NEW v5.0) ─────────
    length_factor = calculate_length_factor(clean_student, clean_model, max_marks)
    if length_factor < 1.0:
        length_adjustment = LENGTH_PENALTY_VALUE + (1.0 - LENGTH_PENALTY_VALUE) * length_factor
    else:
        length_adjustment = 1.0
    marks_after_length: float = raw_score * length_adjustment

    # ── Step 10: Partial-relevance cap ────────────────────────────────────
    if partially_relevant:
        marks_after_length = min(marks_after_length, PARTIAL_RELEVANCE_CAP)

    final_marks: float = round(max(0.0, min(float(MAX_MARKS), marks_after_length)), 2)

    # ── Step 11: Scale to max_marks ───────────────────────────────────────
    final_score: float = round(final_marks * max_marks / MAX_MARKS, 4)
    final_score = min(final_score, float(max_marks))

    # ── Step 12: 0.5-step rounding ────────────────────────────────────────
    rounded_score: float = round_half_step(final_score)
    rounded_score = min(rounded_score, float(max_marks))

    # ── Step 13: Feedback list ────────────────────────────────────────────
    feedback = generate_feedback(
        semantic_score=semantic_score,
        keyword_score=keyword_score,
        missing_keywords=missing_keywords,
        length_factor=length_factor,
        is_empty=False,
        is_irrelevant_answer=False,
        is_partially_relevant_answer=partially_relevant,
        total_keyword_count=len(all_kw_words),
        confidence=confidence,
        concept_score=concept_score,
        missed_concepts=missed_concepts,
        contradiction_score=contradiction_score,
    )

    return {
        "student_answer_clean":  clean_student,
        "semantic_score":        semantic_score,
        "keyword_score":         keyword_score,
        "concept_score":         concept_score,
        "contradiction_score":   round(contradiction_score, 4),
        "raw_score":             raw_score,
        "length_adjustment":     round(length_adjustment, 4),
        "final_marks":           final_marks,   # out of MAX_MARKS=10 (compat)
        "final_score":           final_score,   # scaled to max_marks
        "rounded_score":         rounded_score,
        "confidence":            round(confidence, 4),
        "matched_keywords":      matched_keywords,
        "missing_keywords":      missing_keywords,
        "covered_concepts":      covered_concepts,
        "missed_concepts":       missed_concepts,
        "feedback":              feedback,
        "length_factor":         round(length_factor, 4),
        "semantic_weight":       WEIGHT_SEMANTIC,
        "keyword_weight":        WEIGHT_KEYWORD,
        "is_irrelevant":         False,
        "is_partially_relevant": partially_relevant,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 13. Dataset Loading  (v5.0 — supports concepts + max_marks fields)
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(json_path: str) -> list[dict]:
    """
    Load and validate a Tamil evaluation dataset from a JSON file.

    Supported formats
    -----------------
    **New format** (v2 / v3 / v4 / v5) – multiple student answers per question::

        [
          {
            "question":       "...",
            "model_answer":   "...",
            "max_marks":      5,
            "keywords":       ["kw1", "kw2"]
                              or [{"word": "kw1", "weight": 2}, ...],
            "concepts": [
              ["திருவள்ளுவர்", "வள்ளுவர்"],
              ["1330"],
              ["அறம்", "பொருள்", "இன்பம்"]
            ],
            "student_answers": [
              {"answer": "...", "marks": 9},
              {"answer": "...", "marks": 5}
            ]
          }
        ]

    **Legacy format** (v1) – single student answer per question (unchanged).

    ``concepts`` is optional and defaults to [] for backward-compat.
    ``max_marks`` is optional and defaults to MAX_MARKS (10).

    Args:
        json_path: Absolute or relative path to the JSON dataset file.

    Returns:
        List of normalised entry dicts (v5 structure).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If required fields are missing in any entry.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)

    normalised: list[dict] = []
    required_base = {"question", "model_answer", "keywords"}

    for idx, entry in enumerate(raw):
        missing_fields = required_base - entry.keys()
        if missing_fields:
            raise ValueError(
                f"Entry {idx} is missing required fields: {missing_fields}"
            )

        # ── New format ──────────────────────────────────────────────────────
        if "student_answers" in entry:
            for ans_idx, sa in enumerate(entry["student_answers"]):
                if "answer" not in sa:
                    raise ValueError(
                        f"Entry {idx}, student_answers[{ans_idx}] missing 'answer' field."
                    )
            # Ensure concepts + max_marks present (v5.0 additions)
            entry.setdefault("concepts", [])
            entry.setdefault("max_marks", MAX_MARKS)
            normalised.append(entry)

        # ── Legacy format (backward-compatible) ────────────────────────────
        elif "student_answer" in entry:
            converted = {
                "question":     entry["question"],
                "model_answer": entry["model_answer"],
                "keywords":     entry["keywords"],
                "concepts":     entry.get("concepts", []),
                "max_marks":    int(entry.get("max_marks", MAX_MARKS)),
                "student_answers": [
                    {
                        "answer": entry["student_answer"],
                        "marks":  entry.get("expected_marks"),
                    }
                ],
            }
            normalised.append(converted)

        else:
            raise ValueError(
                f"Entry {idx} must have either 'student_answers' (v2/v3/v4/v5) "
                "or 'student_answer' (v1 legacy)."
            )

    return normalised


# ──────────────────────────────────────────────────────────────────────────────
# 14. Result Printing
# ──────────────────────────────────────────────────────────────────────────────

_SEP_WIDE = "=" * 80
_SEP_NARROW = "-" * 80


def print_answer_result(
    question: str,
    student_answer: str,
    result: dict[str, Any],
    answer_index: int,
    question_index: int,
    expected_marks: float | None = None,
) -> None:
    """
    Print a formatted evaluation block for one student answer.

    Includes all v3.1 fields plus the new v4.0 additions
    (``is_partially_relevant``, confidence-adjustment note).

    Args:
        question:        The Tamil question text.
        student_answer:  Original (un-processed) student answer.
        result:          Dict returned by :func:`evaluate_final`.
        answer_index:    1-based index of this answer within the question.
        question_index:  1-based question number.
        expected_marks:  Ground-truth marks if available, else None.
    """
def print_answer_result(
    question: str,
    student_answer: str,
    result: dict[str, Any],
    answer_index: int,
    question_index: int,
    expected_marks: float | None = None,
    max_marks: int = MAX_MARKS,
) -> None:
    """
    Print a formatted evaluation block for one student answer (v5.0).

    Includes all v5.0 additions: concept_score, contradiction_score,
    raw_score, length_adjustment, final_score, rounded_score, feedback list.
    """
    predicted  = result["rounded_score"]
    error = round(predicted - expected_marks, 2) if expected_marks is not None else None

    print(_SEP_WIDE)
    print(f"  Q{question_index}  ·  Answer #{answer_index}")
    print(_SEP_NARROW)

    display_ans = (
        student_answer[:120] + " …" if len(student_answer) > 122 else student_answer
    )
    print(f"  {'Question':<26}: {question}")
    print(f"  {'Student Answer':<26}: {display_ans}")
    print(_SEP_NARROW)

    # Relevance flags
    if result.get("is_irrelevant", False):
        print(f"  {'[!] Relevance':<26}: IRRELEVANT — answer is off-topic")
        print(_SEP_NARROW)
    elif result.get("is_partially_relevant", False):
        print(f"  {'[~] Relevance':<26}: PARTIAL — weakly on-topic (marks capped at {PARTIAL_RELEVANCE_CAP})")
        print(_SEP_NARROW)

    # Scores block
    print(f"  {'Semantic Score':<26}: {result['semantic_score']:>5.2f} / {MAX_MARKS}")
    print(f"  {'Keyword Score':<26}: {result['keyword_score']:>5.2f} / {MAX_MARKS}")
    print(f"  {'Concept Score':<26}: {result['concept_score']:>5.2f} / {MAX_MARKS}")
    print(f"  {'Contradiction Score':<26}: {result.get('contradiction_score', 0.0):>5.4f}")
    print(f"  {'Raw Score':<26}: {result.get('raw_score', 0.0):>5.2f} / {MAX_MARKS}")
    print(f"  {'Length Factor':<26}: {result.get('length_factor', 1.0):>5.4f}")
    print(f"  {'Length Adjustment':<26}: {result.get('length_adjustment', 1.0):>5.4f}")
    print(f"  {'Final Score (/{max_marks})':<26}: {result.get('final_score', 0.0):>5.2f}")
    print(f"  {'Rounded Score (/{max_marks})':<26}: {result['rounded_score']:>5.2f}")
    conf = result["confidence"]
    conf_label = "High" if conf >= 0.75 else "Medium" if conf >= 0.45 else "Low"
    print(f"  {'Confidence':<26}: {conf:>5.4f}  ({conf_label})")

    if expected_marks is not None:
        sign = "+" if error >= 0 else ""  # type: ignore[operator]
        print(f"  {'Expected Marks':<26}: {expected_marks:>5.2f} / {max_marks}")
        print(f"  {'Error (pred−exp)':<26}: {sign}{error:.2f}")

    # Keywords
    matched = result.get("matched_keywords", [])
    print(f"  {'Matched Keywords':<26}: {', '.join(matched) if matched else 'None'}")
    missing = result.get("missing_keywords", [])
    print(f"  {'Missing Keywords':<26}: {', '.join(missing) if missing else 'None'}")

    # Concepts
    covered = result.get("covered_concepts", [])
    missed_c = result.get("missed_concepts", [])
    if covered or missed_c:
        print(f"  {'Covered Concepts':<26}: {', '.join(covered) if covered else 'None'}")
        print(f"  {'Missed Concepts':<26}: {', '.join(missed_c) if missed_c else 'None'}")

    # Feedback list
    fb = result.get("feedback", [])
    if isinstance(fb, list):
        for i, obs in enumerate(fb, 1):
            label = f"Feedback [{i}]" if i > 1 else "Feedback"
            print(f"  {label:<26}: {obs}")
    else:
        print(f"  {'Feedback':<26}: {fb}")

    print(_SEP_WIDE)
    print()


def print_summary(
    all_results: list[dict[str, Any]],
    total_q: int,
    total_answers: int,
) -> None:
    """
    Print a global summary table including MAE, RMSE, accuracy metrics, and
    counts for irrelevant and partially-relevant answers (v5.0).
    """
    predicted_list = [r["rounded_score"] for r in all_results]
    expected_list  = [r["expected_marks"] for r in all_results if r["expected_marks"] is not None]
    error_list     = [r["error"]          for r in all_results if r["error"]          is not None]
    irrelevant_count = sum(1 for r in all_results if r.get("is_irrelevant", False))
    partial_count    = sum(1 for r in all_results if r.get("is_partially_relevant", False))
    contradiction_count = sum(
        1 for r in all_results if r.get("contradiction_score", 0.0) >= CONTRADICTION_THRESHOLD
    )

    print("\n" + _SEP_WIDE)
    print("  EVALUATION SUMMARY  ·  v5.0")
    print(_SEP_NARROW)
    print(f"  {'Questions evaluated':<38}: {total_q}")
    print(f"  {'Total student answers':<38}: {total_answers}")
    print(f"  {'Irrelevant answers detected':<38}: {irrelevant_count}")
    print(f"  {'Partially relevant answers':<38}: {partial_count}")
    print(f"  {'Contradiction detected':<38}: {contradiction_count}")
    print(_SEP_NARROW)

    print(f"  {'Avg Rounded Score':<38}: {np.mean(predicted_list):.3f}")
    print(f"  {'Min Rounded Score':<38}: {np.min(predicted_list):.3f}")
    print(f"  {'Max Rounded Score':<38}: {np.max(predicted_list):.3f}")
    print(f"  {'Std Dev (Rounded)':<38}: {np.std(predicted_list):.3f}")

    if expected_list and error_list:
        mae  = np.mean(np.abs(error_list))
        rmse = np.sqrt(np.mean(np.array(error_list) ** 2))
        print(_SEP_NARROW)
        print(f"  {'Avg Expected Marks':<38}: {np.mean(expected_list):.3f}")
        print(f"  {'Mean Absolute Error (MAE)':<38}: {mae:.3f}")
        print(f"  {'Root Mean Sq Error (RMSE)':<38}: {rmse:.3f}")
        print(f"  {'Samples with expected marks':<38}: {len(expected_list)}")
    else:
        print(_SEP_NARROW)
        print("  No expected marks found – accuracy metrics skipped.")

    print(_SEP_WIDE + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 15. Main Pipeline  (v5.0)
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(json_path: str) -> list[dict[str, Any]]:
    """
    Full evaluation pipeline (v5.0).

    1. Load & validate the dataset (supports v1 legacy and v2/v3/v4/v5 formats).
    2. Pre-warm the LaBSE sentence-transformer model.
    3. For every question, iterate over all student answers.
    4. Compute semantic (cached embedding) + keyword + concept + contradiction
       scores, confidence, length factor, matched/missing keywords, relevance
       flags, and feedback list.
    5. Print per-answer results.
    6. Print a global accuracy summary (MAE, RMSE, irrelevant/partial counts).
    7. Return all results as a flat list for downstream use.

    Args:
        json_path: Path to the JSON dataset file.

    Returns:
        Flat list of result dicts, one per student answer.
    """
    print("\n" + _SEP_WIDE)
    print("  Tamil Answer Evaluation Engine  ·  v5.0")
    print(f"  Semantic model         : {MODEL_NAME}")
    print(f"  NLI model              : {NLI_MODEL_NAME}")
    print(f"  Formula weights        : {WEIGHT_SEMANTIC:.0%} sem / "
          f"{WEIGHT_KEYWORD:.0%} kw / {WEIGHT_CONCEPT:.0%} concept")
    print(f"  Fuzzy threshold        : {FUZZY_THRESHOLD}")
    print(f"  Concept fuzzy threshold: {CONCEPT_FUZZY_THRESHOLD}")
    print(f"  Min valid words        : {MIN_VALID_WORDS}")
    print(f"  Irrelevance cutoff     : confidence < {IRRELEVANCE_THRESHOLD}")
    print(f"  Partial-rel. band      : {IRRELEVANCE_THRESHOLD} ≤ conf < {PARTIAL_RELEVANCE_THRESHOLD}")
    print(f"  Partial-rel. marks cap : {PARTIAL_RELEVANCE_CAP} / {MAX_MARKS}")
    print(f"  Length words/mark      : {BASE_WORDS_PER_MARK}")
    print(f"  Length insufficient    : < {LENGTH_INSUFFICIENT_RATIO:.0%} of expected words")
    print(f"  Rounding               : nearest 0.5 step")
    print(_SEP_WIDE + "\n")

    dataset = load_dataset(json_path)
    get_model()

    all_results: list[dict[str, Any]] = []
    total_answers: int = 0

    for q_idx, entry in enumerate(dataset, start=1):
        question    = entry["question"]
        model_answer = entry["model_answer"]
        keywords    = entry["keywords"]
        concepts    = entry.get("concepts", [])
        max_marks   = int(entry.get("max_marks", MAX_MARKS))

        # Pre-cache model-answer embedding for this question
        get_model_embedding_cached(preprocess_text(model_answer))

        for a_idx, student_entry in enumerate(entry["student_answers"], start=1):
            raw_answer = student_entry["answer"]
            expected_marks: float | None = (
                float(student_entry["marks"])
                if student_entry.get("marks") is not None
                else None
            )

            result = evaluate_final(
                raw_answer, model_answer, keywords,
                concepts=concepts, max_marks=max_marks,
            )

            error: float | None = (
                round(result["rounded_score"] - expected_marks, 2)
                if expected_marks is not None
                else None
            )

            print_answer_result(
                question=question,
                student_answer=raw_answer,
                result=result,
                answer_index=a_idx,
                question_index=q_idx,
                expected_marks=expected_marks,
                max_marks=max_marks,
            )

            all_results.append({
                "question":              question,
                "student_answer":        raw_answer,
                "semantic_score":        result["semantic_score"],
                "keyword_score":         result["keyword_score"],
                "concept_score":         result["concept_score"],
                "contradiction_score":   result["contradiction_score"],
                "raw_score":             result["raw_score"],
                "length_adjustment":     result["length_adjustment"],
                "final_marks":           result["final_marks"],
                "final_score":           result["final_score"],
                "rounded_score":         result["rounded_score"],
                "confidence":            result["confidence"],
                "length_factor":         result["length_factor"],
                "matched_keywords":      result["matched_keywords"],
                "missing_keywords":      result["missing_keywords"],
                "covered_concepts":      result["covered_concepts"],
                "missed_concepts":       result["missed_concepts"],
                "feedback":              result["feedback"],
                "is_irrelevant":         result["is_irrelevant"],
                "is_partially_relevant": result["is_partially_relevant"],
                "expected_marks":        expected_marks,
                "error":                 error,
            })
            total_answers += 1

    print_summary(all_results, total_q=len(dataset), total_answers=total_answers)
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "tamil_dataset.json"
    run_evaluation(dataset_path)
