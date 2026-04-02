"""
OCR Evaluation Pipeline  ·  v4.0
=================================
Full automated evaluation pipeline for handwritten Tamil answer sheets.

NEW in v4.0 (over v3.0)
------------------------
  • evaluate_from_separate_uploads()  – NEW entry-point for the separate-upload
      mode where question paper and answer sheet are uploaded independently.
      Accepts:
        questions : List[{"question_number": str, "question": str}]
        answers   : List[{"question_number": str, "answer": str, "ocr_confidence": float}]
        dataset   : List[question-bank dicts]   (for model answers + keywords)
      Performs:
        1. qno-normalised matching of OCR questions → dataset (semantic)
        2. qno-normalised matching of OCR answers → OCR questions
        3. evaluate_single_answer() for each matched pair
        4. Returns the same output structure as evaluate_full_sheet()

All v3.0 functions (evaluate_full_sheet, evaluate_single_answer,
match_question_to_dataset, preprocess_ocr_text) are unchanged and
continue to work as before for the combined-upload mode.

Input for separate mode:
  questions : [{"question_number":"1","question":"..."}]
  answers   : [{"question_number":"1","answer":"...","ocr_confidence":0.9}]
  dataset   : [{"question":"...","model_answer":"...","keywords":[...],"max_marks":5}]

Output (identical shape for both modes):
  {
    "total_marks":     float,
    "max_total_marks": float,
    "percentage":      float,
    "details": [
      {
        "question_number":   str,
        "question":          str,
        "student_answer":    str,
        "matched_question":  str,
        "match_confidence":  float,
        "predicted_marks":   float,
        "max_marks":         int,
        "score_out_of_10":   float,
        "feedback":          str,
        "confidence":        float,
        "matched_keywords":  list,
        "missing_keywords":  list,
        "status":            "evaluated"|"skipped"|"no_answer"|"error",
        "ocr_confidence":    float,
      }
    ]
  }
"""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from evaluator_final import (
    _fuzzy_keyword_ratio,
    _normalise_keywords,
    compute_partial_keyword_score,
    evaluate_final,
    evaluate_keywords,
    evaluate_semantic,
    generate_feedback,
    get_model,
    get_model_embedding_cached,
    FUZZY_THRESHOLD,
    IRRELEVANCE_THRESHOLD,
    MAX_MARKS,
    MIN_VALID_WORDS,
    PARTIAL_KW_THRESHOLD,
    preprocess_text,
)
from sklearn.metrics.pairwise import cosine_similarity

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _indic_transliterate
    _PHONETIC_AVAILABLE = True
except ImportError:
    _PHONETIC_AVAILABLE = False

# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def _configure_logging(level: int = logging.INFO, log_file: str = "evaluation.log") -> None:
    if logger.handlers:
        return
    logger.setLevel(level)
    _fmt = "%(asctime)s - %(levelname)s - %(message)s"
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    if log_file:
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(fh)
        except OSError as exc:
            logger.warning("Could not open log file '%s': %s", log_file, exc)


_configure_logging()

# ── configuration ─────────────────────────────────────────────────────────────
QUESTION_MATCH_THRESHOLD: float = 0.5
PHONETIC_MATCH_THRESHOLD: float = 0.7


# ══════════════════════════════════════════════════════════════════════════════
# QNO Normalisation helper (shared by both modes)
# ══════════════════════════════════════════════════════════════════════════════

def _canonical_qno(raw: str) -> str:
    """
    Convert a raw question-number string to a canonical integer-string form.

    "Q1", "Q.1", "01", "(1)", "1.", "1)" all normalise to "1".
    Roman / alpha qnos are lowercased but not converted to integers.
    """
    import unicodedata as _ud
    qno = _ud.normalize("NFC", str(raw)).strip()
    qno = re.sub(r"^\((.+)\)$", r"\1", qno)            # strip surrounding brackets
    qno = re.sub(r"^[Qq][.\-\s]?\s*", "Q", qno)         # normalise Q prefix
    qno = re.sub(r"[\s.):]+$", "", qno)                  # strip trailing delimiters
    qno = re.sub(r"^(Q?)0+(\d)", r"\1\2", qno)           # strip leading zeros
    # Lowercase everything except the Q prefix
    if not (qno.startswith("Q") and len(qno) > 1 and qno[1:].isdigit()):
        qno = qno.lower()
    return qno.strip()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – OCR TEXT PREPROCESSING  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════

TANGLISH_TO_TAMIL: dict[str, str] = {
    "satyagraha": "சத்தியாக்கிரகம்", "satyagrahi": "சத்தியாக்கிரகி",
    "ahimsa": "அஹிம்சை", "swadeshi": "சுதேசி", "swaraj": "சுராஜ்",
    "gandhiji": "காந்தியடிகள்", "bharatham": "பாரதம்", "indhiya": "இந்தியா",
    "vidudhali": "விடுதலை", "porattam": "போராட்டம்",
    "tholkappiyam": "தொல்காப்பியம்", "sangam": "சங்கம்",
    "ilakkiyam": "இலக்கியம்", "ilakkanam": "இலக்கணம்",
    "kural": "குறள்", "thirukkural": "திருக்குறள்",
    "avvaiyar": "ஔவையார்", "valluvar": "வள்ளுவர்",
    "neer": "நீர்", "kaatru": "காற்று", "maram": "மரம்",
    "suryan": "சூரியன்",
}

TAMIL_SPELL_CORRECTIONS: dict[str, str] = {
    "இனதியா": "இந்தியா", "இனதிய": "இந்திய",
    "தமிழனாட": "தமிழ்நாடு", "தமிழனாடு": "தமிழ்நாடு",
    "கானதியடிகள்": "காந்தியடிகள்", "கானதி": "காந்தி",
    "இலககியம": "இலக்கியம்", "இலககணம": "இலக்கணம்",
    "சதியாககிரகம": "சத்தியாக்கிரகம்",
    "விடதலை": "விடுதலை", "போராடம": "போராட்டம்",
    "தோலகாபியம": "தொல்காப்பியம்", "திருககுரல": "திருக்குறள்",
    "அவைவயார": "ஔவையார்", "வளளுவர": "வள்ளுவர்",
}

_RE_REPEATED_CHARS = re.compile(r"(.)\1{2,}", re.UNICODE)
_RE_STRAY_SYMBOLS  = re.compile(r"[^\u0B80-\u0BFF\w\s।?!,\.\-'\"():;]", re.UNICODE)


def _remove_repeated_chars(text: str) -> str:
    return _RE_REPEATED_CHARS.sub(r"\1\1", text)


def _fix_mixed_script_glue(text: str) -> str:
    text = re.sub(r"([\u0B80-\u0BFF])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])([\u0B80-\u0BFF])", r"\1 \2", text)
    return text


def _apply_tanglish_conversion(text: str) -> str:
    for tanglish, tamil in TANGLISH_TO_TAMIL.items():
        text = re.sub(r"\b" + re.escape(tanglish) + r"\b", tamil,
                      text, flags=re.IGNORECASE)
    return text


def _apply_spell_correction(text: str) -> str:
    for wrong, correct in TAMIL_SPELL_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text


def preprocess_ocr_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _fix_mixed_script_glue(text)
    text = _RE_STRAY_SYMBOLS.sub(" ", text)
    text = _remove_repeated_chars(text)
    text = _apply_tanglish_conversion(text)
    text = _apply_spell_correction(text)
    text = re.sub(r"[\s\u00A0\u200B]+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – PHONETIC KEYWORD MATCHING  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════

_phonetic_cache: dict[str, str] = {}


def _to_phonetic(text: str) -> str:
    if not _PHONETIC_AVAILABLE:
        return text
    if text in _phonetic_cache:
        return _phonetic_cache[text]
    try:
        result = _indic_transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
    except Exception:
        result = text
    _phonetic_cache[text] = result
    return result


def compute_partial_keyword_score_phonetic(
    student_answer: str,
    keywords: list,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
    partial_threshold: float = PARTIAL_KW_THRESHOLD,
) -> tuple[float, list[str], list[str]]:
    base_score, matched, missing = compute_partial_keyword_score(
        student_answer, keywords, fuzzy_threshold, partial_threshold
    )
    if not _PHONETIC_AVAILABLE or not missing:
        return base_score, matched, missing

    norm_kws       = _normalise_keywords(keywords)
    kw_weight_map  = {kw["word"]: kw["weight"] for kw in norm_kws}
    total_weight   = sum(kw["weight"] for kw in norm_kws)
    phonetic_answer = _to_phonetic(student_answer)
    phonetic_extra: float = 0.0
    still_missing:  list[str] = []

    for kw_word in missing:
        kw_clean    = preprocess_text(kw_word)
        phonetic_kw = _to_phonetic(kw_clean)
        ph_ratio    = _fuzzy_keyword_ratio(phonetic_kw, phonetic_answer)
        if ph_ratio >= PHONETIC_MATCH_THRESHOLD:
            phonetic_extra += kw_weight_map.get(kw_word, 1.0) * ph_ratio
        still_missing.append(kw_word)

    if phonetic_extra == 0.0:
        return base_score, matched, still_missing

    base_raw     = (base_score / MAX_MARKS) * total_weight if total_weight > 0 else 0.0
    combined_raw = base_raw + phonetic_extra
    new_score    = round(min((combined_raw / total_weight) * MAX_MARKS, float(MAX_MARKS)), 2) if total_weight > 0 else base_score
    return new_score, matched, still_missing


if _PHONETIC_AVAILABLE:
    import evaluator_final as _ef
    _ef.compute_partial_keyword_score = compute_partial_keyword_score_phonetic
    logger.info("Phonetic keyword matching enabled.")
else:
    logger.info("Phonetic matching disabled. pip install indic-transliteration to enable.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DATASET QUESTION MATCHING  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════

def match_question_to_dataset(
    question: str,
    dataset: list[dict],
) -> tuple[dict | None, float]:
    if not dataset or not question.strip():
        return None, 0.0
    transformer = get_model()
    ocr_cleaned = preprocess_ocr_text(question)
    clean_q     = preprocess_text(ocr_cleaned)
    student_q_emb = transformer.encode(
        clean_q, convert_to_numpy=True, normalize_embeddings=True
    )
    best_score: float = -1.0
    best_entry: dict | None = None
    for entry in dataset:
        ds_q_clean = preprocess_text(entry.get("question", ""))
        ds_q_emb   = get_model_embedding_cached(ds_q_clean)
        score = float(cosine_similarity(
            student_q_emb.reshape(1, -1), ds_q_emb.reshape(1, -1)
        )[0][0])
        if score > best_score:
            best_score = score; best_entry = entry

    best_score = round(best_score, 4)
    if best_score < QUESTION_MATCH_THRESHOLD:
        return None, best_score
    return best_entry, best_score


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – PER-QUESTION EVALUATION  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_single_answer(
    answer: str,
    dataset_entry: dict,
) -> dict[str, Any]:
    max_marks    = int(dataset_entry.get("max_marks", 10))
    model_answer = dataset_entry.get("model_answer", "")
    keywords     = dataset_entry.get("keywords", [])

    if not answer or not isinstance(answer, str) or not answer.strip():
        return {
            "score_out_of_10": 0.0, "scaled_marks": 0.0, "max_marks": max_marks,
            "feedback": "No answer provided. Please attempt the question.",
            "matched_keywords": [],
            "missing_keywords": [kw if isinstance(kw, str) else kw.get("word", "") for kw in keywords],
            "confidence": 0.0, "is_irrelevant": False,
            "is_partially_relevant": False, "answer_clean": "",
        }

    ocr_cleaned = preprocess_ocr_text(answer)
    result      = evaluate_final(
        student_answer=ocr_cleaned, model_answer=model_answer, keywords=keywords
    )
    score_out_of_10 = result["final_marks"]
    scaled_marks    = round((score_out_of_10 / 10.0) * max_marks, 2)

    return {
        "score_out_of_10":       score_out_of_10,
        "scaled_marks":          scaled_marks,
        "max_marks":             max_marks,
        "feedback":              result["feedback"],
        "matched_keywords":      result["matched_keywords"],
        "missing_keywords":      result["missing_keywords"],
        "confidence":            result["confidence"],
        "is_irrelevant":         result["is_irrelevant"],
        "is_partially_relevant": result["is_partially_relevant"],
        "answer_clean":          result["student_answer_clean"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – ORIGINAL FULL SHEET PIPELINE  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_full_sheet(
    ocr_json: list[dict],
    dataset:  list[dict],
) -> dict[str, Any]:
    """
    Combined-mode pipeline: OCR JSON (questions + answers in one sheet) → result.

    Input: [{"question_number":"Q1","question":"...","answer":"..."}]
    """
    get_model()
    details: list[dict[str, Any]] = []
    total_marks:     float = 0.0
    max_total_marks: float = 0.0

    logger.info("=" * 70)
    logger.info("OCR Evaluation Pipeline v4.0 — COMBINED MODE")
    logger.info("OCR entries: %d | Dataset: %d", len(ocr_json), len(dataset))
    logger.info("=" * 70)

    for ocr_entry in ocr_json:
        q_number     = str(ocr_entry.get("question_number", "?"))
        raw_question = str(ocr_entry.get("question", "")).strip()
        raw_answer   = str(ocr_entry.get("answer", "")).strip()
        logger.info("Processing %s …", q_number)

        try:
            if not raw_question:
                logger.warning("%s: SKIPPED — question text missing.", q_number)
                details.append(_skipped_entry(q_number, "", "", 0.0,
                               "Question text missing in OCR output — skipped."))
                continue

            try:
                matched_entry, match_conf = match_question_to_dataset(raw_question, dataset)
            except Exception as exc:
                logger.error("%s: Matching error: %s", q_number, exc, exc_info=True)
                details.append(_error_entry(q_number, raw_question))
                continue

            if matched_entry is None:
                logger.warning("%s: SKIPPED — match_confidence=%.3f", q_number, match_conf)
                details.append(_skipped_entry(q_number, raw_question, "", match_conf,
                               "Question could not be matched to dataset."))
                continue

            matched_q_text = matched_entry.get("question", "")
            q_max_marks    = int(matched_entry.get("max_marks", 10))
            max_total_marks += q_max_marks

            if not raw_answer:
                logger.warning("%s: NO ANSWER — 0 / %d.", q_number, q_max_marks)
                details.append(_no_answer_entry(
                    q_number, raw_question, matched_q_text, match_conf,
                    q_max_marks, matched_entry.get("keywords", [])
                ))
                continue

            try:
                eval_result    = evaluate_single_answer(raw_answer, matched_entry)
            except Exception as exc:
                logger.error("%s: Evaluation error: %s", q_number, exc, exc_info=True)
                details.append(_error_entry(q_number, raw_question, matched_q_text,
                                            match_conf, q_max_marks))
                continue

            predicted_marks = eval_result["scaled_marks"]
            total_marks    += predicted_marks
            logger.info("%s: %.2f / %d (score/10=%.2f)", q_number,
                        predicted_marks, q_max_marks, eval_result["score_out_of_10"])

            details.append({
                "question_number":  q_number,
                "question":         raw_question,
                "student_answer":   raw_answer,
                "matched_question": matched_q_text,
                "match_confidence": match_conf,
                "predicted_marks":  predicted_marks,
                "max_marks":        q_max_marks,
                "score_out_of_10":  eval_result["score_out_of_10"],
                "feedback":         eval_result["feedback"],
                "confidence":       eval_result["confidence"],
                "matched_keywords": eval_result["matched_keywords"],
                "missing_keywords": eval_result["missing_keywords"],
                "status":           "evaluated",
                "ocr_confidence":   0.0,
            })

        except Exception as exc:
            logger.error("%s: Unexpected error: %s", q_number, exc, exc_info=True)
            details.append(_error_entry(q_number, raw_question))

    return _build_result(total_marks, max_total_marks, details)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – NEW: SEPARATE UPLOAD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_from_separate_uploads(
    questions: list[dict[str, str]],
    answers:   list[dict[str, Any]],
    dataset:   list[dict],
) -> dict[str, Any]:
    """
    Separate-upload pipeline: independently uploaded question paper + answer sheet.

    NEW in v4.0. Called by the new /evaluate_separate Flask endpoint.

    Algorithm
    ---------
    1. Build qno-indexed lookup for both questions and answers.
    2. For each question from the question paper (in order):
       a. Match question text → dataset (semantic, same as combined mode).
       b. Look up corresponding answer by qno.
       c. Evaluate answer against matched dataset entry.
    3. Answers whose qno has no matching question are appended as "invalid".

    Args:
        questions : List of {"question_number": str, "question": str}
                    from process_question_paper() / question_paper_extractor.
        answers   : List of {"question_number": str, "answer": str,
                    "ocr_confidence": float}
                    from process_answer_sheet_only() / answer_sheet_extractor.
        dataset   : Question bank list with model_answer, keywords, max_marks.

    Returns:
        Same output dict as evaluate_full_sheet() — fully compatible with
        the existing frontend rendering logic.
    """
    get_model()

    # ── Build qno → record lookups ────────────────────────────────────────────
    q_index: dict[str, str] = {
        _canonical_qno(q["question_number"]): q.get("question", "")
        for q in questions
        if q.get("question_number")
    }
    a_index: dict[str, dict] = {
        _canonical_qno(a["question_number"]): a
        for a in answers
        if a.get("question_number")
    }

    logger.info("=" * 70)
    logger.info("OCR Evaluation Pipeline v4.0 — SEPARATE UPLOAD MODE")
    logger.info("Questions: %d | Answers: %d | Dataset: %d",
                len(q_index), len(a_index), len(dataset))
    logger.info("=" * 70)

    details:         list[dict[str, Any]] = []
    total_marks:     float = 0.0
    max_total_marks: float = 0.0
    seen_qnos:       set[str] = set()

    # ── Pass 1: iterate question paper order ──────────────────────────────────
    for q_entry in questions:
        raw_qno      = q_entry.get("question_number", "?")
        canonical    = _canonical_qno(raw_qno)
        raw_question = q_index.get(canonical, "").strip()
        seen_qnos.add(canonical)

        logger.info("Processing Q%s …", raw_qno)

        try:
            if not raw_question:
                logger.warning("Q%s: SKIPPED — empty question text.", raw_qno)
                details.append(_skipped_entry(
                    f"Q{raw_qno}", "", "", 0.0,
                    "Question text missing — skipped."
                ))
                continue

            # Match question → dataset
            try:
                matched_entry, match_conf = match_question_to_dataset(raw_question, dataset)
            except Exception as exc:
                logger.error("Q%s: Matching error: %s", raw_qno, exc, exc_info=True)
                details.append(_error_entry(f"Q{raw_qno}", raw_question))
                continue

            if matched_entry is None:
                logger.warning("Q%s: SKIPPED — match_confidence=%.3f", raw_qno, match_conf)
                details.append(_skipped_entry(
                    f"Q{raw_qno}", raw_question, "", match_conf,
                    "Question could not be matched to dataset."
                ))
                continue

            matched_q_text = matched_entry.get("question", "")
            q_max_marks    = int(matched_entry.get("max_marks", 10))
            max_total_marks += q_max_marks

            # Look up student answer by qno
            ans_record  = a_index.get(canonical, {})
            raw_answer  = str(ans_record.get("answer", "")).strip()
            ocr_conf    = float(ans_record.get("ocr_confidence", 0.0))

            if not raw_answer:
                logger.warning("Q%s: NO ANSWER — 0 / %d.", raw_qno, q_max_marks)
                details.append({
                    **_no_answer_entry(
                        f"Q{raw_qno}", raw_question, matched_q_text,
                        match_conf, q_max_marks, matched_entry.get("keywords", [])
                    ),
                    "ocr_confidence": ocr_conf,
                })
                continue

            # Evaluate
            try:
                eval_result = evaluate_single_answer(raw_answer, matched_entry)
            except Exception as exc:
                logger.error("Q%s: Evaluation error: %s", raw_qno, exc, exc_info=True)
                details.append(_error_entry(
                    f"Q{raw_qno}", raw_question, matched_q_text, match_conf, q_max_marks
                ))
                continue

            predicted_marks  = eval_result["scaled_marks"]
            total_marks     += predicted_marks
            logger.info("Q%s: %.2f / %d (score/10=%.2f, match_conf=%.3f)",
                        raw_qno, predicted_marks, q_max_marks,
                        eval_result["score_out_of_10"], match_conf)

            details.append({
                "question_number":  f"Q{raw_qno}",
                "question":         raw_question,
                "student_answer":   raw_answer,
                "matched_question": matched_q_text,
                "match_confidence": match_conf,
                "predicted_marks":  predicted_marks,
                "max_marks":        q_max_marks,
                "score_out_of_10":  eval_result["score_out_of_10"],
                "feedback":         eval_result["feedback"],
                "confidence":       eval_result["confidence"],
                "matched_keywords": eval_result["matched_keywords"],
                "missing_keywords": eval_result["missing_keywords"],
                "status":           "evaluated",
                "ocr_confidence":   ocr_conf,
            })

        except Exception as exc:
            logger.error("Q%s: Unexpected error: %s", raw_qno, exc, exc_info=True)
            details.append(_error_entry(f"Q{raw_qno}", raw_question))

    # ── Pass 2: answers with no matching question (invalid) ───────────────────
    for a_entry in answers:
        canonical = _canonical_qno(a_entry.get("question_number", ""))
        if not canonical or canonical in seen_qnos:
            continue
        raw_answer = str(a_entry.get("answer", "")).strip()
        ocr_conf   = float(a_entry.get("ocr_confidence", 0.0))
        logger.warning("Q%s: INVALID — no matching question.", a_entry.get("question_number"))
        details.append({
            "question_number":  f"Q{a_entry.get('question_number','')}",
            "question":         "",
            "student_answer":   raw_answer,
            "matched_question": "",
            "match_confidence": 0.0,
            "predicted_marks":  0.0,
            "max_marks":        0,
            "score_out_of_10":  0.0,
            "feedback":         "Answer has no corresponding question on the paper.",
            "confidence":       0.0,
            "matched_keywords": [],
            "missing_keywords": [],
            "status":           "skipped",
            "ocr_confidence":   ocr_conf,
        })

    result = _build_result(total_marks, max_total_marks, details)
    logger.info("SEPARATE MODE RESULT: %.2f / %.2f (%.1f%%)",
                result["total_marks"], result["max_total_marks"], result["percentage"])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Shared result-building helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_result(
    total_marks:     float,
    max_total_marks: float,
    details:         list[dict],
) -> dict[str, Any]:
    total_marks     = round(total_marks, 2)
    max_total_marks = round(max_total_marks, 2)
    percentage      = round((total_marks / max_total_marks * 100), 1) if max_total_marks > 0 else 0.0

    ev = sum(1 for d in details if d["status"] == "evaluated")
    na = sum(1 for d in details if d["status"] == "no_answer")
    sk = sum(1 for d in details if d["status"] == "skipped")
    er = sum(1 for d in details if d["status"] == "error")

    logger.info("=" * 70)
    logger.info("RESULT: %d total | %d evaluated | %d no_answer | %d skipped | %d error",
                len(details), ev, na, sk, er)
    logger.info("Marks: %.2f / %.2f  (%.1f%%)", total_marks, max_total_marks, percentage)
    logger.info("=" * 70)

    return {
        "total_marks":     total_marks,
        "max_total_marks": max_total_marks,
        "percentage":      percentage,
        "details":         details,
    }


def _skipped_entry(
    q_number: str, raw_question: str,
    matched_q: str, match_conf: float, feedback: str,
) -> dict[str, Any]:
    return {
        "question_number":  q_number, "question": raw_question,
        "student_answer":   "", "matched_question": matched_q,
        "match_confidence": match_conf, "predicted_marks": 0.0,
        "max_marks": 0, "score_out_of_10": 0.0, "feedback": feedback,
        "confidence": 0.0, "matched_keywords": [], "missing_keywords": [],
        "status": "skipped", "ocr_confidence": 0.0,
    }


def _no_answer_entry(
    q_number: str, raw_question: str, matched_q: str,
    match_conf: float, q_max_marks: int, keywords: list,
) -> dict[str, Any]:
    return {
        "question_number":  q_number, "question": raw_question,
        "student_answer":   "", "matched_question": matched_q,
        "match_confidence": match_conf, "predicted_marks": 0.0,
        "max_marks": q_max_marks, "score_out_of_10": 0.0,
        "feedback": "No answer provided. Please attempt the question.",
        "confidence": 0.0,
        "matched_keywords": [],
        "missing_keywords": [kw if isinstance(kw, str) else kw.get("word", "") for kw in keywords],
        "status": "no_answer", "ocr_confidence": 0.0,
    }


def _error_entry(
    q_number: str, raw_question: str = "",
    matched_q: str = "", match_conf: float = 0.0, q_max_marks: int = 0,
) -> dict[str, Any]:
    return {
        "question_number":  q_number, "question": raw_question,
        "student_answer":   "", "matched_question": matched_q,
        "match_confidence": match_conf, "predicted_marks": 0.0,
        "max_marks": q_max_marks, "score_out_of_10": 0.0,
        "feedback": "Evaluation error — question skipped.",
        "confidence": 0.0, "matched_keywords": [], "missing_keywords": [],
        "status": "error", "ocr_confidence": 0.0,
    }
