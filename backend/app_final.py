"""
app.py  ·  Tamil Answer Sheet Evaluation System  ·  Flask Backend  v3.2
========================================================================
REST endpoints consumed by the frontend.

Endpoints
---------
  POST /evaluate            – COMBINED mode (legacy – unchanged)
  POST /evaluate_separate   – SEPARATE mode: question paper PDF + answer sheet PDF
                              Uses the new extractor → mapper → evaluator pipeline.
  GET  /demo                – Hard-coded demo for combined mode.
  GET  /demo_separate       – Hard-coded demo for separate mode.
  GET  /health              – Liveness check.
  GET  /                    – Serve frontend/index.html.

Pipeline (/evaluate_separate):
  question_paper_extractor.extract_questions_from_upload()
      → answer_sheet_extractor.extract_answers_from_upload()
      → qa_mapper.map_questions_to_answers()
      → evaluator_final.evaluate_final()   (per attempted question)

Changes in v3.2 (evaluator_final.py v5.0 integration)
------------------------------------------------------
  IMPROVEMENT 3 – _load_question_bank now reads the 'concepts' field
    (list[list[str]]) from question_bank.json and passes it to
    evaluate_final() so concept coverage scoring is active end-to-end.
  IMPROVEMENT 4 – _evaluate_mapped_items passes concepts + max_marks to
    evaluate_final(); uses rounded_score (0.5-step) from v5.0 result
    directly instead of manually re-scaling final_marks.
  IMPROVEMENT 5 – Response detail dicts now include all v5.0 breakdown
    fields: concept_score, contradiction_score, raw_score,
    length_adjustment, final_score, rounded_score, covered_concepts,
    missed_concepts, feedback_list.

Run
---
  pip install flask flask-cors
  python app.py
  open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
import re

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

UPLOADS_DIR        = BASE_DIR / "uploads"
OUTPUTS_DIR        = BASE_DIR / "outputs"
QUESTION_BANK_PATH = (
    BASE_DIR / "question_bank.json"
    if (BASE_DIR / "question_bank.json").exists()
    else BASE_DIR / "data" / "question_bank.json"
)

for d in [UPLOADS_DIR, OUTPUTS_DIR,
          OUTPUTS_DIR / "ocr_json", OUTPUTS_DIR / "reports"]:
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tiff", ".tif", ".gif", ".pdf",
}

# ── Improvement 2: similarity threshold for dataset question matching ─────────
# Scores below this value are treated as "no match".
# 0.75 minimum + must lead second-best by at least _BANK_MARGIN to prevent
# closely-related questions (e.g. "திருக்குறள்" vs "தமிழ் இலக்கியம்") from
# stealing each other's keywords.
_BANK_MATCH_THRESHOLD: float = 0.75
_BANK_MARGIN: float = 0.08   # best score must beat 2nd-best by this much

from logging_config import configure_logging, get_logger
configure_logging(log_dir=BASE_DIR / "logs", level=logging.INFO)
log = get_logger(__name__)

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "frontend"),
    static_url_path="",
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit
CORS(app)


# ════════════════════════════════════════════════════════════════════════════
# Helpers  (original, unchanged)
# ════════════════════════════════════════════════════════════════════════════

def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _load_question_bank() -> list[dict[str, Any]]:
    """Load question_bank.json; returns [] when file is absent."""
    if not QUESTION_BANK_PATH.exists():
        log.warning("question_bank.json not found at %s", QUESTION_BANK_PATH)
        return []
    with open(QUESTION_BANK_PATH, "r", encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)
    bank = [{
        "question":     e.get("question",     ""),
        "model_answer": e.get("model_answer", ""),
        "keywords":     e.get("keywords",     []),
        "concepts":     e.get("concepts",     []),   # v5.0 — concept groups
        "max_marks":    int(e.get("max_marks", 10)),
    } for e in raw]
    log.info("Question bank loaded: %d question(s).", len(bank))
    return bank


def _save_file(file_storage, prefix: str, student_id: str, ts: str) -> Path:
    """Save a FileStorage object and return its path."""
    safe_student_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", student_id)
    suffix    = Path(file_storage.filename).suffix.lower()
    safe_name = f"{prefix}_{safe_student_id}_{ts}{suffix}"
    save_path = UPLOADS_DIR / safe_name
    file_storage.save(str(save_path))
    log.info("Saved %s → %s (%d bytes)", prefix, save_path, save_path.stat().st_size)
    return save_path


def _ocr_to_pipeline(ocr_payload: dict[str, Any]) -> list[dict[str, str]]:
    """Convert OCR engine output to evaluate_full_sheet() input format."""
    result: list[dict[str, str]] = []
    for ans in ocr_payload.get("answers", []):
        raw_num = str(ans.get("question_number", "?"))
        q_num   = raw_num if raw_num.upper().startswith("Q") else f"Q{raw_num}"
        result.append({
            "question_number": q_num,
            "question":        ans.get("question", ""),
            "answer":          ans.get("student_text", ""),
        })
    return result


def _build_response(
    eval_result: dict[str, Any],
    student_id:  str,
    ocr_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge OCR metadata and evaluation results into a single response dict."""
    ocr_by_qnum: dict[str, dict] = {}
    if ocr_payload:
        for ans in ocr_payload.get("answers", []):
            raw = str(ans.get("question_number", "?"))
            key = raw if raw.upper().startswith("Q") else f"Q{raw}"
            ocr_by_qnum[key] = ans

    enriched: list[dict[str, Any]] = []
    for detail in eval_result.get("details", []):
        q_num   = detail.get("question_number", "")
        ocr_ans = ocr_by_qnum.get(q_num, {})
        enriched.append({
            **detail,
            "question":       detail.get("question")       or ocr_ans.get("question",     ""),
            "student_answer": detail.get("student_answer") or ocr_ans.get("student_text", ""),
            "ocr_confidence": round(float(
                detail.get("ocr_confidence") or ocr_ans.get("ocr_confidence", 0.0)
            ), 4),
        })

    return {
        "status":     "success",
        "student_id": student_id,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_marks":     eval_result.get("total_marks",     0.0),
            "max_total_marks": eval_result.get("max_total_marks", 0.0),
            "percentage":      eval_result.get("percentage",      0.0),
        },
        "details": enriched,
    }


# ════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1 — module-level _FileLike with safe file-copy guards
# ════════════════════════════════════════════════════════════════════════════

class _FileLike:
    """
    Minimal shim that lets extract_*_from_upload() accept a file already
    saved to disk (werkzeug FileStorage interface: .filename + .save()).

    Defined once at module level — no duplication inside the route.

    .save() validates the source path before copying and raises a descriptive
    RuntimeError on any OS-level failure, so the surrounding try/except in
    evaluate_separate returns a clean 500 JSON response instead of a raw
    exception traceback.
    """

    def __init__(self, path: Path) -> None:
        self.filename = path.name
        self._path    = path

    def save(self, dest: str) -> None:
        # Guard 1: source must exist
        if not self._path.exists():
            raise RuntimeError(
                f"Source file does not exist: {self._path}"
            )
        # Guard 2: source must be a regular file, not a directory
        if not self._path.is_file():
            raise RuntimeError(
                f"Source path is not a regular file: {self._path}"
            )
        # Guard 3: ensure destination directory exists
        Path(dest).parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(str(self._path), dest)
        except OSError as exc:
            raise RuntimeError(
                f"File copy failed ({self._path.name} → {dest}): {exc}"
            ) from exc


# ════════════════════════════════════════════════════════════════════════════
# v3.0 / v3.1: Extractor + Mapper + Evaluator integration helpers
# ════════════════════════════════════════════════════════════════════════════

# IMPROVEMENT 2 — semantic bank matching with cosine similarity + threshold
def _find_bank_entry(
    qno: str,
    question_text: str,
    bank: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Find the best-matching question bank entry for a given extracted question.

    Match strategy (in priority order):
      1. Exact string match (case-insensitive, stripped) — zero cost, instant.
      2. Semantic cosine similarity via the SentenceTransformer model already
         loaded in evaluator_final.  Reuses get_model_embedding_cached so
         bank question embeddings are computed only once per process,
         regardless of how many answer sheets are evaluated in the session.
      3. Returns None if the best score is below _BANK_MATCH_THRESHOLD (0.55).

    Why semantic matching beats substring matching
    -----------------------------------------------
    Substring containment fails when two bank questions share a common phrase
    (e.g. "சங்கம் பற்றி எழுதுக" vs "சங்கம் காலம் பற்றி எழுதுக") because
    the shorter string is always contained in the longer one. Cosine similarity
    scores the full meaning of both strings and picks the closest one.

    Args:
        qno:           Canonical question number (used only for log messages).
        question_text: Question text from the question paper extractor.
        bank:          Loaded question_bank.json entries.

    Returns:
        Best-matching bank entry dict, or None if no match exceeds the threshold.
    """
    if not question_text or not bank:
        return None

    q_stripped = question_text.strip()
    # Normalize for exact matching: lowercase + strip trailing punctuation
    # so OCR variants like "திருக்குறள் பற்றி முழுமையான விளக்கம் தருக."
    # match bank entries ending with or without "."
    import re as _re
    def _norm(s: str) -> str:
        return _re.sub(r"[.?!।\s]+$", "", s.strip().lower())

    q_norm = _norm(q_stripped)

    # ── Step 1: EXACT MATCH (no embedding needed — always tried first) ─────
    for entry in bank:
        if _norm(entry["question"]) == q_norm:
            log.debug("qno=%r → exact bank match.", qno)
            return entry

    # ── Steps 2 & 3: SEMANTIC MATCH with threshold + margin gates ─────────
    # Only reached when no exact match exists.
    try:
        from evaluator_final import get_model_embedding_cached, preprocess_text
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError as exc:
        # evaluator_final not available (e.g. unit-test environment).
        # Degrade gracefully rather than crashing.
        log.warning(
            "Semantic bank matching unavailable (%s); returning no match for qno=%r.",
            exc, qno,
        )
        return None

    q_clean = preprocess_text(q_stripped)
    if not q_clean:
        log.warning("qno=%r question is empty after preprocessing — no match.", qno)
        return None

    q_emb: np.ndarray = get_model_embedding_cached(q_clean)

    # Collect ALL scores so we can apply a margin check (best must beat 2nd-best
    # by _BANK_MARGIN). This prevents closely-related questions from stealing
    # each other's keywords when their similarity scores are very close.
    scored: list[tuple[float, dict]] = []

    for entry in bank:
        bank_q_clean = preprocess_text(entry["question"].strip())
        if not bank_q_clean:
            continue
        bank_emb = get_model_embedding_cached(bank_q_clean)
        score: float = float(
            cosine_similarity(q_emb.reshape(1, -1), bank_emb.reshape(1, -1))[0][0]
        )
        scored.append((score, entry))

    if not scored:
        return None

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_entry = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    # Reject if below absolute threshold
    if best_score < _BANK_MATCH_THRESHOLD:
        log.warning(
            "qno=%r → no bank match above threshold %.2f (best=%.3f) for question=%r",
            qno, _BANK_MATCH_THRESHOLD, best_score, question_text[:60],
        )
        return None

    # Reject if the margin over 2nd-best is too small (ambiguous match)
    margin = best_score - second_score
    if margin < _BANK_MARGIN:
        log.warning(
            "qno=%r → ambiguous bank match (best=%.3f, 2nd=%.3f, margin=%.3f < %.3f) "
            "for question=%r — rejecting to avoid wrong keywords.",
            qno, best_score, second_score, margin, _BANK_MARGIN, question_text[:60],
        )
        return None

    log.debug(
        "qno=%r → semantic bank match (score=%.3f, margin=%.3f): %r",
        qno, best_score, margin, best_entry["question"][:60],
    )
    return best_entry


def _evaluate_mapped_items(
    mapped: list[dict[str, str]],
    bank: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, float]:
    """
    Run evaluate_final() for every 'attempted' mapped item.

    For each record in ``mapped``:
      - status == 'unattempted' → marks = 0, feedback = "Not attempted."
      - status == 'invalid'     → skip entirely (not included in output)
      - status == 'attempted'   → look up bank entry, call evaluate_final()

    Args:
        mapped: Output of qa_mapper.map_questions_to_answers().
        bank:   Loaded question bank with model answers and keywords.

    Returns:
        (details, total_marks, max_total_marks)
    """
    from evaluator_final import evaluate_final  # lazy import — keeps startup fast

    details: list[dict[str, Any]] = []
    total_marks:     float = 0.0
    max_total_marks: float = 0.0

    for item in mapped:
        qno            = item.get("qno", "")
        question_text  = item.get("question", "")
        student_answer = item.get("answer", "")
        map_status     = item.get("status", "")

        # ── Skip invalid answers (question not on paper) ──────────────────
        if map_status == "invalid":
            log.debug("Skipping invalid qno=%r (no matching question on paper).", qno)
            continue

        # ── Unattempted: zero marks, no evaluation needed ─────────────────
        if map_status == "unattempted":
            bank_entry = _find_bank_entry(qno, question_text, bank)
            max_marks  = bank_entry["max_marks"] if bank_entry else 10
            max_total_marks += max_marks
            details.append({
                "qno":                   qno,
                "question":              question_text,
                "student_answer":        "",
                "marks":                 0,
                "max_marks":             max_marks,
                "status":                "unattempted",
                "feedback":              "Not attempted.",
                "matched_keywords":      [],
                "missing_keywords":      [],
                "confidence":            0.0,
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            })
            log.debug("qno=%r → unattempted (0 / %d marks).", qno, max_marks)
            continue

        # ── Attempted: find bank entry and evaluate ───────────────────────
        bank_entry = _find_bank_entry(qno, question_text, bank)

        if bank_entry is None:
            log.warning("qno=%r attempted but no bank entry found — skipping evaluation.", qno)
            details.append({
                "qno":                   qno,
                "question":              question_text,
                "student_answer":        student_answer,
                "marks":                 0,
                "max_marks":             10,
                "status":                "unevaluable",
                "feedback":              "No model answer available for this question.",
                "matched_keywords":      [],
                "missing_keywords":      [],
                "confidence":            0.0,
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            })
            continue

        model_answer = bank_entry["model_answer"]
        keywords     = bank_entry["keywords"]
        concepts     = bank_entry.get("concepts", [])   # v5.0
        max_marks    = bank_entry["max_marks"]

        result = evaluate_final(
            student_answer, model_answer, keywords,
            concepts=concepts, max_marks=max_marks,    # v5.0 additions
        )

        # v5.0: use rounded_score (0.5-step) already scaled to max_marks
        rounded_score: float = result.get("rounded_score", 0.0)
        scaled_marks: float  = min(rounded_score, float(max_marks))

        total_marks     += scaled_marks
        max_total_marks += max_marks

        # Normalise feedback: v5.0 returns list; join for backward-compat string field
        feedback_raw = result.get("feedback", [])
        feedback_str = "; ".join(feedback_raw) if isinstance(feedback_raw, list) else str(feedback_raw)

        details.append({
            "qno":                   qno,
            "question":              question_text,
            "student_answer":        student_answer,
            "marks":                 scaled_marks,
            "max_marks":             max_marks,
            "status":                "evaluated",
            "feedback":              feedback_str,
            "feedback_list":         feedback_raw,          # v5.0 — list form for rich UI
            "matched_keywords":      result.get("matched_keywords",  []),
            "missing_keywords":      result.get("missing_keywords",  []),
            "covered_concepts":      result.get("covered_concepts",  []),  # v5.0
            "missed_concepts":       result.get("missed_concepts",   []),  # v5.0
            "confidence":            round(result.get("confidence",  0.0), 4),
            "match_confidence":      round(result.get("confidence",  0.0), 4),
            "ocr_confidence":        0.0,
            "is_irrelevant":         result.get("is_irrelevant",         False),
            "is_partially_relevant": result.get("is_partially_relevant", False),
            # Detailed score breakdown (v5.0)
            "semantic_score":        result.get("semantic_score",      0.0),
            "keyword_score":         result.get("keyword_score",       0.0),
            "concept_score":         result.get("concept_score",       0.0),
            "contradiction_score":   result.get("contradiction_score", 0.0),
            "raw_score":             result.get("raw_score",           0.0),
            "length_factor":         result.get("length_factor",       1.0),
            "length_adjustment":     result.get("length_adjustment",   1.0),
            "score_out_of_10":       result.get("final_marks",         0.0),
            "final_score":           result.get("final_score",         0.0),
            "rounded_score":         scaled_marks,
        })
        log.debug(
            "qno=%r → evaluated: %.2f / %d marks (confidence=%.3f).",
            qno, scaled_marks, max_marks, result.get("confidence", 0.0),
        )

    return details, round(total_marks, 2), round(max_total_marks, 2)


def _build_new_response(
    details: list[dict[str, Any]],
    total_marks: float,
    max_total_marks: float,
    student_id: str,
) -> dict[str, Any]:
    """Build the final JSON response for the new extractor+mapper pipeline."""
    percentage = round(total_marks / max_total_marks * 100, 2) if max_total_marks > 0 else 0.0

    # Normalize field names so the frontend always receives consistent keys:
    #   qno   -> question_number  (frontend reads d.question_number)
    #   marks -> predicted_marks  (frontend reads d.predicted_marks)
    normalized_details: list[dict[str, Any]] = []
    for d in details:
        nd = dict(d)
        if "question_number" not in nd or not nd.get("question_number"):
            nd["question_number"] = nd.pop("qno", "") or ""
        else:
            nd.pop("qno", None)
        if "predicted_marks" not in nd:
            nd["predicted_marks"] = nd.pop("marks", 0.0)
        else:
            nd.pop("marks", None)
        normalized_details.append(nd)

    return {
        "status":     "success",
        "student_id": student_id,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_marks":     total_marks,
            "max_total_marks": max_total_marks,
            "percentage":      percentage,
        },
        "details": normalized_details,
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes — Static
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def serve_index():
    return send_from_directory(str(BASE_DIR / "frontend"), "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"})


# ════════════════════════════════════════════════════════════════════════════
# Route — COMBINED mode  (original, unchanged)
# ════════════════════════════════════════════════════════════════════════════

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Combined-mode evaluation.

    Form fields:
      file        (required) – single answer sheet image/PDF containing
                               both questions and answers.
      student_id  (optional)
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file field in request."}), 400

    f = request.files["file"]
    if not f.filename or not _allowed_file(f.filename):
        return jsonify({"status": "error", "message": "Invalid or missing file."}), 400

    student_id = (request.form.get("student_id", "") or "student").strip()
    ts         = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    try:
        save_path = _save_file(f, "sheet", student_id, ts)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not save file: {exc}"}), 500

    try:
        question_bank = _load_question_bank()
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Question bank error: {exc}"}), 500

    # ── OCR ───────────────────────────────────────────────────────────────────
    try:
        from ocr_engine import process_answer_sheet
        ocr_payload = process_answer_sheet(
            file_path=save_path, question_bank=question_bank,
            student_id=student_id, output_dir=OUTPUTS_DIR / "ocr_json", save_json=True,
        )
        log.info("OCR done: %d answer(s).", len(ocr_payload.get("answers", [])))
    except Exception as exc:
        log.error("OCR failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"status": "error", "message": f"OCR failed: {exc}"}), 500

    # ── Evaluate (same pipeline as /evaluate_separate) ────────────────────────
    try:
        mapped = [
            {
                "qno":      item.get("question_number", ""),
                "question": item.get("question", ""),
                "answer":   item.get("student_text", ""),
                "status":   "attempted" if item.get("student_text", "").strip() else "unattempted",
            }
            for item in ocr_payload.get("answers", [])
        ]
        details, total_marks, max_total_marks = _evaluate_mapped_items(mapped, question_bank)
        log.info("Eval done: %.2f / %.2f (%.1f%%)",
                 total_marks, max_total_marks,
                 (total_marks / max_total_marks * 100) if max_total_marks else 0)
    except Exception as exc:
        log.error("Evaluation failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"status": "error", "message": f"Evaluation failed: {exc}"}), 500

    # ── Build and persist response ────────────────────────────────────────────
    try:
        response    = _build_new_response(details, total_marks, max_total_marks, student_id)
        report_path = OUTPUTS_DIR / "reports" / f"report_{student_id}_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(response, fh, ensure_ascii=False, indent=2)
        log.info("Report saved → %s", report_path)
        return jsonify(response)
    except Exception as exc:
        log.error("Response build failed: %s", exc)
        return jsonify({"status": "error", "message": f"Response error: {exc}"}), 500


# ════════════════════════════════════════════════════════════════════════════
# Route — SEPARATE mode  (v3.1: _FileLike is now module-level)
# ════════════════════════════════════════════════════════════════════════════

@app.route("/evaluate_separate", methods=["POST"])
def evaluate_separate():
    """
    Separate-upload evaluation mode (v3.1).

    Form fields:
      question_paper  (required) – PDF of the question paper
      answer_sheet    (required) – PDF of the student answer sheet
      student_id      (optional)

    Pipeline:
      1. question_paper_extractor.extract_questions_from_upload()
      2. answer_sheet_extractor.extract_answers_from_upload()
      3. qa_mapper.map_questions_to_answers()
      4. _evaluate_mapped_items() → evaluate_final() per attempted question
    """
    # ── Validate uploads ──────────────────────────────────────────────────────
    if "question_paper" not in request.files:
        return jsonify({"status": "error",
                        "message": "Missing 'question_paper' file field."}), 400
    if "answer_sheet" not in request.files:
        return jsonify({"status": "error",
                        "message": "Missing 'answer_sheet' file field."}), 400

    qp_file = request.files["question_paper"]
    as_file = request.files["answer_sheet"]

    for fobj, label in [(qp_file, "question_paper"), (as_file, "answer_sheet")]:
        if not fobj.filename or not _allowed_file(fobj.filename):
            return jsonify({"status": "error",
                            "message": f"Invalid or missing {label} file."}), 400

    student_id = (request.form.get("student_id", "") or "student").strip()
    ts         = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ── Save both files ───────────────────────────────────────────────────────
    try:
        qp_path = _save_file(qp_file, "qpaper", student_id, ts)
        as_path = _save_file(as_file, "asheet", student_id, ts)
    except Exception as exc:
        return jsonify({"status": "error",
                        "message": f"Could not save uploaded files: {exc}"}), 500

    # ── Load question bank ────────────────────────────────────────────────────
    try:
        question_bank = _load_question_bank()
    except Exception as exc:
        return jsonify({"status": "error",
                        "message": f"Question bank error: {exc}"}), 500

    # ── Step 1: Extract questions from question paper ─────────────────────────
    try:
        from question_paper_extractor import extract_questions_from_upload
        log.info("Extracting questions from: %s", qp_path.name)
        questions = extract_questions_from_upload(
            _FileLike(qp_path),      # module-level class — safe copy guards included
            tmp_dir=str(UPLOADS_DIR),
        )
        log.info("Question extraction done: %d question(s).", len(questions))
    except Exception as exc:
        log.error("Question extraction failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"status": "error",
                        "message": f"Question extraction failed: {exc}"}), 500

    # ── Step 2: Extract answers from answer sheet ─────────────────────────────
    try:
        from answer_sheet_extractor import extract_answers_from_upload
        log.info("Extracting answers from: %s", as_path.name)
        expected_qnos = [q["qno"] for q in questions] if questions else None
        answers = extract_answers_from_upload(
            _FileLike(as_path),      # same module-level class
            expected_qnos=expected_qnos,
            tmp_dir=str(UPLOADS_DIR),
        )
        log.info("Answer extraction done: %d answer block(s).", len(answers))
    except Exception as exc:
        log.error("Answer extraction failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"status": "error",
                        "message": f"Answer extraction failed: {exc}"}), 500

    # ── Step 3: Map questions → answers ───────────────────────────────────────
    try:
        from qa_mapper import map_questions_to_answers
        log.info("Mapping questions to answers …")
        mapped = map_questions_to_answers(questions, answers)
        log.info(
            "Mapping done: %d attempted, %d unattempted, %d invalid.",
            sum(1 for m in mapped if m["status"] == "attempted"),
            sum(1 for m in mapped if m["status"] == "unattempted"),
            sum(1 for m in mapped if m["status"] == "invalid"),
        )
    except Exception as exc:
        log.error("Mapping failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"status": "error",
                        "message": f"Q→A mapping failed: {exc}"}), 500

    # ── Step 4: Evaluate each attempted answer ────────────────────────────────
    try:
        log.info("Running evaluation …")
        details, total_marks, max_total_marks = _evaluate_mapped_items(mapped, question_bank)
        log.info(
            "Evaluation done: %.2f / %.2f marks (%.1f%%).",
            total_marks, max_total_marks,
            (total_marks / max_total_marks * 100) if max_total_marks else 0,
        )
    except Exception as exc:
        log.error("Evaluation failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"status": "error",
                        "message": f"Evaluation failed: {exc}"}), 500

    # ── Build and persist response ─────────────────────────────────────────────
    try:
        response    = _build_new_response(details, total_marks, max_total_marks, student_id)
        report_path = OUTPUTS_DIR / "reports" / f"report_sep_{student_id}_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(response, fh, ensure_ascii=False, indent=2)
        log.info("Report saved → %s", report_path)
        return jsonify(response)
    except Exception as exc:
        log.error("Response build failed: %s", exc)
        return jsonify({"status": "error",
                        "message": f"Response error: {exc}"}), 500


# ════════════════════════════════════════════════════════════════════════════
# Demo endpoints  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

@app.route("/demo", methods=["GET"])
def demo():
    """Hard-coded demo for combined mode (no API key needed)."""
    return jsonify({
        "status":     "success",
        "student_id": "demo_student",
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_marks":     11.5,
            "max_total_marks": 17.0,
            "percentage":      67.6,
        },
        "details": [
            {
                "question_number":       "Q1",
                "question":              "திருக்குறள் பற்றி முழுமையான விளக்கம் தருக.",
                "student_answer":        "திருக்குறள் வள்ளுவர் இயற்றிய நூல். 1330 குறள்கள் உள்ளன. அறம் பொருள் இன்பம் என மூன்று பிரிவுகள் உள்ளன.",
                "matched_question":      "திருக்குறள் பற்றி முழுமையான விளக்கம் தருக.",
                "match_confidence":      0.92,
                "predicted_marks":       4.1,
                "max_marks":             5,
                "score_out_of_10":       8.2,
                "feedback":              "Good answer — covers the key concepts well.",
                "confidence":            0.82,
                "matched_keywords":      ["திருக்குறள்", "வள்ளுவர்", "1330", "அறம்"],
                "missing_keywords":      ["குறட்பா", "இன்பம்"],
                "status":                "evaluated",
                "ocr_confidence":        0.94,
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            },
            {
                "question_number":       "Q2",
                "question":              "தமிழ் சங்க காலம் பற்றி எழுதுக.",
                "student_answer":        "தமிழ் சங்க காலம் மூன்று சங்கங்களை கொண்டிருந்தது. தொல்காப்பியம் இக்காலத்தில் உருவானது.",
                "matched_question":      "தமிழ் சங்க இலக்கியம் பற்றி விவரி.",
                "match_confidence":      0.79,
                "predicted_marks":       3.8,
                "max_marks":             5,
                "score_out_of_10":       7.6,
                "feedback":              "Decent answer but could use more detail on literary works.",
                "confidence":            0.76,
                "matched_keywords":      ["சங்கம்", "தொல்காப்பியம்"],
                "missing_keywords":      ["அகம்", "புறம்", "புலவர்கள்"],
                "status":                "evaluated",
                "ocr_confidence":        0.91,
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            },
            {
                "question_number":       "Q3",
                "question":              "காந்தியடிகள் சத்தியாக்கிரகம் பற்றி விவரி.",
                "student_answer":        "",
                "matched_question":      "காந்தியடிகளின் சத்தியாக்கிரக இயக்கம்.",
                "match_confidence":      0.88,
                "predicted_marks":       0.0,
                "max_marks":             7,
                "score_out_of_10":       0.0,
                "feedback":              "No answer provided. Please attempt the question.",
                "confidence":            0.0,
                "matched_keywords":      [],
                "missing_keywords":      ["சத்தியாக்கிரகம்", "அஹிம்சை", "விடுதலை", "போராட்டம்"],
                "status":                "no_answer",
                "ocr_confidence":        0.0,
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            },
        ],
    })


@app.route("/demo_separate", methods=["GET"])
def demo_separate():
    """Hard-coded demo for separate-upload mode."""
    return jsonify({
        "status":     "success",
        "student_id": "demo_student",
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_marks":     13.2,
            "max_total_marks": 19.0,
            "percentage":      69.5,
        },
        "details": [
            {
                "qno":                   "1",
                "question":              "திருக்குறள் பற்றி முழுமையான விளக்கம் தருக.",
                "student_answer":        "திருக்குறள் வள்ளுவரால் எழுதப்பட்டது. இதில் 1330 குறள்கள் உள்ளன. அறம், பொருள், இன்பம் என மூன்று பிரிவுகளாக பிரிக்கப்பட்டுள்ளது.",
                "marks":                 4.5,
                "max_marks":             5,
                "score_out_of_10":       9.0,
                "feedback":              "Excellent answer — well structured and complete.",
                "confidence":            0.90,
                "matched_keywords":      ["1330", "அறம்", "பொருள்", "இன்பம்"],
                "missing_keywords":      [],
                "status":                "evaluated",
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            },
            {
                "qno":                   "2",
                "question":              "தமிழ் சங்க இலக்கியம் பற்றி விவரி.",
                "student_answer":        "சங்க இலக்கியம் முதல் சங்கம், இடைச்சங்கம், கடைச்சங்கம் என மூன்று சங்கங்களில் தோன்றியது. தொல்காப்பியம் முக்கியமான நூல்.",
                "marks":                 4.2,
                "max_marks":             5,
                "score_out_of_10":       8.4,
                "feedback":              "Good answer. Could mention akam/puram divisions.",
                "confidence":            0.84,
                "matched_keywords":      ["சங்கம்", "தொல்காப்பியம்"],
                "missing_keywords":      ["அகம்", "புறம்"],
                "status":                "evaluated",
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            },
            {
                "qno":                   "3",
                "question":              "காந்தியடிகளின் சத்தியாக்கிரக இயக்கம் பற்றி விவரி.",
                "student_answer":        "",
                "marks":                 0.0,
                "max_marks":             7,
                "score_out_of_10":       0.0,
                "feedback":              "Not attempted.",
                "confidence":            0.0,
                "matched_keywords":      [],
                "missing_keywords":      ["சத்தியாக்கிரகம்", "அஹிம்சை", "விடுதலை", "போராட்டம்", "உப்பு"],
                "status":                "unattempted",
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            },
            {
                "qno":                   "4",
                "question":              "தொல்காப்பியம் என்றால் என்ன?",
                "student_answer":        "தொல்காப்பியம் தமிழ் மொழியின் மிகப் பழமையான இலக்கண நூல். எழுத்து, சொல், பொருள் என மூன்று அதிகாரங்கள் உள்ளன.",
                "marks":                 4.5,
                "max_marks":             5,
                "score_out_of_10":       9.0,
                "feedback":              "Excellent answer.",
                "confidence":            0.91,
                "matched_keywords":      ["தொல்காப்பியம்", "இலக்கணம்", "எழுத்து", "சொல்", "பொருள்"],
                "missing_keywords":      ["தொல்காப்பியர்"],
                "status":                "evaluated",
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            },
        ],
    })


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Tamil Answer Sheet Evaluation System v3.2 — Flask Backend")
    log.info("Base      : %s", BASE_DIR)
    log.info("Bank      : %s", QUESTION_BANK_PATH)
    log.info("Uploads   : %s", UPLOADS_DIR)
    log.info("Outputs   : %s", OUTPUTS_DIR)
    log.info("Endpoints : /evaluate  /evaluate_separate  /demo  /demo_separate")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
