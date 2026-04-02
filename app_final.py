"""
app_final.py  ·  Multi-Subject Answer Sheet Evaluation System  ·  Flask Backend  v4.0
======================================================================================
Phase 2 — adds multi-subject routing with clean language / medium separation.

What changed in v4.0
--------------------
  NEW  subject_router.resolve_dataset_and_language()  — central routing logic
  NEW  /evaluate_separate  now accepts subject + medium form fields
  NEW  _load_question_bank_for_subject()  — subject-aware bank loader
  NEW  language hints passed dynamically to both extractors
  NEW  /subjects  endpoint — returns subject/medium metadata for the frontend
  KEPT evaluator_final.py, scoring logic, QA mapper — UNCHANGED

Subject Rules
-------------
  Language subjects  (tamil, english)        → no medium needed
  Content  subjects  (science, social, maths) → medium required (tamil | english)

Endpoints
---------
  POST /evaluate            – COMBINED mode (legacy – unchanged)
  POST /evaluate_separate   – SEPARATE mode: question paper PDF + answer sheet PDF
  GET  /subjects            – returns subject/medium options for the frontend
  GET  /demo                – Hard-coded demo for combined mode
  GET  /demo_separate       – Hard-coded demo for separate mode
  GET  /health              – Liveness check
  GET  /                    – Serve frontend/index.html

Run
---
  pip install flask flask-cors
  python app_final.py
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
from typing import Any, Optional
import re

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR    = BASE_DIR / "data"

# Legacy fallback for question_bank.json (Phase 1 compat)
QUESTION_BANK_PATH = (
    BASE_DIR / "question_bank.json"
    if (BASE_DIR / "question_bank.json").exists()
    else DATA_DIR / "question_bank.json"
)

for d in [UPLOADS_DIR, OUTPUTS_DIR,
          OUTPUTS_DIR / "ocr_json", OUTPUTS_DIR / "reports"]:
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tiff", ".tif", ".gif", ".pdf",
}

# Bank matching thresholds (unchanged from v3.2)
_BANK_MATCH_THRESHOLD: float = 0.75
_BANK_MARGIN:          float = 0.08

from logging_config import configure_logging, get_logger
configure_logging(log_dir=BASE_DIR / "logs", level=logging.INFO)
log = get_logger(__name__)

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "frontend"),
    static_url_path="",
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
CORS(app)


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — Subject Router import
# ════════════════════════════════════════════════════════════════════════════

from subject_router import (
    resolve_dataset_and_language,
    get_subject_meta,
    LANGUAGE_SUBJECTS,
    MEDIUM_SUBJECTS,
    ALL_SUBJECTS,
    VALID_MEDIUMS,
)


# ════════════════════════════════════════════════════════════════════════════
# Helpers — file utilities (unchanged from v3.2)
# ════════════════════════════════════════════════════════════════════════════

def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _save_file(file_storage, prefix: str, student_id: str, ts: str) -> Path:
    """Save a FileStorage object and return its path."""
    safe_id   = re.sub(r"[^a-zA-Z0-9_\-]", "_", student_id)
    suffix    = Path(file_storage.filename).suffix.lower()
    safe_name = f"{prefix}_{safe_id}_{ts}{suffix}"
    save_path = UPLOADS_DIR / safe_name
    file_storage.save(str(save_path))
    log.info("Saved %s → %s (%d bytes)", prefix, save_path, save_path.stat().st_size)
    return save_path


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — Subject-aware question bank loader
# ════════════════════════════════════════════════════════════════════════════

def _load_question_bank_for_subject(
    subject: str,
    medium:  Optional[str] = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Load the question bank for the given subject + medium combination.

    Resolution order:
      1. Resolve dataset path via subject_router.
      2. Try BASE_DIR / dataset_path  (e.g. ./data/tamil_tamil.json)
      3. Fall back to legacy QUESTION_BANK_PATH (Phase 1 compat).

    Returns
    -------
    (bank_entries, resolved_path_str)

    Raises
    ------
    FileNotFoundError  – if no bank file can be found anywhere.
    ValueError         – propagated from resolve_dataset_and_language().
    """
    dataset_rel, _lang_hints = resolve_dataset_and_language(subject, medium)
    dataset_abs = BASE_DIR / dataset_rel

    if dataset_abs.exists():
        bank_path = dataset_abs
    elif QUESTION_BANK_PATH.exists():
        log.warning(
            "Dataset '%s' not found — falling back to legacy question_bank.json. "
            "Create '%s' to enable subject-specific evaluation.",
            dataset_abs, dataset_abs,
        )
        bank_path = QUESTION_BANK_PATH
    else:
        raise FileNotFoundError(
            f"No question bank found for subject='{subject}', medium='{medium}'. "
            f"Expected: {dataset_abs}"
        )

    with open(bank_path, "r", encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)

    bank = [{
        "question":     e.get("question",     ""),
        "model_answer": e.get("model_answer", ""),
        "keywords":     e.get("keywords",     []),
        "concepts":     e.get("concepts",     []),
        "max_marks":    int(e.get("max_marks", 10)),
    } for e in raw]

    log.info("Question bank loaded: %d question(s) from %s", len(bank), bank_path)
    return bank, str(bank_path)


def _load_question_bank() -> list[dict[str, Any]]:
    """Legacy loader — used by /evaluate (combined mode). Unchanged."""
    if not QUESTION_BANK_PATH.exists():
        log.warning("question_bank.json not found at %s", QUESTION_BANK_PATH)
        return []
    with open(QUESTION_BANK_PATH, "r", encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)
    bank = [{
        "question":     e.get("question",     ""),
        "model_answer": e.get("model_answer", ""),
        "keywords":     e.get("keywords",     []),
        "concepts":     e.get("concepts",     []),
        "max_marks":    int(e.get("max_marks", 10)),
    } for e in raw]
    log.info("Question bank loaded: %d question(s).", len(bank))
    return bank


# ════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1 — _FileLike  (module-level, unchanged from v3.2)
# ════════════════════════════════════════════════════════════════════════════

class _FileLike:
    """
    Minimal shim that lets extract_*_from_upload() accept a file already
    saved to disk (werkzeug FileStorage interface: .filename + .save()).
    """

    def __init__(self, path: Path) -> None:
        self.filename = path.name
        self._path    = path

    def save(self, dest: str) -> None:
        if not self._path.exists():
            raise RuntimeError(f"Source file does not exist: {self._path}")
        if not self._path.is_file():
            raise RuntimeError(f"Source path is not a regular file: {self._path}")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(self._path), dest)
        except OSError as exc:
            raise RuntimeError(
                f"File copy failed ({self._path.name} → {dest}): {exc}"
            ) from exc


# ════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2 — semantic bank matching (unchanged from v3.2)
# ════════════════════════════════════════════════════════════════════════════

def _find_bank_entry(
    qno: str,
    question_text: str,
    bank: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not question_text or not bank:
        return None

    q_stripped = question_text.strip()
    import re as _re
    def _norm(s: str) -> str:
        return _re.sub(r"[.?!।\s]+$", "", s.strip().lower())

    q_norm = _norm(q_stripped)

    for entry in bank:
        if _norm(entry["question"]) == q_norm:
            log.debug("qno=%r → exact bank match.", qno)
            return entry

    try:
        from evaluator_final import get_model_embedding_cached, preprocess_text
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError as exc:
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

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_entry = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    if best_score < _BANK_MATCH_THRESHOLD:
        log.warning(
            "qno=%r → no bank match above threshold %.2f (best=%.3f) for question=%r",
            qno, _BANK_MATCH_THRESHOLD, best_score, question_text[:60],
        )
        return None

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


# ════════════════════════════════════════════════════════════════════════════
# Evaluation helpers (unchanged from v3.2)
# ════════════════════════════════════════════════════════════════════════════

def _evaluate_mapped_items(
    mapped: list[dict[str, str]],
    bank:   list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, float]:
    """
    Run evaluate_final() for every 'attempted' mapped item.
    (Logic unchanged from v3.2 — only evaluate_final.py is called here.)
    """
    from evaluator_final import evaluate_final

    details:         list[dict[str, Any]] = []
    total_marks:     float = 0.0
    max_total_marks: float = 0.0

    for item in mapped:
        qno            = item.get("qno", "")
        question_text  = item.get("question", "")
        student_answer = item.get("answer", "")
        map_status     = item.get("status", "")

        if map_status == "invalid":
            log.debug("Skipping invalid qno=%r.", qno)
            continue

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

        bank_entry = _find_bank_entry(qno, question_text, bank)

        if bank_entry is None:
            log.warning("qno=%r attempted but no bank entry found — skipping.", qno)
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
        concepts     = bank_entry.get("concepts", [])
        max_marks    = bank_entry["max_marks"]

        result = evaluate_final(
            student_answer, model_answer, keywords,
            concepts=concepts, max_marks=max_marks,
        )

        rounded_score: float = result.get("rounded_score", 0.0)
        scaled_marks:  float = min(rounded_score, float(max_marks))
        total_marks     += scaled_marks
        max_total_marks += max_marks

        feedback_raw = result.get("feedback", [])
        feedback_str = (
            "; ".join(feedback_raw)
            if isinstance(feedback_raw, list)
            else str(feedback_raw)
        )

        details.append({
            "qno":                   qno,
            "question":              question_text,
            "student_answer":        student_answer,
            "marks":                 scaled_marks,
            "max_marks":             max_marks,
            "status":                "evaluated",
            "feedback":              feedback_str,
            "feedback_list":         feedback_raw,
            "matched_keywords":      result.get("matched_keywords",  []),
            "missing_keywords":      result.get("missing_keywords",  []),
            "covered_concepts":      result.get("covered_concepts",  []),
            "missed_concepts":       result.get("missed_concepts",   []),
            "confidence":            round(result.get("confidence",  0.0), 4),
            "match_confidence":      round(result.get("confidence",  0.0), 4),
            "ocr_confidence":        0.0,
            "is_irrelevant":         result.get("is_irrelevant",         False),
            "is_partially_relevant": result.get("is_partially_relevant", False),
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
            "qno=%r → evaluated: %.2f / %d (conf=%.3f).",
            qno, scaled_marks, max_marks, result.get("confidence", 0.0),
        )

    return details, round(total_marks, 2), round(max_total_marks, 2)


def _build_new_response(
    details:         list[dict[str, Any]],
    total_marks:     float,
    max_total_marks: float,
    student_id:      str,
    subject_meta:    Optional[dict] = None,
) -> dict[str, Any]:
    """Build the final JSON response — now includes subject metadata."""
    percentage = (
        round(total_marks / max_total_marks * 100, 2)
        if max_total_marks > 0 else 0.0
    )

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

    response: dict[str, Any] = {
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

    # Attach subject context when available (Phase 2)
    if subject_meta:
        response["subject_context"] = subject_meta

    return response


# ── OCR payload → pipeline format (unchanged from v3.2) ──────────────────────

def _ocr_to_pipeline(ocr_payload: dict[str, Any]) -> list[dict[str, str]]:
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
# Phase 2 — Request validator
# ════════════════════════════════════════════════════════════════════════════

def _validate_subject_medium(
    subject: Optional[str],
    medium:  Optional[str],
) -> tuple[str, Optional[str], Optional[dict]]:
    """
    Validate and normalise the subject / medium fields from a request.

    Returns
    -------
    (subject_key, medium_key, error_response)
      error_response is None on success, or a dict to return as JSON 400 on failure.
    """
    if not subject:
        # Default to Tamil for backward compatibility
        subject = "tamil"

    subject_key = subject.strip().lower()
    medium_key  = medium.strip().lower() if medium else None

    if subject_key not in ALL_SUBJECTS:
        return subject_key, medium_key, {
            "status":  "error",
            "message": (
                f"Unknown subject '{subject}'. "
                f"Valid subjects: {sorted(ALL_SUBJECTS)}"
            ),
        }

    if subject_key in MEDIUM_SUBJECTS and not medium_key:
        return subject_key, medium_key, {
            "status":  "error",
            "message": (
                f"Subject '{subject}' requires a medium. "
                f"Please provide medium='tamil' or medium='english'."
            ),
        }

    if medium_key and medium_key not in VALID_MEDIUMS:
        return subject_key, medium_key, {
            "status":  "error",
            "message": (
                f"Invalid medium '{medium}'. "
                f"Valid mediums: {sorted(VALID_MEDIUMS)}"
            ),
        }

    return subject_key, medium_key, None


# ════════════════════════════════════════════════════════════════════════════
# Routes — Static
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def serve_index():
    return send_from_directory(str(BASE_DIR / "frontend"), "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version":   "4.0",
    })


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — /subjects  (new)
# ════════════════════════════════════════════════════════════════════════════

@app.route("/subjects", methods=["GET"])
def subjects():
    """
    Return subject configuration metadata for the frontend.

    Response
    --------
    {
      "language_subjects": ["english", "tamil"],
      "medium_subjects":   ["maths", "science", "social"],
      "valid_mediums":     ["english", "tamil"],
      "datasets": {
        "tamil_tamil":   { "exists": true,  "path": "data/tamil_tamil.json" },
        "english_english": { "exists": false, "path": "data/english_english.json" },
        ...
      }
    }
    """
    dataset_inventory: dict[str, dict] = {}

    for subj in sorted(ALL_SUBJECTS):
        if subj in LANGUAGE_SUBJECTS:
            mediums_to_check = [None]
        else:
            mediums_to_check = list(sorted(VALID_MEDIUMS))

        for med in mediums_to_check:
            try:
                rel_path, lang_hints = resolve_dataset_and_language(subj, med)
                abs_path = BASE_DIR / rel_path
                key = rel_path.replace("data/", "").replace(".json", "")
                dataset_inventory[key] = {
                    "path":           rel_path,
                    "exists":         abs_path.exists(),
                    "language_hints": lang_hints,
                }
            except ValueError:
                pass

    return jsonify({
        "language_subjects": sorted(LANGUAGE_SUBJECTS),
        "medium_subjects":   sorted(MEDIUM_SUBJECTS),
        "valid_mediums":     sorted(VALID_MEDIUMS),
        "datasets":          dataset_inventory,
    })


# ════════════════════════════════════════════════════════════════════════════
# Route — COMBINED mode  (legacy, unchanged — just uses default Tamil bank)
# ════════════════════════════════════════════════════════════════════════════

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Combined-mode evaluation (legacy).

    Form fields:
      file        (required) – single answer sheet image/PDF
      student_id  (optional)
      subject     (optional, default: "tamil")
      medium      (optional)
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file field in request."}), 400

    f = request.files["file"]
    if not f.filename or not _allowed_file(f.filename):
        return jsonify({"status": "error", "message": "Invalid or missing file."}), 400

    subject    = request.form.get("subject", "tamil")
    medium     = request.form.get("medium",  None) or None
    student_id = (request.form.get("student_id", "") or "student").strip()
    ts         = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Validate subject/medium
    subject_key, medium_key, err = _validate_subject_medium(subject, medium)
    if err:
        return jsonify(err), 400

    try:
        dataset_path, language_hints = resolve_dataset_and_language(subject_key, medium_key)
        subject_meta = get_subject_meta(subject_key, medium_key)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    log.info("Subject=%s, Medium=%s", subject_key, medium_key)
    log.info("Dataset=%s", dataset_path)
    log.info("Language hints=%s", language_hints)

    try:
        save_path = _save_file(f, "sheet", student_id, ts)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not save file: {exc}"}), 500

    try:
        question_bank, _ = _load_question_bank_for_subject(subject_key, medium_key)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"status": "error", "message": f"Question bank error: {exc}"}), 500

    try:
        from ocr_engine import process_answer_sheet
        ocr_payload = process_answer_sheet(
            file_path=save_path,
            question_bank=question_bank,
            student_id=student_id,
            output_dir=OUTPUTS_DIR / "ocr_json",
            save_json=True,
            language_hints=language_hints,     # ← Phase 2: dynamic hints
        )
        log.info("OCR done: %d answer(s).", len(ocr_payload.get("answers", [])))
    except Exception as exc:
        log.error("OCR failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"status": "error", "message": f"OCR failed: {exc}"}), 500

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

    try:
        response    = _build_new_response(details, total_marks, max_total_marks,
                                          student_id, subject_meta=subject_meta)
        report_path = OUTPUTS_DIR / "reports" / f"report_{student_id}_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(response, fh, ensure_ascii=False, indent=2)
        log.info("Report saved → %s", report_path)
        return jsonify(response)
    except Exception as exc:
        log.error("Response build failed: %s", exc)
        return jsonify({"status": "error", "message": f"Response error: {exc}"}), 500


# ════════════════════════════════════════════════════════════════════════════
# Route — SEPARATE mode  (v4.0: Phase 2 multi-subject)
# ════════════════════════════════════════════════════════════════════════════

@app.route("/evaluate_separate", methods=["POST"])
def evaluate_separate():
    """
    Separate-upload evaluation mode — Phase 2 (v4.0).

    Form fields:
      question_paper  (required) – PDF of the question paper
      answer_sheet    (required) – PDF of the student answer sheet
      student_id      (optional)
      subject         (optional, default: "tamil")
                        One of: tamil | english | science | social | maths
      medium          (optional, required for science/social/maths)
                        One of: tamil | english
                        Ignored  for tamil / english subjects.

    Subject routing
    ---------------
      Tamil   → data/tamil_tamil.json,   OCR hints: ["ta"]
      English → data/english_english.json, OCR hints: ["en"]
      Science + Tamil medium  → data/science_tamil.json,  OCR hints: ["ta"]
      Science + English medium → data/science_english.json, OCR hints: ["en"]
      (same pattern for social / maths)

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

    # ── Parse subject / medium ────────────────────────────────────────────────
    subject    = request.form.get("subject", "tamil") or "tamil"
    medium     = request.form.get("medium",  None)    or None
    student_id = (request.form.get("student_id", "") or "student").strip()
    ts         = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ── Validate subject / medium ─────────────────────────────────────────────
    subject_key, medium_key, err = _validate_subject_medium(subject, medium)
    if err:
        return jsonify(err), 400

    try:
        dataset_path, language_hints = resolve_dataset_and_language(subject_key, medium_key)
        subject_meta = get_subject_meta(subject_key, medium_key)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    # ── Phase 2 structured logging ────────────────────────────────────────────
    log.info("Subject=%s, Medium=%s", subject_key, medium_key)
    log.info("Dataset=%s", dataset_path)
    log.info("Language hints=%s", language_hints)

    # ── Save both files ───────────────────────────────────────────────────────
    try:
        qp_path = _save_file(qp_file, "qpaper", student_id, ts)
        as_path = _save_file(as_file, "asheet", student_id, ts)
    except Exception as exc:
        return jsonify({"status": "error",
                        "message": f"Could not save uploaded files: {exc}"}), 500

    # ── Load subject-specific question bank ───────────────────────────────────
    try:
        question_bank, bank_path_used = _load_question_bank_for_subject(
            subject_key, medium_key
        )
        log.info("Bank loaded: %d Q from %s", len(question_bank), bank_path_used)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"status": "error",
                        "message": f"Question bank error: {exc}"}), 500

    # ── Step 1: Extract questions (with language hints) ───────────────────────
    try:
        from question_paper_extractor import extract_questions_from_upload
        log.info("Extracting questions from: %s (hints=%s)", qp_path.name, language_hints)
        questions = extract_questions_from_upload(
            _FileLike(qp_path),
            tmp_dir=str(UPLOADS_DIR),
            language_hints=language_hints,    # ← Phase 2: dynamic OCR hints
        )
        log.info("Question extraction done: %d question(s).", len(questions))
    except TypeError:
        # Extractor does not yet accept language_hints — fall back gracefully
        log.warning("extract_questions_from_upload() does not accept language_hints yet; "
                    "calling without hints.")
        from question_paper_extractor import extract_questions_from_upload
        questions = extract_questions_from_upload(
            _FileLike(qp_path),
            tmp_dir=str(UPLOADS_DIR),
        )
        log.info("Question extraction done (no hints): %d question(s).", len(questions))
    except Exception as exc:
        log.error("Question extraction failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"status": "error",
                        "message": f"Question extraction failed: {exc}"}), 500

    # ── Step 2: Extract answers (with language hints) ─────────────────────────
    try:
        from answer_sheet_extractor import extract_answers_from_upload
        log.info("Extracting answers from: %s (hints=%s)", as_path.name, language_hints)
        expected_qnos = [q["qno"] for q in questions] if questions else None
        answers = extract_answers_from_upload(
            _FileLike(as_path),
            expected_qnos=expected_qnos,
            tmp_dir=str(UPLOADS_DIR),
            language_hints=language_hints,    # ← Phase 2: dynamic OCR hints
        )
        log.info("Answer extraction done: %d answer block(s).", len(answers))
    except TypeError:
        log.warning("extract_answers_from_upload() does not accept language_hints yet; "
                    "calling without hints.")
        from answer_sheet_extractor import extract_answers_from_upload
        expected_qnos = [q["qno"] for q in questions] if questions else None
        answers = extract_answers_from_upload(
            _FileLike(as_path),
            expected_qnos=expected_qnos,
            tmp_dir=str(UPLOADS_DIR),
        )
        log.info("Answer extraction done (no hints): %d answer block(s).", len(answers))
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
        response = _build_new_response(
            details, total_marks, max_total_marks,
            student_id, subject_meta=subject_meta,
        )
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
# Demo endpoints (unchanged from v3.2)
# ════════════════════════════════════════════════════════════════════════════

@app.route("/demo", methods=["GET"])
def demo():
    return jsonify({
        "status":     "success",
        "student_id": "demo_student",
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "subject_context": {
            "subject": "tamil", "medium": None,
            "is_language_sub": True, "eval_language": "tamil",
        },
        "summary": {
            "total_marks":     11.5,
            "max_total_marks": 17.0,
            "percentage":      67.6,
        },
        "details": [
            {
                "question_number":       "Q1",
                "question":              "திருக்குறள் பற்றி முழுமையான விளக்கம் தருக.",
                "student_answer":        "திருக்குறள் வள்ளுவர் இயற்றிய நூல். 1330 குறள்கள் உள்ளன.",
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
                "student_answer":        "தமிழ் சங்க காலம் மூன்று சங்கங்களை கொண்டிருந்தது.",
                "matched_question":      "தமிழ் சங்க இலக்கியம் பற்றி விவரி.",
                "match_confidence":      0.79,
                "predicted_marks":       3.8,
                "max_marks":             5,
                "score_out_of_10":       7.6,
                "feedback":              "Decent — could use more detail on literary works.",
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
                "feedback":              "No answer provided.",
                "confidence":            0.0,
                "matched_keywords":      [],
                "missing_keywords":      ["சத்தியாக்கிரகம்", "அஹிம்சை", "விடுதலை"],
                "status":                "no_answer",
                "ocr_confidence":        0.0,
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            },
        ],
    })


@app.route("/demo_separate", methods=["GET"])
def demo_separate():
    return jsonify({
        "status":     "success",
        "student_id": "demo_student",
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "subject_context": {
            "subject": "tamil", "medium": None,
            "is_language_sub": True, "eval_language": "tamil",
        },
        "summary": {
            "total_marks":     13.2,
            "max_total_marks": 19.0,
            "percentage":      69.5,
        },
        "details": [
            {
                "qno":                   "1",
                "question":              "திருக்குறள் பற்றி முழுமையான விளக்கம் தருக.",
                "student_answer":        "திருக்குறள் வள்ளுவரால் எழுதப்பட்டது. இதில் 1330 குறள்கள் உள்ளன.",
                "marks":                 4.5,
                "max_marks":             5,
                "score_out_of_10":       9.0,
                "feedback":              "Excellent — well structured.",
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
                "student_answer":        "சங்க இலக்கியம் மூன்று சங்கங்களில் தோன்றியது.",
                "marks":                 4.2,
                "max_marks":             5,
                "score_out_of_10":       8.4,
                "feedback":              "Good. Could mention akam/puram divisions.",
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
                "missing_keywords":      ["சத்தியாக்கிரகம்", "அஹிம்சை", "விடுதலை"],
                "status":                "unattempted",
                "is_irrelevant":         False,
                "is_partially_relevant": False,
            },
            {
                "qno":                   "4",
                "question":              "தொல்காப்பியம் என்றால் என்ன?",
                "student_answer":        "தொல்காப்பியம் தமிழின் மிகப் பழமையான இலக்கண நூல்.",
                "marks":                 4.5,
                "max_marks":             5,
                "score_out_of_10":       9.0,
                "feedback":              "Excellent answer.",
                "confidence":            0.91,
                "matched_keywords":      ["தொல்காப்பியம்", "இலக்கணம்", "எழுத்து"],
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
    log.info("=" * 68)
    log.info("Multi-Subject Answer Sheet Evaluation System v4.0 — Flask Backend")
    log.info("Base      : %s", BASE_DIR)
    log.info("Data dir  : %s", DATA_DIR)
    log.info("Uploads   : %s", UPLOADS_DIR)
    log.info("Outputs   : %s", OUTPUTS_DIR)
    log.info("Subjects  : %s", sorted(ALL_SUBJECTS))
    log.info("Endpoints : /evaluate  /evaluate_separate  /subjects  /demo  /demo_separate")
    log.info("=" * 68)
    app.run(host="0.0.0.0", port=5000, debug=True)
