"""
qa_mapper.py  ·  Tamil Answer Sheet Evaluation System
======================================================
Maps extracted questions to extracted answers using question numbers.

This module is the bridge between:
  • question_paper_extractor  → List[{"qno": str, "question": str}]
  • answer_sheet_extractor    → List[{"qno": str, "answer": str}]

And produces:
  List[{
      "qno":      str,   # canonical question number
      "question": str,   # question text  (empty string if not in question paper)
      "answer":   str,   # student answer (empty string if not attempted)
      "status":   str,   # "attempted" | "unattempted" | "invalid"
  }]

Status semantics
----------------
  attempted    – qno found in both question paper and answer sheet
  unattempted  – qno in question paper but NO matching answer was given
  invalid      – qno in answer sheet but NO matching question exists
                 (e.g. the student wrote a qno that was not on the paper)

Design notes
------------
• All qnos are normalised to a canonical form before matching so that
  "Q1", "Q.1", "01", "(1)" all resolve to the same key "1".
• The question paper defines the PRIMARY ordering; invalid answers are
  appended at the end in the order they appear in the answer sheet.
• Pure dictionary lookup — O(1) per match regardless of paper length.
• No evaluation / scoring logic here; this module only does mapping.

Standalone usage
----------------
  python qa_mapper.py --questions questions.json --answers answers.json
  python qa_mapper.py --questions questions.json --answers answers.json --output mapped.json
  python qa_mapper.py --questions questions.json --answers answers.json --save
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Literal

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

QuestionRecord = dict[str, str]   # {"qno": ..., "question": ...}
AnswerRecord   = dict[str, str]   # {"qno": ..., "answer": ...}
MappedRecord   = dict[str, str]   # {"qno", "question", "answer", "status"}

Status = Literal["attempted", "unattempted", "invalid"]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  QNO Canonicalisation
# ─────────────────────────────────────────────────────────────────────────────
# Both extractors apply their own normalisation internally, but they use
# slightly different rules (the question extractor does not strip leading
# zeros or surrounding brackets).  This function provides a SINGLE,
# authoritative canonical form so that qnos from both sources always
# compare equal when they refer to the same question.
#
# Canonical rules (applied in order):
#   1. Unicode NFC — so Tamil combining chars compare correctly.
#   2. Strip surrounding whitespace.
#   3. Remove surrounding parentheses:  (1) → 1,  (i) → i.
#   4. Normalise Q-prefix:  Q.1 / Q-1 / Q 1 / q1  →  Q1.
#   5. Strip leading zeros from the numeric part:  01 → 1,  Q03 → Q3.
#   6. Strip trailing delimiters:  1. / 1) / 1:  →  1.
#   7. Lowercase everything EXCEPT the Q prefix, so "II" and "ii" match.
# ─────────────────────────────────────────────────────────────────────────────

def canonical_qno(raw: str) -> str:
    """
    Return the canonical form of a question-number string.

    This is the single source of truth for qno comparison across the whole
    evaluation system.  Both extractors should route through here before
    storing or comparing qnos.

    Args:
        raw: A qno string as emitted by either extractor, e.g. "Q.1", "01",
             "(iii)", "அ)", "1.a".

    Returns:
        Canonical string, e.g. "Q1", "1", "iii", "அ", "1.a".

    Examples:
        >>> canonical_qno("01")
        '1'
        >>> canonical_qno("(1)")
        '1'
        >>> canonical_qno("Q.3")
        'Q3'
        >>> canonical_qno("Q-2")
        'Q2'
        >>> canonical_qno("Q 5")
        'Q5'
        >>> canonical_qno("II")
        'ii'
        >>> canonical_qno("அ)")
        'அ'
        >>> canonical_qno("1.a")
        '1.a'
    """
    qno = unicodedata.normalize("NFC", raw).strip()

    # 1. Remove surrounding parentheses  (1) → 1,  (i) → i
    qno = re.sub(r"^\((.+)\)$", r"\1", qno)

    # 2. Normalise Q-prefix variants  Q.1 / Q-1 / Q 1 / q1  →  Q<digits>
    qno = re.sub(r"^[Qq][.\-\s]?\s*", "Q", qno)

    # 3. Strip trailing delimiters  1. / 1) / 1:  →  1  (but keep "1.a")
    qno = re.sub(r"[\s.):]+$", "", qno)

    # 4. Strip leading zeros on numeric or Q-numeric portion  01 → 1,  Q03 → Q3
    qno = re.sub(r"^(Q?)0+(\d)", r"\1\2", qno)

    # 5. Lowercase everything except the Q prefix
    #    so "III" == "iii" and "IV" == "iv"
    if qno.startswith("Q") and len(qno) > 1 and qno[1].isdigit():
        pass  # keep "Q" uppercase, digits are case-neutral
    else:
        qno = qno.lower()
        # Restore Tamil characters — lowercasing a Tamil string is a no-op
        # but we call it explicitly so the intent is clear

    return qno.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Index builders
# ─────────────────────────────────────────────────────────────────────────────

def _index_questions(
    questions: list[QuestionRecord],
) -> tuple[dict[str, str], list[str]]:
    """
    Build a canonical-qno → question-text lookup dict and an ordered key list.

    The ordered key list preserves the sequence in which questions appear on
    the paper (this defines the output order).

    Args:
        questions: Output of question_paper_extractor — list of
                   {"qno": ..., "question": ...} dicts.

    Returns:
        (lookup, order) where:
          lookup – {canonical_qno: question_text}
          order  – [canonical_qno, ...] in paper order (deduped)
    """
    lookup: dict[str, str] = {}
    order: list[str] = []

    for record in questions:
        key = canonical_qno(record.get("qno", "").strip())
        if not key:
            log.warning("Skipping question record with empty qno: %r", record)
            continue
        if key in lookup:
            log.debug("Duplicate question qno=%r — keeping first occurrence.", key)
            continue
        lookup[key] = record.get("question", "").strip()
        order.append(key)

    log.debug("Question index: %d unique qnos.", len(lookup))
    return lookup, order


def _index_answers(
    answers: list[AnswerRecord],
) -> dict[str, str]:
    """
    Build a canonical-qno → answer-text lookup dict.

    If the same qno appears more than once (OCR artefact), the answers are
    concatenated in encounter order.

    Args:
        answers: Output of answer_sheet_extractor — list of
                 {"qno": ..., "answer": ...} dicts.

    Returns:
        {canonical_qno: answer_text}
    """
    lookup: dict[str, list[str]] = {}

    for record in answers:
        key = canonical_qno(record.get("qno", "").strip())
        if not key:
            log.warning("Skipping answer record with empty qno: %r", record)
            continue
        text = record.get("answer", "").strip()
        if key not in lookup:
            lookup[key] = []
        if text:
            lookup[key].append(text)

    merged: dict[str, str] = {
        k: " ".join(parts) for k, parts in lookup.items()
    }
    log.debug("Answer index: %d unique qnos.", len(merged))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Core mapping logic
# ─────────────────────────────────────────────────────────────────────────────

def map_questions_to_answers(
    questions: list[QuestionRecord],
    answers: list[AnswerRecord],
) -> list[MappedRecord]:
    """
    Combine question and answer records into a unified mapping.

    Algorithm
    ---------
    Pass 1 — Iterate over questions (defines primary order):
      • For each question qno, look up answer dict.
      • If found → status = "attempted".
      • If not found → status = "unattempted", answer = "".

    Pass 2 — Iterate over answers:
      • For each answer qno, check if it was seen in the question paper.
      • If NOT seen → status = "invalid", question = "".
        These are appended after the paper-order records.

    Args:
        questions: List of {"qno": str, "question": str} from
                   question_paper_extractor.
        answers:   List of {"qno": str, "answer": str} from
                   answer_sheet_extractor.

    Returns:
        List of MappedRecord dicts in the order:
          [paper-order records] + [invalid/orphan answer records]
    """
    q_lookup, q_order = _index_questions(questions)
    a_lookup          = _index_answers(answers)

    mapped: list[MappedRecord] = []
    seen_qnos: set[str] = set()   # tracks qnos processed in pass 1

    # ── Pass 1: question paper order ─────────────────────────────────────────
    for qno in q_order:
        answer_text = a_lookup.get(qno, "")
        status: Status = "attempted" if answer_text else "unattempted"
        mapped.append({
            "qno":      qno,
            "question": q_lookup[qno],
            "answer":   answer_text,
            "status":   status,
        })
        seen_qnos.add(qno)
        log.debug("[%s] qno=%r", status.upper(), qno)

    # ── Pass 2: orphan answers (not in question paper) ────────────────────────
    # Preserve the relative order from the answer sheet.
    answer_order = [
        canonical_qno(r.get("qno", "")) for r in answers
    ]
    # Deduplicate while keeping first occurrence order
    seen_in_pass2: set[str] = set()
    for qno in answer_order:
        if not qno or qno in seen_qnos or qno in seen_in_pass2:
            continue
        seen_in_pass2.add(qno)
        mapped.append({
            "qno":      qno,
            "question": "",
            "answer":   a_lookup.get(qno, ""),
            "status":   "invalid",
        })
        log.debug("[INVALID] qno=%r — no matching question.", qno)

    # ── Summary log ──────────────────────────────────────────────────────────
    counts = {s: 0 for s in ("attempted", "unattempted", "invalid")}
    for r in mapped:
        counts[r["status"]] += 1
    log.info(
        "Mapping complete: %d attempted, %d unattempted, %d invalid.",
        counts["attempted"], counts["unattempted"], counts["invalid"],
    )

    return mapped


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

def map_from_files(
    questions_path: str | Path,
    answers_path: str | Path,
) -> list[MappedRecord]:
    """
    Load question and answer JSON files from disk and run the mapping.

    Args:
        questions_path: Path to questions JSON (question_paper_extractor output).
        answers_path:   Path to answers JSON (answer_sheet_extractor output).

    Returns:
        List of MappedRecord dicts.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If either file does not contain a JSON list.
    """
    def _load(path: Path, label: str) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"{label} file must contain a JSON array; got {type(data).__name__}.")
        return data

    q_path = Path(questions_path)
    a_path = Path(answers_path)
    questions = _load(q_path, "Questions")
    answers   = _load(a_path, "Answers")

    log.info("Loaded %d question(s) from %s", len(questions), q_path.name)
    log.info("Loaded %d answer(s)   from %s", len(answers),   a_path.name)

    return map_questions_to_answers(questions, answers)


def map_from_upload(
    questions_file_storage,   # werkzeug FileStorage
    answers_file_storage,     # werkzeug FileStorage
) -> list[MappedRecord]:
    """
    Flask-friendly wrapper: accepts werkzeug FileStorage objects, reads JSON
    content from them directly (no temp file needed), and returns the mapped list.

    Usage in app.py::

        from qa_mapper import map_from_upload

        @app.route("/map", methods=["POST"])
        def map_route():
            mapped = map_from_upload(
                request.files["questions"],
                request.files["answers"],
            )
            return jsonify({"status": "success", "mapped": mapped})

    Args:
        questions_file_storage: werkzeug FileStorage for questions JSON.
        answers_file_storage:   werkzeug FileStorage for answers JSON.

    Returns:
        List of MappedRecord dicts.
    """
    questions = json.load(questions_file_storage.stream)
    answers   = json.load(answers_file_storage.stream)

    if not isinstance(questions, list):
        raise ValueError("Questions upload must be a JSON array.")
    if not isinstance(answers, list):
        raise ValueError("Answers upload must be a JSON array.")

    return map_questions_to_answers(questions, answers)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map question paper questions to answer sheet answers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qa_mapper.py --questions questions.json --answers answers.json
  python qa_mapper.py -q questions.json -a answers.json --save
  python qa_mapper.py -q questions.json -a answers.json --output mapped.json
        """,
    )
    p.add_argument("--questions", "-q", required=True,
                   help="Path to questions JSON (question_paper_extractor output).")
    p.add_argument("--answers",   "-a", required=True,
                   help="Path to answers JSON (answer_sheet_extractor output).")
    p.add_argument("--output", "-o", default=None,
                   help="Output JSON path (default: <answers_stem>_mapped.json).")
    p.add_argument("--save", "-s", action="store_true",
                   help="Save mapped output to a JSON file.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging.")
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    mapped = map_from_files(args.questions, args.answers)

    # ── Print to stdout ───────────────────────────────────────────────────────
    print(json.dumps(mapped, ensure_ascii=False, indent=2))

    # ── Optionally save ───────────────────────────────────────────────────────
    if args.save or args.output:
        out_path = Path(args.output) if args.output else \
                   Path(args.answers).with_name(
                       Path(args.answers).stem + "_mapped.json"
                   )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(mapped, fh, ensure_ascii=False, indent=2)
        log.info("Mapped output saved → %s", out_path)

    # ── Stats to stderr ───────────────────────────────────────────────────────
    counts = {s: 0 for s in ("attempted", "unattempted", "invalid")}
    for r in mapped:
        counts[r["status"]] += 1
    print(
        f"\n[INFO] Total: {len(mapped)} record(s) — "
        f"{counts['attempted']} attempted, "
        f"{counts['unattempted']} unattempted, "
        f"{counts['invalid']} invalid.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
