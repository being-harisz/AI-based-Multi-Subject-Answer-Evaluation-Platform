"""
question_paper_extractor.py  ·  Tamil Answer Sheet Evaluation System
=====================================================================
Extracts questions and their numbers from a scanned question paper PDF.

Pipeline
--------
  Input: Question paper PDF
       │
       ▼
  pdf_to_images()          ← convert each page to a PIL image (pdf2image, 300 DPI)
       │
       ▼
  preprocess_image()       ← deskew / denoise / enhance (reuses ocr_engine logic)
       │
       ▼
  ocr_page()               ← Google Vision DOCUMENT_TEXT_DETECTION
       │
       ▼
  extract_questions()      ← regex-based question-number detection + multi-line grouping
       │
       ▼
  List[{"qno": ..., "question": ...}]

Supported question-number formats (Tamil + English papers)
----------------------------------------------------------
  1.    1)    1:    Q1    Q.1    (1)    (a)    1.a    1.a)
  i.    ii.   iii.  iv.    (i)   (ii)
  அ)   ஆ)   இ)   ஈ)         ← Tamil-letter sub-questions

Dependencies
------------
  pip install google-cloud-vision pillow pdf2image opencv-python-headless

Environment variable required:
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

Usage (standalone)
------------------
  python question_paper_extractor.py question_paper.pdf
  python question_paper_extractor.py question_paper.pdf --output questions.json --lang ta
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

# ── third-party ───────────────────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is not installed.\nRun: pip install pillow")

try:
    import numpy as np
except ImportError:
    pass

# ── shared OCR utilities (single source of truth) ─────────────────────────────
from ocr_engine import pdf_to_images, preprocess_image, ocr_page

# ── logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PDF_DPI = 300          # higher DPI → better OCR for handwritten / small print
VISION_LANG_HINTS = ["ta", "en"]   # Tamil + English; adjust as needed

# Lines that match these patterns are treated as section headers / instructions,
# NOT as questions, even if they start with a number-like token.
SKIP_LINE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(page|பக்கம்|register|reg\.?\s*no|roll|name|பெயர்|date|தேதி|time|நேரம்|marks|மதிப்பெண்|maximum|instructions|விதிமுறை|part\s+[a-z]|பகுதி)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*\*+\s*$"),           # lines of asterisks
    re.compile(r"^\s*[-–—_]{3,}\s*$"),    # horizontal rules
    re.compile(r"^\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s*$"),  # dates like 12/04/2024
]

# ─────────────────────────────────────────────────────────────────────────────
# 1-3.  pdf_to_images / preprocess_image / ocr_page
#       → imported from ocr_engine (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from PIL import Image

from pathlib import Path
from PIL import Image

def ocr_pdf(
    pdf_path: str | Path,
    language_hints: list[str] | None = None,
    lang_hints: list[str] | None = None,
    dpi: int = 300,   # ✅ ADD THIS
) -> list[str]:

    path = Path(pdf_path)

    # Handle both parameter names
    if lang_hints is not None:
        language_hints = lang_hints

    # Handle file types
    if path.suffix.lower() == ".pdf":
        images = pdf_to_images(path, dpi=dpi)   # ✅ use dpi
    elif path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]:
        images = [Image.open(path)]
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    page_texts: list[str] = []

    for i, img in enumerate(images, start=1):
        log.info("OCR-ing page %d / %d …", i, len(images))

        processed = preprocess_image(img)
        text = ocr_page(processed, language_hints)

        page_texts.append(text)

    return page_texts


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean_line(line: str) -> str:
    """
    Normalise a single OCR output line.

    - Strip leading/trailing whitespace.
    - Collapse multiple internal whitespace characters into a single space.
    - Normalise Unicode to NFC (important for Tamil combining characters).
    """
    line = unicodedata.normalize("NFC", line)
    line = re.sub(r"[ \t]+", " ", line)   # collapse horizontal whitespace
    return line.strip()


def _is_skip_line(line: str) -> bool:
    """Return True if the line should be ignored (header, rule, date, etc.)."""
    return any(p.match(line) for p in SKIP_LINE_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Question-Number Detection
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Master regex: matches the question-number token at the START of a line.
#
# Groups captured (only one will match per line):
#   G1 – Arabic numeral variants  : 1.  1)  1:  (1)  Q1  Q.1  Q-1
#   G2 – Sub-question alpha       : 1.a  1.a)  1(a)
#   G3 – Stand-alone alpha        : a)  a.  (a)  b)  …
#   G4 – Roman numerals           : i.  ii)  (iii)  iv.
#   G5 – Tamil letters            : அ)  ஆ)  இ)  ஈ)  உ)
# ---------------------------------------------------------------------------

_QNO_RE = re.compile(
    r"""
    ^
    (?:
      # G1 – numeric  (Q1 / Q.1 / 1. / 1) / (1))
      (?P<arabic>
        (?:Q\.?\s*)?          # optional Q or Q.
        \d{1,3}               # 1-3 digit number
        (?:\s*[.):\-]\s*)?    # optional delimiter
        (?!\s*\d)             # not followed by another digit (avoids matching years)
      )
      |
      # G2 – compound sub-question  1.a  /  1.a)  /  1(a)
      (?P<compound>
        \d{1,3}[.\s]\s*[a-zA-Z][).]?
      )
      |
      # G3 – stand-alone letter  a)  a.  (a)  b)  – only a-f to avoid false positives
      (?P<alpha>
        \(?[a-fA-F][).]\s*
      )
      |
      # G4 – roman numerals up to viii
      (?P<roman>
        \(?\b(?:i{1,3}|iv|vi{0,3}|viii)\b[).]\s*
      )
      |
      # G5 – Tamil letter sub-questions  அ)  ஆ)  இ)  ஈ)  உ)  ஊ)
      (?P<tamil_letter>
        [அஆஇஈஉஊ][)]\s*
      )
    )
    """,
    re.VERBOSE | re.UNICODE,
)


def _detect_qno(line: str) -> tuple[str | None, str]:
    """
    Try to detect a question-number token at the beginning of ``line``.

    Returns:
        (qno, remainder) where qno is the normalised question number string
        (e.g. "1", "1.a", "Q1", "i", "அ") and remainder is the rest of the line.
        If no question number detected, returns (None, line).
    """
    m = _QNO_RE.match(line)
    if m is None:
        return None, line

    raw_token = m.group(0).strip()
    remainder = line[m.end():].strip()

    # Normalise the token into a clean qno string
    # Remove trailing punctuation but keep sub-question structure
    qno = re.sub(r"[\s.):]+$", "", raw_token)   # strip trailing . ) : space
    qno = re.sub(r"^Q\.?\s*", "Q", qno, flags=re.IGNORECASE)  # normalise Q prefix
    qno = qno.strip()

    return qno, remainder


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Multi-line Question Grouping
# ─────────────────────────────────────────────────────────────────────────────

def _is_continuation(line: str) -> bool:
    """
    Heuristic: a line is a *continuation* of the previous question if it:
      - starts with a lowercase letter or Tamil character (not a numeral / Q)
      - OR starts with a conjunction / article typical in question continuations
      - AND does NOT match a question-number pattern
    """
    if not line:
        return False
    # If the line begins a new question, it's definitely not a continuation
    qno, _ = _detect_qno(line)
    if qno is not None:
        return False
    # Lines starting with lowercase, Tamil text, or common continuation cues
    first_char = line[0]
    return (
        first_char.islower()
        or "\u0B80" <= first_char <= "\u0BFF"   # Tamil Unicode block
        or first_char in "("                     # e.g. (with examples)
    )


def group_questions(lines: list[str]) -> list[dict[str, str]]:
    """
    Walk through cleaned OCR lines and group them into question records.

    Algorithm:
      1. For each line, check if it starts with a question number.
      2. If yes → start a new question record.
      3. If no  → try to append to the current question (continuation check).
      4. After all lines are processed, flush the last pending question.

    Args:
        lines: Cleaned text lines from all pages of the question paper.

    Returns:
        List of dicts: [{"qno": "1", "question": "..."}, ...]
    """
    questions: list[dict[str, str]] = []
    current_qno: str | None = None
    current_parts: list[str] = []

    def _flush():
        if current_qno is not None and current_parts:
            q_text = " ".join(current_parts).strip()
            # Discard if the "question" is just noise (too short or only punctuation)
            if len(q_text) >= 5 and not re.fullmatch(r"[^\w\u0B80-\u0BFF]+", q_text):
                questions.append({"qno": current_qno, "question": q_text})

    for line in lines:
        line = _clean_line(line)
        if not line:
            continue
        if _is_skip_line(line):
            log.debug("Skipping header/instruction line: %r", line)
            continue

        qno, remainder = _detect_qno(line)

        if qno is not None:
            _flush()
            current_qno = qno
            current_parts = [remainder] if remainder else []
        else:
            if current_qno is not None:
                # Decide whether this line continues the current question
                if _is_continuation(line) or (current_parts and len(line) > 3):
                    current_parts.append(line)
                # else: orphan line between questions – silently drop
            # Lines before the very first question number are ignored

    _flush()
    return questions


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Post-processing / Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Remove duplicate qno entries that can arise from multi-page OCR.

    When the same question number appears twice (e.g. page footer repeated),
    the LONGER question text is kept.
    """
    seen: dict[str, dict[str, str]] = {}
    for q in questions:
        key = q["qno"]
        if key not in seen or len(q["question"]) > len(seen[key]["question"]):
            seen[key] = q
    return list(seen.values())


def _sort_questions(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Sort questions by their question number.

    Handles mixed formats: numeric first (1, 2, 3 …), then alpha (a, b …),
    then roman (i, ii …), then Tamil (அ, ஆ …).
    """
    def sort_key(q: dict[str, str]):
        qno = q["qno"]
        # Strip Q prefix
        qno_stripped = re.sub(r"^Q", "", qno, flags=re.IGNORECASE)

        # Pure integer
        m = re.fullmatch(r"(\d+)", qno_stripped)
        if m:
            return (0, int(m.group(1)), 0, "")

        # Compound  1.a  → (1, 0, ord('a'), '')
        m = re.fullmatch(r"(\d+)[.\s]([a-zA-Z])", qno_stripped)
        if m:
            return (0, int(m.group(1)), ord(m.group(2).lower()), "")

        # Stand-alone letter
        m = re.fullmatch(r"([a-zA-Z])", qno_stripped)
        if m:
            return (1, 0, ord(m.group(1).lower()), "")

        # Roman
        roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
                     "vi": 6, "vii": 7, "viii": 8}
        if qno_stripped.lower() in roman_map:
            return (2, roman_map[qno_stripped.lower()], 0, "")

        # Tamil letters
        tamil_order = "அஆஇஈඋஊ"
        if qno_stripped in tamil_order:
            return (3, tamil_order.index(qno_stripped), 0, "")

        # Fallback – sort lexicographically
        return (4, 0, 0, qno)

    return sorted(questions, key=sort_key)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main Extraction Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def extract_questions_from_text(raw_text: str) -> list[dict[str, str]]:
    """
    Given a single string of OCR text (one or multiple pages concatenated),
    return a structured list of questions.

    Args:
        raw_text: Raw OCR output text.

    Returns:
        List of {"qno": ..., "question": ...} dicts.
    """
    lines = raw_text.splitlines()
    questions = group_questions(lines)
    questions = _deduplicate(questions)
    questions = _sort_questions(questions)
    return questions


def extract_questions_from_pdf(
    pdf_path: str | Path,
    language_hints: list[str] | None = None,
    save_json: bool = False,
    output_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """
    Full pipeline: PDF → images → OCR → parsed question list.

    Args:
        pdf_path:       Path to the scanned question paper PDF.
        language_hints: BCP-47 language hint list (default: ["ta", "en"]).
        save_json:      If True, write output to a JSON file.
        output_path:    Path for the JSON file (auto-named if None).

    Returns:
        List of dicts: [{"qno": "1", "question": "…"}, …]

    Example::

        questions = extract_questions_from_pdf("question_paper.pdf")
        for q in questions:
            print(q["qno"], "→", q["question"])
    """
    pdf_path = Path(pdf_path)
    log.info("=" * 60)
    log.info("Question Paper Extractor")
    log.info("Input: %s", pdf_path.name)
    log.info("=" * 60)

    # Step 1: OCR all pages
    page_texts = ocr_pdf(pdf_path, language_hints=language_hints or VISION_LANG_HINTS)
    log.info("OCR complete: %d page(s).", len(page_texts))

    # Step 2: Merge all pages into a single text stream
    full_text = "\n".join(page_texts)
    log.debug("Total OCR characters: %d", len(full_text))

    # Step 3: Extract and structure questions
    questions = extract_questions_from_text(full_text)
    log.info("Extracted %d question(s).", len(questions))

    if not questions:
        log.warning(
            "No questions were detected. Check that:\n"
            "  • The PDF is a scanned question paper (not a text PDF).\n"
            "  • Google Vision credentials are configured correctly.\n"
            "  • The language hints are appropriate (%s).",
            language_hints or VISION_LANG_HINTS,
        )

    # Step 4: Optionally save to JSON
    if save_json:
        if output_path is None:
            output_path = pdf_path.with_name(pdf_path.stem + "_questions.json")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(questions, fh, ensure_ascii=False, indent=2)
        log.info("Questions saved → %s", output_path)

    return questions


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Flask integration helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_questions_from_upload(
    file_storage,               # werkzeug FileStorage object from Flask
    language_hints: list[str] | None = None,
    tmp_dir: str | Path = "/tmp",
) -> list[dict[str, str]]:
    """
    Convenience wrapper for use inside a Flask route.

    Saves the uploaded FileStorage to a temp file, runs the extraction
    pipeline, and returns the question list.

    Usage in app.py::

        from question_paper_extractor import extract_questions_from_upload

        @app.route("/extract-questions", methods=["POST"])
        def extract_questions_route():
            f = request.files["question_paper"]
            questions = extract_questions_from_upload(f)
            return jsonify({"questions": questions})

    Args:
        file_storage:   werkzeug FileStorage (from ``request.files["…"]``).
        language_hints: Optional BCP-47 language codes.
        tmp_dir:        Directory for the temporary PDF file.

    Returns:
        List of {"qno": ..., "question": ...} dicts.
    """
    import tempfile as _tmp
    suffix = Path(file_storage.filename).suffix.lower() or ".pdf"
    with _tmp.NamedTemporaryFile(
        dir=str(tmp_dir), suffix=suffix, delete=False
    ) as tf:
        file_storage.save(tf.name)
        tmp_path = Path(tf.name)

    try:
        return extract_questions_from_pdf(tmp_path, language_hints=language_hints)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 10.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract questions from a scanned question paper PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python question_paper_extractor.py paper.pdf
  python question_paper_extractor.py paper.pdf --output questions.json
  python question_paper_extractor.py paper.pdf --lang ta en --save
        """,
    )
    p.add_argument("pdf", help="Path to the scanned question paper PDF.")
    p.add_argument(
        "--output", "-o", default=None,
        help="Output JSON path (default: <pdf_stem>_questions.json).",
    )
    p.add_argument(
        "--lang", "-l", nargs="+", default=["ta", "en"],
        metavar="LANG",
        help="BCP-47 language hints (default: ta en).",
    )
    p.add_argument(
        "--save", "-s", action="store_true",
        help="Save extracted questions to a JSON file.",
    )
    p.add_argument(
        "--dpi", type=int, default=PDF_DPI,
        help=f"DPI for PDF-to-image conversion (default: {PDF_DPI}).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Override global DPI if user supplied a different value
    global PDF_DPI
    PDF_DPI = args.dpi

    questions = extract_questions_from_pdf(
        pdf_path=args.pdf,
        language_hints=args.lang,
        save_json=args.save or (args.output is not None),
        output_path=args.output,
    )

    print(json.dumps(questions, ensure_ascii=False, indent=2))
    print(f"\n[INFO] {len(questions)} question(s) extracted.", file=sys.stderr)


if __name__ == "__main__":
    main()
