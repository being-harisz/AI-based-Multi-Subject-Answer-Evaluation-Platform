"""
answer_sheet_extractor.py  ·  Tamil Answer Sheet Evaluation System
==================================================================
Extracts question numbers and corresponding answers from a scanned
answer sheet PDF. The answer sheet contains NO question text —
only "question number + answer body".

Pipeline
--------
  Input: Answer sheet PDF (scanned, handwritten or printed)
       │
       ▼
  pdf_to_images()          ← pdf2image, 300 DPI
       │
       ▼
  preprocess_image()       ← deskew / denoise / CLAHE / Otsu binarise
       │
       ▼
  ocr_page()               ← Google Vision DOCUMENT_TEXT_DETECTION
       │
       ▼
  clean_lines()            ← normalise Unicode, collapse whitespace, drop noise
       │
       ▼
  group_answers()          ← detect qno tokens, accumulate multi-line answers
       │
       ▼
  post_process()           ← deduplicate, fill gaps, sort
       │
       ▼
  List[{"qno": "1", "answer": "திருக்குறள் ..."}]

Supported question-number formats
----------------------------------
  Numeric  : 1.  1)  1:  (1)  01.
  Q-prefix : Q1  Q.1  Q-1  Q 1
  Compound : 1.a  1.a)  1(a)  2.i
  Alpha    : a)  b.  (c)       ← only a–f to avoid false positives
  Roman    : i.  ii)  (iii)    ← up to viii
  Tamil    : அ)  ஆ)  இ)  ஈ)

Answer-sheet–specific design choices
--------------------------------------
• Answer lines are often much longer than question lines and may start
  with upper-case Tamil/English without any number.
• A line that does NOT start with a question-number token is ALWAYS
  appended to the current answer (more aggressive than the question
  extractor's continuation heuristic).
• Blank / very-short lines between paragraphs of the same answer are
  preserved as a single space so paragraph structure is not lost.
• Questions that were skipped by the student (no answer text) are
  recorded with answer = "" so downstream code can detect them.

Dependencies
------------
  pip install google-cloud-vision pillow pdf2image opencv-python-headless

Environment variable:
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

Standalone usage
----------------
  python answer_sheet_extractor.py answer_sheet.pdf
  python answer_sheet_extractor.py answer_sheet.pdf --output answers.json --save
  python answer_sheet_extractor.py answer_sheet.pdf --lang ta en --dpi 400 -v
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is not installed.\nRun: pip install pillow")

# ── shared OCR utilities (single source of truth) ─────────────────────────────
from ocr_engine import pdf_to_images, preprocess_image, ocr_page

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PDF_DPI: int = 300
VISION_LANG_HINTS: list[str] = ["ta", "en"]

# Minimum characters for an answer line to be kept (filters OCR garbage)
MIN_ANSWER_CHARS: int = 2

# Lines whose content matches any of these patterns are treated as noise
# (page headers, footers, instruction lines, etc.) and discarded.
_NOISE_PATTERNS: list[re.Pattern] = [
    # Headers / metadata
    re.compile(
        r"^\s*(page|பக்கம்|reg(?:ister)?\.?\s*no|roll|name|பெயர்|"
        r"date|தேதி|time|நேரம்|marks|மதிப்பெண்|maximum|total|"
        r"instructions|விதிமுறை|part\s+[a-z]|பகுதி|subject|பாடம்)\b",
        re.IGNORECASE | re.UNICODE,
    ),
    # Lines that are only punctuation / decorators
    re.compile(r"^\s*[*\-–—_=]{2,}\s*$"),
    # Standalone page numbers  e.g. "- 1 -"  or  "1"  alone
    re.compile(r"^\s*[-–]?\s*\d{1,3}\s*[-–]?\s*$"),
    # Date strings  12/04/2024  or  12-04-2024
    re.compile(r"^\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s*$"),
    # "Contd..." / "P.T.O." markers
    re.compile(r"^\s*(contd\.?|p\.?t\.?o\.?|over)\s*$", re.IGNORECASE),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1-3.  pdf_to_images / preprocess_image / ocr_page
#       → imported from ocr_engine (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

def ocr_pdf(
    pdf_path: str | Path,
    lang_hints: list[str] | None = None,
    dpi: int = PDF_DPI,
) -> list[str]:
    """
    OCR every page of a PDF.

    Args:
        pdf_path:   Path to the answer sheet PDF.
        lang_hints: BCP-47 language codes.
        dpi:        DPI for PDF rasterisation.

    Returns:
        List of raw text strings, one per page.
    """
    images = pdf_to_images(pdf_path, dpi=dpi)
    texts: list[str] = []
    for idx, img in enumerate(images, 1):
        log.info("OCR page %d / %d …", idx, len(images))
        texts.append(ocr_page(preprocess_image(img), lang_hints))
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Line Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean_line(raw: str) -> str:
    """
    Normalise a single raw OCR line.

    - Unicode NFC normalisation (critical for Tamil combining characters).
    - Collapse horizontal whitespace runs to a single space.
    - Strip leading / trailing whitespace.
    """
    line = unicodedata.normalize("NFC", raw)
    line = re.sub(r"[ \t\u00a0\u200b]+", " ", line)   # collapse all whitespace kinds
    return line.strip()


def _is_noise(line: str) -> bool:
    """Return True if the line should be discarded entirely."""
    return any(p.match(line) for p in _NOISE_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Question-Number Detection
# ─────────────────────────────────────────────────────────────────────────────
#
# The regex is intentionally identical in structure to question_paper_extractor
# so both modules share the same detection capability.  Any fixes here should
# be mirrored there and vice-versa.
# ─────────────────────────────────────────────────────────────────────────────

_QNO_RE = re.compile(
    r"""
    ^
    (?:
      # ── Numeric  (most common in answer sheets) ──────────────────────────
      # Matches: 1.  1)  1:  (1)  01)  Q1  Q.1  Q-1  Q 1
      (?P<arabic>
        (?:Q[.\-\s]?\s*)?             # optional Q prefix
        \d{1,3}                       # 1–3 digit number
        (?:\s*[.):\-]\s*)?            # optional trailing delimiter
        (?!\d)                        # not immediately followed by another digit
      )

      |

      # ── Compound sub-question  1.a  1.a)  1(a)  2.i ─────────────────────
      (?P<compound>
        \d{1,3}[.\s]\s*[a-zA-Z][).]?\s*
      )

      |

      # ── Stand-alone letter  a)  b.  (c)  — only a–f ──────────────────────
      (?P<alpha>
        \(?[a-fA-F][).]\s*
      )

      |

      # ── Roman numerals up to viii ─────────────────────────────────────────
      (?P<roman>
        \(?\b(?:i{1,3}|iv|vi{0,3}|viii)\b[).]\s*
      )

      |

      # ── Tamil letter sub-questions  அ)  ஆ)  இ)  ஈ)  உ)  ஊ) ─────────────
      (?P<tamil_letter>
        [அஆஇஈஉஊ][)]\s*
      )
    )
    """,
    re.VERBOSE | re.UNICODE,
)


def detect_qno(line: str) -> tuple[str | None, str]:
    """
    Attempt to detect a question-number token at the start of ``line``.

    Args:
        line: A single cleaned text line.

    Returns:
        ``(qno, remainder)`` where *qno* is the normalised question number
        (e.g. ``"1"``, ``"Q2"``, ``"1.a"``, ``"ii"``, ``"அ"``) and
        *remainder* is the rest of the line after the token.
        Returns ``(None, line)`` if no question number is found.
    """
    m = _QNO_RE.match(line)
    if m is None:
        return None, line

    token = m.group(0).strip()
    remainder = line[m.end():].strip()

    # Normalise: strip trailing punctuation, normalise Q prefix
    qno = re.sub(r"[\s.):]+$", "", token)
    qno = re.sub(r"^Q[.\-\s]?\s*", "Q", qno, flags=re.IGNORECASE)
    qno = qno.strip()

    return qno, remainder


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Answer Grouping  (core logic)
# ─────────────────────────────────────────────────────────────────────────────

def group_answers(lines: list[str]) -> list[dict[str, str]]:
    """
    Walk through cleaned OCR lines and group them into answer records.

    Algorithm
    ---------
    State machine with two states:
      • SEEKING  – no current question started yet; waiting for a qno line.
      • COLLECTING – inside an answer; accumulate lines until next qno.

    Transition rules:
      1. Any line whose first token is a question number → FLUSH current
         answer (if any), START new answer with that qno and any inline text.
      2. Otherwise → APPEND to current answer (even blank lines between
         paragraphs are kept as a single space to preserve flow).

    Answer-sheet–specific decision:
      Unlike the question extractor (which uses a "continuation" heuristic),
      here we ALWAYS append non-qno lines to the current answer.  Answer
      sheets have long, flowing paragraphs in Tamil that may start with
      uppercase letters or digits that should NOT be confused with a new
      question number.

    Args:
        lines: Cleaned text lines from all OCR pages (in order).

    Returns:
        List of ``{"qno": str, "answer": str}`` dicts in encounter order.
    """
    answers: list[dict[str, str]] = []
    current_qno: str | None = None
    current_parts: list[str] = []

    def _flush() -> None:
        """Commit the current answer to the output list."""
        if current_qno is None:
            return
        answer_text = " ".join(p for p in current_parts if p).strip()
        # Normalise internal whitespace one more time
        answer_text = re.sub(r"\s{2,}", " ", answer_text)
        answers.append({"qno": current_qno, "answer": answer_text})

    for raw_line in lines:
        line = _clean_line(raw_line)

        # ── Drop empty lines entirely (they carry no information)
        if not line:
            continue

        # ── Drop noise lines (headers, page numbers, etc.)
        if _is_noise(line):
            log.debug("Noise drop: %r", line)
            continue

        qno, remainder = detect_qno(line)

        if qno is not None:
            # ── New question number detected ──────────────────────────────
            _flush()
            current_qno = qno
            current_parts = [remainder] if remainder else []
            log.debug("New answer block: qno=%r  inline=%r", qno, remainder)

        else:
            # ── Continuation line ─────────────────────────────────────────
            if current_qno is not None:
                # Only keep lines with meaningful content
                if len(line) >= MIN_ANSWER_CHARS:
                    current_parts.append(line)
            else:
                # Lines before the very first question number are pre-amble
                # (student name, subject, etc.) — discard them.
                log.debug("Pre-amble drop: %r", line)

    _flush()
    return answers


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate(answers: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    If the same question number appears more than once (e.g., the student
    continued an answer on the next page and the page header repeated the
    question number), MERGE the answer texts rather than dropping one.

    Merge strategy: concatenate with a space in encounter order.
    """
    merged: dict[str, list[str]] = {}
    order: list[str] = []

    for entry in answers:
        key = entry["qno"]
        if key not in merged:
            merged[key] = []
            order.append(key)
        if entry["answer"]:
            merged[key].append(entry["answer"])

    return [
        {"qno": k, "answer": " ".join(merged[k]).strip()}
        for k in order
    ]


def _fill_gaps(
    answers: list[dict[str, str]],
    expected_qnos: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Insert placeholder entries for question numbers that were skipped.

    If ``expected_qnos`` is supplied (e.g. from the question paper extractor),
    every expected question that has no answer gets ``{"qno": ..., "answer": ""}``.

    If ``expected_qnos`` is None, the function fills numeric gaps in the
    detected sequence (e.g., if 1, 3 are found, inserts 2 with answer="").

    Args:
        answers:       De-duplicated answer list.
        expected_qnos: Optional list of qno strings from the question paper.

    Returns:
        Answer list with gaps filled (sorted by qno).
    """
    found_qnos = {a["qno"] for a in answers}

    if expected_qnos is not None:
        extra = [
            {"qno": q, "answer": ""}
            for q in expected_qnos
            if q not in found_qnos
        ]
        answers = answers + extra

    else:
        # Auto-fill numeric gaps only
        numeric_qnos = sorted(
            int(a["qno"]) for a in answers
            if re.fullmatch(r"\d+", a["qno"])
        )
        if numeric_qnos:
            full_range = range(numeric_qnos[0], numeric_qnos[-1] + 1)
            existing = set(numeric_qnos)
            for n in full_range:
                if n not in existing:
                    answers.append({"qno": str(n), "answer": ""})

    return answers


def _sort_answers(answers: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Sort answers by their question number using the same ordering scheme as
    question_paper_extractor: numeric → alpha → roman → Tamil letter.
    """
    roman_map = {
        "i": 1, "ii": 2, "iii": 3, "iv": 4,
        "v": 5, "vi": 6, "vii": 7, "viii": 8,
    }
    tamil_order = "அஆஇஈඋஊ"

    def _key(entry: dict[str, str]):
        qno = re.sub(r"^Q", "", entry["qno"], flags=re.IGNORECASE)

        # Pure integer
        if re.fullmatch(r"\d+", qno):
            return (0, int(qno), 0, "")

        # Compound  1.a
        m = re.fullmatch(r"(\d+)[.\s]([a-zA-Z])", qno)
        if m:
            return (0, int(m.group(1)), ord(m.group(2).lower()), "")

        # Stand-alone letter
        if re.fullmatch(r"[a-zA-Z]", qno):
            return (1, 0, ord(qno.lower()), "")

        # Roman
        if qno.lower() in roman_map:
            return (2, roman_map[qno.lower()], 0, "")

        # Tamil
        if qno in tamil_order:
            return (3, tamil_order.index(qno), 0, "")

        return (4, 0, 0, qno)

    return sorted(answers, key=_key)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main Extraction Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def extract_answers_from_text(
    raw_text: str,
    expected_qnos: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Parse a raw OCR text string into a structured answer list.

    This is the pure-text entry point — useful for unit testing or when
    you have already obtained OCR text by other means.

    Args:
        raw_text:      Concatenated OCR output from all pages.
        expected_qnos: Optional list of qno strings from the question paper
                       (used to fill missing / skipped questions).

    Returns:
        Sorted list of ``{"qno": str, "answer": str}`` dicts.
    """
    lines = raw_text.splitlines()
    answers = group_answers(lines)
    answers = _deduplicate(answers)
    answers = _fill_gaps(answers, expected_qnos)
    answers = _sort_answers(answers)

    log.info(
        "Answer extraction complete: %d answer block(s) "
        "(%d non-empty, %d empty/skipped).",
        len(answers),
        sum(1 for a in answers if a["answer"]),
        sum(1 for a in answers if not a["answer"]),
    )
    return answers


def extract_answers_from_pdf(
    pdf_path: str | Path,
    lang_hints: list[str] | None = None,
    dpi: int = PDF_DPI,
    expected_qnos: list[str] | None = None,
    save_json: bool = False,
    output_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """
    Full pipeline: Answer sheet PDF → structured answer list.

    Args:
        pdf_path:      Path to the scanned answer sheet PDF.
        lang_hints:    BCP-47 language codes (default: ["ta", "en"]).
        dpi:           DPI for PDF rasterisation (default: 300).
        expected_qnos: Optional qno list from question_paper_extractor to
                       detect and fill skipped questions.
        save_json:     Write output to a JSON file when True.
        output_path:   Path for the JSON file (auto-named when None).

    Returns:
        List of ``{"qno": str, "answer": str}`` dicts.

    Example::

        from answer_sheet_extractor import extract_answers_from_pdf

        answers = extract_answers_from_pdf(
            "student_01_answers.pdf",
            expected_qnos=["1", "2", "3", "4", "5"],
            save_json=True,
        )
        for a in answers:
            print(a["qno"], "→", a["answer"][:60])
    """
    pdf_path = Path(pdf_path)
    log.info("=" * 60)
    log.info("Answer Sheet Extractor")
    log.info("Input : %s", pdf_path.name)
    log.info("=" * 60)

    # Step 1 – OCR all pages
    page_texts = ocr_pdf(pdf_path, lang_hints=lang_hints, dpi=dpi)
    log.info("OCR complete: %d page(s) processed.", len(page_texts))

    # Step 2 – Merge page texts (page break = newline, preserves line order)
    full_text = "\n".join(page_texts)

    # Step 3 – Extract and structure
    answers = extract_answers_from_text(full_text, expected_qnos)

    if not answers:
        log.warning(
            "No answers detected.  Check:\n"
            "  • Answer sheet has question numbers in a supported format.\n"
            "  • Google Vision credentials are configured.\n"
            "  • Language hints match the sheet language (%s).",
            lang_hints or VISION_LANG_HINTS,
        )

    # Step 4 – Optionally persist
    if save_json:
        if output_path is None:
            output_path = pdf_path.with_name(pdf_path.stem + "_answers.json")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(answers, fh, ensure_ascii=False, indent=2)
        log.info("Answers saved → %s", output_path)

    return answers


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Flask Integration Helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_answers_from_upload(
    file_storage,                           # werkzeug FileStorage
    lang_hints: list[str] | None = None,
    expected_qnos: list[str] | None = None,
    tmp_dir: str | Path = "/tmp",
) -> list[dict[str, str]]:
    """
    Convenience wrapper for Flask routes.

    Saves the uploaded FileStorage to a temporary file, runs the extraction
    pipeline, cleans up the temp file, and returns the answer list.

    Usage in app.py::

        from answer_sheet_extractor import extract_answers_from_upload

        @app.route("/extract-answers", methods=["POST"])
        def extract_answers_route():
            answer_file = request.files["answer_sheet"]

            # Optionally pass qnos from a previously extracted question paper
            qnos = request.form.get("expected_qnos", "")
            expected = qnos.split(",") if qnos else None

            answers = extract_answers_from_upload(
                answer_file,
                expected_qnos=expected,
                tmp_dir=UPLOADS_DIR,
            )
            return jsonify({"status": "success", "answers": answers})

    Args:
        file_storage:  werkzeug FileStorage from ``request.files["…"]``.
        lang_hints:    BCP-47 language codes.
        expected_qnos: Optional qno list to fill skipped questions.
        tmp_dir:       Directory for the temporary PDF file.

    Returns:
        List of ``{"qno": str, "answer": str}`` dicts.
    """
    import tempfile as _tmp

    suffix = Path(file_storage.filename).suffix.lower() or ".pdf"
    with _tmp.NamedTemporaryFile(dir=str(tmp_dir), suffix=suffix, delete=False) as tf:
        file_storage.save(tf.name)
        tmp_path = Path(tf.name)

    try:
        return extract_answers_from_pdf(
            tmp_path,
            lang_hints=lang_hints,
            expected_qnos=expected_qnos,
        )
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Diagnostic Utility
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_ocr_text(raw_text: str, max_lines: int = 40) -> None:
    """
    Print a labelled view of how each OCR line is classified.

    Useful during development to tune the regex and noise patterns.

    Args:
        raw_text:  Raw OCR output (full text, may be multi-page).
        max_lines: Maximum number of lines to print (default 40).
    """
    lines = raw_text.splitlines()
    print(f"\n{'─'*70}")
    print(f"  OCR DIAGNOSTIC  ({len(lines)} total lines, showing first {max_lines})")
    print(f"{'─'*70}")
    for i, raw in enumerate(lines[:max_lines], 1):
        line = _clean_line(raw)
        if not line:
            label = "[ BLANK ]"
        elif _is_noise(line):
            label = "[ NOISE ]"
        else:
            qno, rem = detect_qno(line)
            if qno is not None:
                label = f"[ QNO={qno!r:<6} ] remainder={rem!r:.40s}"
            else:
                label = f"[ ANSWER ]  {line[:60]!r}"
        print(f"  {i:>3}.  {label}")
    print(f"{'─'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract answers from a scanned answer sheet PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python answer_sheet_extractor.py sheet.pdf
  python answer_sheet_extractor.py sheet.pdf --save --output answers.json
  python answer_sheet_extractor.py sheet.pdf --qnos 1 2 3 4 5 --lang ta en
  python answer_sheet_extractor.py sheet.pdf --diagnose
        """,
    )
    p.add_argument("pdf", help="Path to the scanned answer sheet PDF.")
    p.add_argument(
        "--output", "-o", default=None,
        help="Output JSON path (default: <pdf_stem>_answers.json).",
    )
    p.add_argument(
        "--save", "-s", action="store_true",
        help="Save extracted answers to a JSON file.",
    )
    p.add_argument(
        "--lang", "-l", nargs="+", default=["ta", "en"], metavar="LANG",
        help="BCP-47 language hints for Vision API (default: ta en).",
    )
    p.add_argument(
        "--dpi", type=int, default=PDF_DPI,
        help=f"DPI for PDF rasterisation (default: {PDF_DPI}).",
    )
    p.add_argument(
        "--qnos", nargs="+", default=None, metavar="QNO",
        help="Expected question numbers (from question paper extractor). "
             "Used to detect and fill skipped questions. E.g. --qnos 1 2 3 4",
    )
    p.add_argument(
        "--diagnose", action="store_true",
        help="Print a labelled line-by-line diagnostic of OCR output and exit.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.diagnose:
        log.info("Running in diagnostic mode — OCR only, no extraction.")
        page_texts = ocr_pdf(args.pdf, lang_hints=args.lang, dpi=args.dpi)
        diagnose_ocr_text("\n".join(page_texts))
        return

    answers = extract_answers_from_pdf(
        pdf_path=args.pdf,
        lang_hints=args.lang,
        dpi=args.dpi,
        expected_qnos=args.qnos,
        save_json=args.save or (args.output is not None),
        output_path=args.output,
    )

    print(json.dumps(answers, ensure_ascii=False, indent=2))
    print(
        f"\n[INFO] {len(answers)} answer block(s) extracted "
        f"({sum(1 for a in answers if a['answer'])} non-empty, "
        f"{sum(1 for a in answers if not a['answer'])} empty/skipped).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
