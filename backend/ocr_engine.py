"""
Tamil Answer Sheet OCR Engine  ·  v2.0
=======================================
Extracts Tamil handwritten / printed text from Images and PDFs.

NEW in v2.0
-----------
  • process_question_paper()     – OCR a QUESTION PAPER and return
                                   [{question_number, question}]
  • process_answer_sheet_only()  – OCR an ANSWER SHEET in answers-only mode
                                   (no question bank) → [{question_number, answer}]
  • build_answers_only_payload() – minimal payload for separate-upload mode

The original process_answer_sheet() (combined mode) is fully unchanged.

Dependencies
------------
  pip install google-cloud-vision pillow pdf2image opencv-python-headless

Environment variable:
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from google.cloud import vision
except ImportError:
    raise ImportError("Run: pip install google-cloud-vision")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import PIL.ImageOps
except ImportError:
    raise ImportError("Run: pip install pillow")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import numpy as np

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS: set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"
}
PDF_DPI:         int = 300
MIN_TEXT_LENGTH: int = 5
MIN_ANSWER_WORDS: int = 3
QUESTION_MARKERS: list[str] = ["கேள்வி", "வினா", "Q.", "Q:"]
ANSWER_MARKERS:   list[str] = ["விடை", "Answer", "Ans.", "A."]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Image Pre-processing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image: Image.Image) -> Image.Image:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < 1000:
        scale = 1000 / min(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return _preprocess_cv2(image) if CV2_AVAILABLE else _preprocess_pillow(image)


def _preprocess_pillow(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray.convert("RGB")


def _preprocess_cv2(image: Image.Image) -> Image.Image:
    img_np = np.array(image.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray   = clahe.apply(gray)
    gray   = cv2.fastNlMeansDenoising(gray, h=10)
    gray   = _deskew_cv2(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))


def _deskew_cv2(gray: np.ndarray) -> np.ndarray:
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return gray
        angles = [np.degrees(theta) - 90 for rho, theta in lines[:, 0]
                  if -45 < np.degrees(theta) - 90 < 45]
        if not angles:
            return gray
        angle = float(np.median(angles))
        if abs(angle) < 0.5:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as exc:
        log.warning("Deskew failed (%s) – using original.", exc)
        return gray


# ══════════════════════════════════════════════════════════════════════════════
# 2. PDF → Images
# ══════════════════════════════════════════════════════════════════════════════

def pdf_to_images(pdf_path: str | Path, dpi: int = PDF_DPI) -> list[Image.Image]:
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError("Run: pip install pdf2image")
    log.info("Converting PDF → images at %d DPI: %s", dpi, pdf_path)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    log.info("  → %d page(s).", len(pages))
    return pages


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = PDF_DPI) -> list[Image.Image]:
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError("Run: pip install pdf2image")
    return convert_from_bytes(pdf_bytes, dpi=dpi)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Google Vision OCR
# ══════════════════════════════════════════════════════════════════════════════

def _get_vision_client() -> vision.ImageAnnotatorClient:
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""):
        raise EnvironmentError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set.\n"
            "Set it to your service-account JSON key path."
        )
    return vision.ImageAnnotatorClient()


def extract_text_from_image(
    image: Image.Image,
    language_hints: list[str] | None = None,
) -> dict[str, Any]:
    client = _get_vision_client()
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    vision_image  = vision.Image(content=buf.getvalue())
    
    if language_hints:
        image_context = vision.ImageContext(language_hints=language_hints)
    else:
        image_context = None
    log.info("Sending image to Google Vision API …")
    response = client.document_text_detection(
        image=vision_image, image_context=image_context
    )
    if response.error.message:
        raise RuntimeError(f"Google Vision API error: {response.error.message}")

    annotation = response.full_text_annotation
    full_text  = annotation.text if annotation else ""
    blocks:      list[str]   = []
    confidences: list[float] = []

    for page in annotation.pages:
        for block in page.blocks:
            block_text = ""
            block_confs = []
            for para in block.paragraphs:
                for word in para.words:
                    word_text = "".join(s.text for s in word.symbols)
                    block_text += word_text + " "
                    if word.confidence:
                        block_confs.append(word.confidence)
            block_text = block_text.strip()
            if len(block_text) >= MIN_TEXT_LENGTH:
                blocks.append(block_text)
            confidences.extend(block_confs)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    log.info("  → %d words | %d blocks | conf %.2f%%",
             len(full_text.split()), len(blocks), avg_conf * 100)
    return {
        "full_text":  full_text,
        "blocks":     blocks,
        "confidence": round(avg_conf, 4),
        "word_count": len(full_text.split()) if full_text else 0,
    }
def ocr_page(image, lang_hints=None):
    """
    Wrapper function to match expected interface in other modules.
    """
    lang = lang_hints[0] if lang_hints else "ta"
    
    result = extract_text_from_image(image, language_hints=lang)
    
    return result.get("full_text", "")


def extract_text_from_file(
    file_path: str | Path,
    language_hints: list[str] | None = None,
) -> list[dict[str, Any]]:
    path = Path(file_path)
    pages = pdf_to_images(path) if path.suffix.lower() == ".pdf" else [Image.open(path)]
    results: list[dict[str, Any]] = []
    for num, img in enumerate(pages, 1):
        log.info("OCR page %d / %d …", num, len(pages))
        result = extract_text_from_image(preprocess_image(img), language_hints)
        result["page"] = num
        results.append(result)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. Shared text utilities
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_tamil(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _insert_question_newlines(full_text: str) -> str:
    """Insert newlines before mid-line question markers so the splitter works."""
    full_text = re.sub(
        r"(?<=[^\n])((?:Q\s*\d+\s*[\)\.\:\-])|(?:\b\d+\s*[\)\.\:])"
        r"|(?:கேள்வி\s*\d+)|(?:வினா\s*\d+))",
        r"\n\1", full_text, flags=re.IGNORECASE | re.UNICODE,
    )
    return re.sub(r"\n{2,}", "\n", full_text)


_Q_SPLIT = re.compile(
    r"(?:^|\n)(?:(?:Q|கேள்வி|வினா)[\s\.\:\-]*(\d+)|(\d+)\s*[\.\)\:])",
    re.IGNORECASE | re.MULTILINE,
)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Answer Sheet Parsing  (combined mode — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def parse_answer_sheet(
    ocr_pages: list[dict[str, Any]],
    question_markers: list[str] = QUESTION_MARKERS,
    answer_markers:   list[str] = ANSWER_MARKERS,
) -> list[dict[str, str]]:
    full_text = _insert_question_newlines(
        "\n".join(p["full_text"] for p in ocr_pages)
    )
    avg_conf = float(np.mean([p["confidence"] for p in ocr_pages]))
    matches  = list(_Q_SPLIT.finditer(full_text))
    qa_pairs: list[dict[str, str]] = []

    if not matches:
        log.warning("No numbered question blocks detected in answer sheet.")
        qa_pairs.append({
            "question_number": "1", "question_text": "",
            "answer_text": _normalise_tamil(full_text),
            "page": 1, "ocr_confidence": round(avg_conf, 4),
        })
        return qa_pairs

    for i, match in enumerate(matches):
        q_num = match.group(1) or match.group(2) or str(i + 1)
        start = match.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()

        q_text = ""; a_text = block
        ans_pat = re.compile(r"(விடை\s*:|vidai\s*:|answer\s*:|ans\s*:)", re.IGNORECASE)
        ans_m = ans_pat.search(block)

        if ans_m:
            q_text = _normalise_tamil(block[:ans_m.start()])
            a_text = _normalise_tamil(block[ans_m.end():])
        else:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if len(lines) >= 2:
                QWORDS = ["என்ன", "ஏன்", "எப்படி", "விளக்குக", "கூறுக", "எழுதுக"]
                q_lines, split_idx = [], None
                for idx, line in enumerate(lines):
                    q_lines.append(line)
                    if line.endswith(".") or any(w in line for w in QWORDS):
                        split_idx = idx; break
                if split_idx is not None:
                    a_text = _normalise_tamil(" ".join(lines[split_idx + 1:]))
                else:
                    q_lines = [lines[0]]
                    a_text  = _normalise_tamil(" ".join(lines[1:]))
                q_text = _normalise_tamil(" ".join(q_lines))
            else:
                sents = re.split(r'[\.?]', block)
                if len(sents) >= 2:
                    q_text = _normalise_tamil(sents[0])
                    a_text = _normalise_tamil(" ".join(sents[1:]))
                else:
                    a_text = _normalise_tamil(block)

        qa_pairs.append({
            "question_number": q_num.strip(),
            "question_text":   q_text,
            "answer_text":     a_text,
            "page":            1,
            "ocr_confidence":  round(avg_conf, 4),
        })

    log.info("Parsed %d Q&A pair(s) from answer sheet.", len(qa_pairs))
    return qa_pairs


# ══════════════════════════════════════════════════════════════════════════════
# 6. NEW — Question Paper Parsing (separate upload mode)
# ══════════════════════════════════════════════════════════════════════════════

def parse_question_paper(ocr_pages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Parse OCR text from a QUESTION PAPER into [{question_number, question}].

    Only extracts the question text per numbered block. No answer parsing.

    Args:
        ocr_pages: List of per-page OCR dicts.

    Returns:
        List of {"question_number": str, "question": str} dicts in order.
    """
    full_text = _insert_question_newlines(
        "\n".join(p["full_text"] for p in ocr_pages)
    )
    matches   = list(_Q_SPLIT.finditer(full_text))
    questions: list[dict[str, str]] = []

    if not matches:
        log.warning("No numbered questions detected in question paper.")
        questions.append({
            "question_number": "1",
            "question": _normalise_tamil(full_text),
        })
        return questions

    for i, match in enumerate(matches):
        q_num = match.group(1) or match.group(2) or str(i + 1)
        start = match.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()
        # Strip marks hints like "(5 marks)" or "[7]"
        block = re.sub(r"[\[\(]\s*\d+\s*(?:marks?)?\s*[\]\)]", "", block,
                       flags=re.IGNORECASE).strip()
        questions.append({
            "question_number": q_num.strip(),
            "question":        _normalise_tamil(block),
        })

    log.info("Parsed %d question(s) from question paper.", len(questions))
    return questions


def process_question_paper(
    file_path:     str | Path,
    language_hint: str = "ta",
) -> list[dict[str, str]]:
    """
    End-to-end pipeline: question paper file → [{question_number, question}].

    NEW in v2.0.

    Args:
        file_path:     Path to image or PDF of the question paper.
        language_hint: BCP-47 language hint.

    Returns:
        List of {"question_number": str, "question": str} dicts.
    """
    path = Path(file_path)
    log.info("=" * 60)
    log.info("Processing QUESTION PAPER: %s", path.name)
    log.info("=" * 60)
    ocr_pages = extract_text_from_file(path, language_hint=language_hint)
    return parse_question_paper(ocr_pages)


# ══════════════════════════════════════════════════════════════════════════════
# 7. NEW — Answers-only payload (separate upload mode)
# ══════════════════════════════════════════════════════════════════════════════

def build_answers_only_payload(
    qa_pairs:   list[dict[str, str]],
    student_id: str = "unknown",
    sheet_id:   str = "sheet_001",
) -> dict[str, Any]:
    """
    Build a minimal answers-only payload for separate-upload mode.

    Returns:
        {
          "student_id": ..., "sheet_id": ..., "timestamp": ...,
          "answers": [
            {"question_number": "1", "answer": "...", "ocr_confidence": 0.94},
            ...
          ]
        }
    """
    answers = [
        {
            "question_number": p["question_number"],
            "answer":          p.get("answer_text", ""),
            "ocr_confidence":  p.get("ocr_confidence", 0.0),
        }
        for p in qa_pairs
    ]
    return {
        "student_id": student_id,
        "sheet_id":   sheet_id,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "answers":    answers,
    }


def process_answer_sheet_only(
    file_path:     str | Path,
    student_id:    str = "unknown",
    language_hint: str = "ta",
) -> dict[str, Any]:
    """
    OCR an answer sheet without a question bank (separate-upload mode).

    NEW in v2.0. Returns an answers-only payload.
    """
    path = Path(file_path)
    log.info("=" * 60)
    log.info("Processing ANSWER SHEET (answers-only): %s | student=%s", path.name, student_id)
    log.info("=" * 60)
    ocr_pages = extract_text_from_file(path, language_hint=language_hint)
    qa_pairs  = parse_answer_sheet(ocr_pages)
    return build_answers_only_payload(qa_pairs, student_id=student_id, sheet_id=path.stem)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Combined mode helpers (original, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_answer(student_text: str) -> tuple[str, str]:
    if not student_text or not student_text.strip():
        return "not_attempted", "Not Attempted"
    if len(student_text.split()) < MIN_ANSWER_WORDS:
        return "empty", "No valid answer provided"
    return "valid", ""


def _build_ocr_index(qa_pairs: list[dict[str, str]]) -> dict[str, dict]:
    return {p["question_number"].strip(): p for p in qa_pairs}


def build_ocr_payload(
    qa_pairs:      list[dict[str, str]],
    question_bank: list[dict[str, Any]] | None = None,
    student_id:    str = "unknown",
    sheet_id:      str = "sheet_001",
) -> dict[str, Any]:
    answers:   list[dict[str, Any]] = []
    ocr_index: dict[str, dict]      = _build_ocr_index(qa_pairs)

    if question_bank:
        for idx, bank_entry in enumerate(question_bank):
            q_num    = str(idx + 1)
            ocr_pair = ocr_index.get(q_num)
            if ocr_pair:
                raw_text = ocr_pair.get("answer_text", "")
                page     = ocr_pair.get("page", 1)
                ocr_conf = ocr_pair.get("ocr_confidence", 0.0)
                q_text   = ocr_pair.get("question_text", "") or bank_entry.get("question", "")
            else:
                raw_text = ""; page = 0; ocr_conf = 0.0
                q_text   = bank_entry.get("question", "")
            status, feedback = _classify_answer(raw_text)
            answers.append({
                "question_number": q_num, "question": q_text,
                "student_text": raw_text, "model_answer": bank_entry.get("model_answer", ""),
                "keywords": bank_entry.get("keywords", []), "page": page,
                "ocr_confidence": ocr_conf, "status": status, "feedback": feedback,
            })
    else:
        for pair in qa_pairs:
            raw_text = pair.get("answer_text", "")
            status, feedback = _classify_answer(raw_text)
            answers.append({
                "question_number": pair["question_number"],
                "question": pair.get("question_text", ""),
                "student_text": raw_text, "model_answer": "", "keywords": [],
                "page": pair.get("page", 1),
                "ocr_confidence": pair.get("ocr_confidence", 0.0),
                "status": status, "feedback": feedback,
            })

    return {
        "student_id": student_id, "sheet_id": sheet_id,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "answers":    answers,
    }


def save_ocr_result(
    payload:    dict[str, Any],
    output_dir: str | Path = "ocr_output",
    filename:   str | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sid      = re.sub(r"[^\w\-]", "_", payload.get("student_id", "student"))
        filename = f"ocr_{sid}_{ts}.json"
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    log.info("OCR result saved → %s", out_path)
    return out_path


def process_answer_sheet(
    file_path:     str | Path,
    question_bank: list[dict[str, Any]] | None = None,
    student_id:    str = "unknown",
    sheet_id:      str | None = None,
    output_dir:    str | Path = "ocr_output",
    language_hint: str = "ta",
    save_json:     bool = True,
) -> dict[str, Any]:
    """End-to-end: answer sheet file → OCR payload (combined or answers-only)."""
    path     = Path(file_path)
    sheet_id = sheet_id or path.stem
    log.info("=" * 60)
    log.info("ANSWER SHEET: %s | mode: %s",
             path.name, "combined" if question_bank else "answers-only")
    log.info("=" * 60)
    ocr_pages = extract_text_from_file(path, language_hint=language_hint)
    qa_pairs  = parse_answer_sheet(ocr_pages)
    payload   = build_ocr_payload(qa_pairs, question_bank, student_id, sheet_id)
    if save_json:
        save_ocr_result(payload, output_dir=output_dir)
    return payload


def process_folder(
    folder_path:   str | Path,
    question_bank: list[dict[str, Any]] | None = None,
    output_dir:    str | Path = "ocr_output",
    language_hint: str = "ta",
) -> list[dict[str, Any]]:
    folder = Path(folder_path)
    files  = sorted(f for f in folder.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS | {".pdf"})
    if not files:
        log.warning("No supported files found in: %s", folder)
        return []
    results: list[dict[str, Any]] = []
    for fp in files:
        try:
            results.append(process_answer_sheet(
                fp, question_bank, student_id=fp.stem, output_dir=output_dir,
                language_hint=language_hint,
            ))
        except Exception as exc:
            log.error("Failed to process %s: %s", fp.name, exc)
    log.info("Batch: %d / %d processed.", len(results), len(files))
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <image_or_pdf> [student_id]")
        sys.exit(1)
    target  = Path(sys.argv[1])
    student = sys.argv[2] if len(sys.argv) > 2 else target.stem
    out = process_answer_sheet(file_path=target, student_id=student, output_dir="ocr_output")
    print(json.dumps(out, ensure_ascii=False, indent=2))
