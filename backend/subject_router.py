"""
subject_router.py  ·  Multi-Subject Dataset & Language Router  ·  v1.0
=======================================================================
Centralises all subject → (dataset_path, language_hints) resolution logic.

Rules
-----
  Language subjects  (Tamil, English)   → no medium required
  Content subjects   (Science, Social, Maths) → medium required (tamil | english)

Scaling path
------------
  data/
    tamil_tamil.json          ← exists (Phase 1)
    english_english.json      ← added (Phase 2)
    science_tamil.json        ← future
    science_english.json      ← future
    social_tamil.json         ← future
    social_english.json       ← future
    maths_tamil.json          ← future
    maths_english.json        ← future

Usage
-----
  from subject_router import resolve_dataset_and_language, LANGUAGE_SUBJECTS, MEDIUM_SUBJECTS

  dataset_path, lang_hints = resolve_dataset_and_language("tamil")
  dataset_path, lang_hints = resolve_dataset_and_language("science", medium="english")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# ── Subject classification ─────────────────────────────────────────────────────
#
# LANGUAGE_SUBJECTS: evaluated in their own language — medium is irrelevant.
# MEDIUM_SUBJECTS  : require an explicit medium to select the correct dataset.

LANGUAGE_SUBJECTS: frozenset[str] = frozenset({"tamil", "english"})
MEDIUM_SUBJECTS:   frozenset[str] = frozenset({"science", "social", "maths"})
ALL_SUBJECTS:      frozenset[str] = LANGUAGE_SUBJECTS | MEDIUM_SUBJECTS

VALID_MEDIUMS: frozenset[str] = frozenset({"tamil", "english"})

# Language hint tokens used by Google Vision / Tesseract OCR engines.
_LANG_HINTS: dict[str, list[str]] = {
    "tamil":   ["ta"],
    "english": ["en"],
}


# ── Public API ─────────────────────────────────────────────────────────────────

def resolve_dataset_and_language(
    subject: str,
    medium:  Optional[str] = None,
) -> tuple[str, list[str]]:
    """
    Resolve the dataset file path and OCR language hints for a given subject.

    Parameters
    ----------
    subject : str
        One of: tamil | english | science | social | maths  (case-insensitive)
    medium  : str | None
        Required for MEDIUM_SUBJECTS (science / social / maths).
        Ignored for LANGUAGE_SUBJECTS (tamil / english).
        One of: tamil | english  (case-insensitive)

    Returns
    -------
    dataset_path  : str   – relative path inside the project's data/ directory
    language_hints: list  – OCR language hint tokens, e.g. ["ta"] or ["en"]

    Raises
    ------
    ValueError  – unknown subject, invalid medium, or medium missing for content subject
    """
    subject_key = subject.strip().lower() if subject else ""
    medium_key  = medium.strip().lower()  if medium  else None

    # ── Validate subject ───────────────────────────────────────────────────────
    if subject_key not in ALL_SUBJECTS:
        raise ValueError(
            f"Unknown subject '{subject}'. "
            f"Valid subjects: {sorted(ALL_SUBJECTS)}"
        )

    # ── Language subjects (Tamil / English) ────────────────────────────────────
    if subject_key in LANGUAGE_SUBJECTS:
        # Medium is silently ignored — language IS the subject.
        dataset_path   = f"data/{subject_key}_{subject_key}.json"
        language_hints = _LANG_HINTS[subject_key]
        return dataset_path, language_hints

    # ── Medium-dependent subjects (Science / Social / Maths) ──────────────────
    if medium_key is None:
        raise ValueError(
            f"Subject '{subject}' requires a medium. "
            f"Please specify medium='tamil' or medium='english'."
        )

    if medium_key not in VALID_MEDIUMS:
        raise ValueError(
            f"Invalid medium '{medium}'. "
            f"Valid mediums: {sorted(VALID_MEDIUMS)}"
        )

    dataset_path   = f"data/{subject_key}_{medium_key}.json"
    language_hints = _LANG_HINTS[medium_key]
    return dataset_path, language_hints


def get_subject_meta(subject: str, medium: Optional[str] = None) -> dict:
    """
    Return a metadata dict describing the resolved subject configuration.
    Useful for logging and API responses.

    Returns
    -------
    {
        "subject":         "science",
        "medium":          "tamil",          # None for language subjects
        "is_language_sub": False,
        "dataset_path":    "data/science_tamil.json",
        "language_hints":  ["ta"],
        "eval_language":   "tamil"
    }
    """
    subject_key = subject.strip().lower() if subject else ""
    dataset_path, language_hints = resolve_dataset_and_language(subject, medium)

    medium_key     = medium.strip().lower() if medium else None
    is_lang_sub    = subject_key in LANGUAGE_SUBJECTS
    eval_language  = subject_key if is_lang_sub else (medium_key or "unknown")

    return {
        "subject":         subject_key,
        "medium":          None if is_lang_sub else medium_key,
        "is_language_sub": is_lang_sub,
        "dataset_path":    dataset_path,
        "language_hints":  language_hints,
        "eval_language":   eval_language,
    }
