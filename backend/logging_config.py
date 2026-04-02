"""
logging_config.py  ·  Tamil Answer Sheet Evaluation System
===========================================================
Centralized logging configuration.

Usage (add to the TOP of every module, replacing any existing basicConfig):
    from logging_config import get_logger
    log = get_logger(__name__)

From app.py (call ONCE at startup, before importing any module):
    from logging_config import configure_logging
    configure_logging(log_dir="logs", level=logging.INFO)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_LOG_DIR   = Path("logs")
LOG_FORMAT        = "%(asctime)s  [%(levelname)-8s]  %(name)s  —  %(message)s"
DATE_FORMAT       = "%Y-%m-%d %H:%M:%S"
MAX_BYTES         = 10 * 1024 * 1024   # 10 MB per log file
BACKUP_COUNT      = 5                  # keep 5 rotated files

# One log file per concern — keeps grep simple and log viewers fast.
_LOG_FILES = {
    "ocr":        "ocr.log",
    "extraction": "extraction.log",
    "mapping":    "mapping.log",
    "evaluation": "evaluation.log",
    "errors":     "errors.log",
    "app":        "app.log",
}

_configured = False   # guard: configure_logging() is idempotent


# ── Public API ────────────────────────────────────────────────────────────────

def configure_logging(
    log_dir: str | Path = DEFAULT_LOG_DIR,
    level: int = logging.INFO,
    also_stdout: bool = True,
) -> None:
    """
    Configure the root logger with rotating file handlers and (optionally)
    a console handler.  Call this ONCE at application startup — in app.py's
    module body, before any other import that uses logging.

    Args:
        log_dir:     Directory where log files will be written.  Created if
                     it does not exist.
        level:       Minimum severity to capture (default: INFO).
        also_stdout: If True (default), also echo all messages ≥ level to
                     stdout.  Useful in development.  Set to False in
                     production or when running under a supervisor that
                     captures stdout separately.
    """
    global _configured
    if _configured:
        return
    _configured = True

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Per-concern rotating file handlers ────────────────────────────────────
    for _concern, filename in _LOG_FILES.items():
        handler = logging.handlers.RotatingFileHandler(
            log_path / filename,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # ── Dedicated ERROR file (captures WARNING+ERROR+CRITICAL from all loggers)
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / _LOG_FILES["errors"],
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    root.addHandler(error_handler)

    # ── Console handler ───────────────────────────────────────────────────────
    if also_stdout:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        root.addHandler(console)

    root.info("Logging initialised — log_dir=%s  level=%s", log_path, logging.getLevelName(level))


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  Always use this instead of logging.getLogger()
    directly so that module names are consistent and configure_logging()
    stays as the single configuration point.

    Example::

        from logging_config import get_logger
        log = get_logger(__name__)
        log.info("Module loaded.")

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)
