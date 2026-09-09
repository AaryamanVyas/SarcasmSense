"""Structured logging used by scripts and by the FastAPI backend.

Two formatters are provided:

* a human-readable console formatter for training scripts and local dev, and
* a JSON-lines formatter used in production (``LOG_FORMAT=json``) so that logs
  can be shipped to any aggregator without extra parsing.

The backend attaches a request id to every log record via a
:class:`contextvars.ContextVar`, which lets us correlate the retrieval /
inference / explanation stages of a single ``POST /predict`` call.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Mapping

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


def new_request_id() -> str:
    """Return a short, collision-resistant identifier for one API request."""
    return uuid.uuid4().hex[:12]


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        record.request_id = REQUEST_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "request_id":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact, aligned console output."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-38s | [%(request_id)s] %(message)s",
            datefmt="%H:%M:%S",
        )


def configure_logging(
    level: str | int | None = None,
    log_format: str | None = None,
    log_file: str | Path | None = None,
) -> None:
    """Install the root logging configuration.

    Args:
        level: Log level name or number. Defaults to ``$LOG_LEVEL`` or ``INFO``.
        log_format: ``"json"`` or ``"console"``. Defaults to ``$LOG_FORMAT`` or
            ``"console"``.
        log_file: Optional path; when given, logs are additionally appended
            there (used by the training scripts to persist run logs).
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_format = (log_format or os.getenv("LOG_FORMAT", "console")).lower()

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter: logging.Formatter = (
        JsonFormatter() if log_format == "json" else ConsoleFormatter()
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(_RequestIdFilter())
    root.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(_RequestIdFilter())
        root.addHandler(file_handler)

    # Third-party libraries are extremely chatty at INFO.
    for noisy in ("urllib3", "filelock", "sentence_transformers", "matplotlib", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log a structured event.

    Free-form user text is *never* passed here in production paths; callers pass
    lengths, hashes and metrics instead (see the "safe logging" requirement).
    """
    logger.info(event, extra={"event": event, **fields})


def redact(text: str, keep: int = 24) -> str:
    """Return a truncated preview of user text for debug-level logs only."""
    text = " ".join(text.split())
    return text if len(text) <= keep else f"{text[:keep]}…({len(text)} chars)"


def summarise_mapping(mapping: Mapping[str, Any]) -> str:
    """Render a mapping as ``k=v`` pairs for one-line progress logs."""
    return " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in mapping.items()
    )
