"""structlog configuration (NFR-06). Run logs are JSON lines under ``results/logs/``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, TextIO

import structlog

_handles: list[TextIO] = []


def configure_logging(level: str = "INFO", jsonl_path: Path | None = None) -> None:
    """Console logs to stderr; with ``jsonl_path`` every event is also appended as one JSON line.

    httpx and httpcore are held at WARNING: their INFO lines print full request URLs, and FRED's
    v1 API can only take its key as a query parameter.
    """
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    numeric = logging.getLevelNamesMapping()[level]
    while _handles:
        _handles.pop().close()

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    sinks: list[Any] = [structlog.dev.ConsoleRenderer(colors=False)]
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        handle = jsonl_path.open("a", encoding="utf-8", newline="\n")
        _handles.append(handle)
        renderer = structlog.processors.JSONRenderer(sort_keys=True)

        def tee(logger: Any, method: str, event: dict[str, Any]) -> dict[str, Any]:
            line = renderer(logger, method, dict(event))
            handle.write((line if isinstance(line, str) else str(line, "utf-8")) + "\n")
            handle.flush()
            return event

        processors.append(tee)
    structlog.configure(
        processors=[*processors, *sinks],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=_stderr_logger,
        cache_logger_on_first_use=False,
    )


def _stderr_logger(*_args: Any) -> structlog.PrintLogger:
    """Resolve ``sys.stderr`` per logger, so redirected streams are honoured."""
    return structlog.PrintLogger(file=sys.stderr)


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
