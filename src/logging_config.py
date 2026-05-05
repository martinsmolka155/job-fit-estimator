"""structlog configuration — JSON output with correlation ID context.

Use:
    from src.logging_config import configure_logging, bind_run_id
    configure_logging()
    log = bind_run_id(run_id)
    log.info("pipeline_start", file=str(path))
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib logging + structlog to emit JSON to stderr."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def bind_run_id(run_id: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger with run_id bound in context."""
    structlog.contextvars.bind_contextvars(run_id=run_id)
    return structlog.get_logger()
