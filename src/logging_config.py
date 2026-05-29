"""structlog configuration — JSON output with correlation ID context.

Use:
    from src.logging_config import configure_logging, bind_run_id
    configure_logging()           # call once at startup (idempotent)
    log = bind_run_id(run_id)
    log.info("pipeline_start")
"""

from __future__ import annotations

import logging
import sys

import structlog

# Guards against reconfiguring structlog on every request when configure_logging()
# was inadvertently left inside a hot path. After the first call this becomes a
# no-op so repeated calls in tests or retried code paths have no effect.
_configured: bool = False


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib logging + structlog to emit JSON to stderr.

    Idempotent: subsequent calls return immediately without reconfiguring.
    Intended to be called once from the application lifespan, not per-request.
    """
    global _configured
    if _configured:
        return
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
    _configured = True


def bind_run_id(run_id: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger with run_id bound in context."""
    structlog.contextvars.bind_contextvars(run_id=run_id)
    return structlog.get_logger()
