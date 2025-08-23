"""Logging helpers for Peakfit 3.x."""
from __future__ import annotations

import logging

from typing import Optional

_configured = False


def _ensure_configured(level: int = logging.INFO) -> None:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _configured = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger instance.

    The first call configures the root logger with a basic format. Subsequent
    calls simply return ``logging.getLogger(name)``. ``level`` can override the
    global logging level on the first call.
    """

    _ensure_configured(logging.INFO if level is None else level)
    return logging.getLogger(name)

