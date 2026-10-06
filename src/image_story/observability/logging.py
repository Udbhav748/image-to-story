"""Logging helpers for the Image -> Story system.

Central configuration lives here so that library code never calls
`logging.basicConfig` itself and never writes to stdout.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_ROOT_NAME = "image_story"

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO, stream=None) -> None:
    """Configure the `image_story` logger tree once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `image_story` namespace, configuring on first use."""
    configure_logging()
    if name.startswith(_ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
