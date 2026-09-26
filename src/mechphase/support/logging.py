"""Logging setup shared by the command-line entry points.

Ref: Sec. 4.4 (reported wall-clock and hardware accounting).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: str = "INFO", stream: Any = None) -> None:
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("mechphase")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("mechphase") else f"mechphase.{name}")


