from __future__ import annotations

import logging
import sys
from typing import Any


try:
    from loguru import logger
except Exception:  # pragma: no cover - only used when loguru is absent.
    logger = logging.getLogger("kg_rag")  # type: ignore[assignment]


def configure_logging(level: str = "INFO") -> Any:
    if hasattr(logger, "remove") and hasattr(logger, "add"):
        logger.remove()
        logger.add(sys.stderr, level=level, enqueue=False)
    else:
        logging.basicConfig(level=level)
    return logger
