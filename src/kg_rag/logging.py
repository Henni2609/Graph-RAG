from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator


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


@contextmanager
def log_timing(label: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"TIMING {label}: {time.perf_counter() - t0:.3f}s")
