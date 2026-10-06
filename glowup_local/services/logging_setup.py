"""Rotating local diagnostic log; never records image pixels or photo metadata."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .settings import user_data_directory


def configure_logging(log_path: str | Path | None = None) -> Path:
    destination = (
        Path(log_path) if log_path else user_data_directory() / "logs" / "glowup.log"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")) == destination
        for handler in logger.handlers
    ):
        handler = RotatingFileHandler(
            destination, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return destination
