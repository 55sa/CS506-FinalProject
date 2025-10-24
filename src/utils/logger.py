"""Centralized logging configuration for Boston 311 analysis."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return configured logger instance for the given module."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
