"""
logger.py — Centralised logging for the RERA chatbot.
Uses loguru for clean, coloured console output + optional file logging.
"""

import sys
from loguru import logger
from config import settings


def setup_logger() -> None:
    """Configure loguru logger based on settings."""
    # Remove the default handler
    logger.remove()

    # Console handler — coloured, readable
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Optional file handler — plain text, rotated daily
    logger.add(
        "logs/chatbot.log",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        enqueue=True,       # thread-safe
        catch=True,         # don't crash on log errors
    )


# Run setup the moment this module is imported
setup_logger()

# Re-export logger so other modules just do: from logger import logger
__all__ = ["logger"]
