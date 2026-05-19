"""Shared utilities: logging setup, input parsing, polite delays."""

import logging
import re
import time
from pathlib import Path


def parse_competition_input(value: str) -> str:
    """Extract a competition slug from a full URL or return the slug as-is."""
    value = value.strip().rstrip("/")
    match = re.search(r"kaggle\.com/competitions/([^/?#]+)", value)
    if match:
        return match.group(1)
    return value


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both console and a file inside output_dir."""
    log_file = output_dir / "collection_log.txt"

    logger = logging.getLogger("kaggle_collector")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on re-runs within the same process
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def polite_delay(seconds: float = 2.0) -> None:
    """Sleep politely between web requests to respect rate limits."""
    time.sleep(seconds)


def sanitize_filename(name: str) -> str:
    """Replace characters invalid in Windows/Linux filenames."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()


def bytes_to_human(size_bytes: int) -> str:
    """Convert a byte count to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
