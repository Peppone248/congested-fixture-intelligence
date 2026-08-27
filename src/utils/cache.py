"""Filesystem cache for scraped HTML pages and expensive intermediate frames."""

from __future__ import annotations

import hashlib
from pathlib import Path


# DESIGN: cache keys are derived from the *canonical* request URL rather than
# from ad-hoc labels, so two call sites that fetch the same page cannot
# accidentally maintain divergent copies of the same content.
def cache_key(url: str) -> str:
    """Deterministic cache key for a URL.

    Args:
        url: Request URL, exactly as issued to the scraper.

    Returns:
        Hex digest suitable as a filename stem.

    Raises:
        ValueError: If ``url`` is empty.
    """
    raise NotImplementedError


def read_cached(cache_dir: Path, url: str) -> str | None:
    """Return cached response text for ``url`` if present.

    Args:
        cache_dir: Root directory holding cached pages.
        url: Request URL.

    Returns:
        Cached response body, or ``None`` if the key is not on disk.

    Raises:
        OSError: On unexpected filesystem failure.
    """
    raise NotImplementedError


def write_cached(cache_dir: Path, url: str, content: str) -> Path:
    """Persist a response body to disk under a URL-derived key.

    Args:
        cache_dir: Root directory holding cached pages.
        url: Request URL.
        content: Response body to persist.

    Returns:
        Path of the written cache file.

    Raises:
        OSError: If the cache directory is not writable.
    """
    raise NotImplementedError
