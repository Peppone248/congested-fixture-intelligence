"""On-disk cache for scraped HTML pages.

Both scrapers (FBref, Transfermarkt) share this one caching layer so the
retry, rate-limit, and layout policies live in exactly one place. The layout
mirrors the pattern used in the injury-analytics project's Transfermarkt
scraper:

    <cache_dir>/<hash[:2]>/<hash>.html

The caller supplies ``cache_dir`` (typically ``data/raw/fbref/`` or
``data/raw/transfermarkt/``) so the same primitives serve any source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


# DESIGN: use md5 not sha256 — this hash is a filename derived from a URL,
# not a security primitive; the 128-bit space is more than sufficient to
# avoid collisions across a season's worth of pages, and md5 is shorter.
def cache_path_for_url(url: str, cache_dir: Path | str) -> Path:
    """Deterministic on-disk path for the cached HTML of a URL.

    Args:
        url: Request URL, exactly as it will be issued to the source.
        cache_dir: Root directory for this source's cache
            (e.g. ``data/raw/fbref/``).

    Returns:
        Path of the form ``<cache_dir>/<hash[:2]>/<hash>.html``. The path
        may or may not exist on disk — the function does not create it.

    Raises:
        ValueError: If ``url`` is empty.
    """
    if not url:
        raise ValueError("url must be a non-empty string")

    # DESIGN: hash the URL as UTF-8 bytes so equivalent strings on different
    # platforms (Linux/macOS/Windows notebooks) collide deterministically.
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()

    # DESIGN: 2-char hex fanout caps any single directory at ~256 siblings
    # even across thousands of pages — matters on ext4 where directory
    # scans slow down noticeably past a few thousand entries.
    return Path(cache_dir) / digest[:2] / f"{digest}.html"


def is_cached(url: str, cache_dir: Path | str) -> bool:
    """Return whether a cached copy of ``url`` already exists on disk.

    Args:
        url: Request URL.
        cache_dir: Root directory for this source's cache.

    Returns:
        ``True`` if the derived cache path exists as a regular file,
        ``False`` otherwise.

    Raises:
        ValueError: If ``url`` is empty.
    """
    # DESIGN: use ``is_file`` rather than ``exists`` so a bogus directory
    # sharing the target name does not fool the caller into skipping the
    # fetch and then failing on read.
    return cache_path_for_url(url, cache_dir).is_file()
