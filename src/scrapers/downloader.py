"""Cloudflare-safe page downloader, backed by soccerdata's headless browser.

WHY THIS EXISTS:

    FBref sits behind Cloudflare. A plain ``requests.get()`` — no matter how
    complete the headers — gets a 403, because Cloudflare requires a real
    browser that executes its JavaScript challenge. Our custom scraper's HTTP
    layer therefore cannot reach FBref directly.

    soccerdata solves exactly this: its FBref reader is a ``BaseSeleniumReader``
    that drives a headless browser (via seleniumbase), which clears Cloudflare.
    Crucially, soccerdata exposes a low-level ``.get(url, filepath)`` method that
    downloads ANY FBref URL and caches it to disk — not just the pages it knows
    how to parse.

    So we split responsibilities cleanly:
        * soccerdata  = DOWNLOAD engine (clears Cloudflare, caches HTML)
        * our fbref.py = PARSE engine (handles the all-competitions pages that
          soccerdata does not parse at match granularity)

    This module is the thin bridge: give it an FBref URL, it returns the HTML,
    fetched through soccerdata's browser and cached on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# DESIGN: soccerdata's FBref reader needs a league + season to construct, even
# though we only use its low-level .get(). We pass a harmless default (Serie A,
# a recent season); it does not constrain which URLs .get() can fetch.
_DEFAULT_LEAGUE = "ITA-Serie A"
_DEFAULT_SEASON = "2324"

# Module-level singleton so we build the (expensive) browser-backed reader once.
_READER = None


def _get_reader():
    """Construct (once) and return a soccerdata FBref reader.

    Returns:
        A soccerdata.FBref instance whose ``.get()`` clears Cloudflare.

    Raises:
        RuntimeError: If soccerdata isn't installed, with install guidance.
    """
    global _READER
    if _READER is not None:
        return _READER

    try:
        import soccerdata as sd
    except ImportError as exc:
        raise RuntimeError(
            "soccerdata is required to download FBref pages (it clears "
            "Cloudflare via a headless browser). Install it with:\n"
            "    pip install soccerdata\n"
            "Plain requests cannot reach FBref."
        ) from exc

    # DESIGN: no_cache=False so soccerdata keeps its own on-disk cache too.
    # We still pass our own filepath to .get() to control where HTML lands.
    _READER = sd.FBref(leagues=_DEFAULT_LEAGUE, seasons=_DEFAULT_SEASON)
    logger.info("Initialized soccerdata FBref reader (headless browser backend)")
    return _READER


def download_fbref_html(
    url: str,
    cache_path: Path | str,
    max_age_days: int = 365,
) -> str:
    """Download an FBref page via soccerdata's browser, return the HTML.

    Args:
        url: Full FBref URL (e.g. a team's all_comps match-log page).
        cache_path: Where to cache the downloaded HTML on disk. If the file
            already exists and is fresh, it's returned without re-downloading.
        max_age_days: Max cache age before re-download. Historical seasons
            never change, so a long default (365 days) is safe.

    Returns:
        The page HTML as a string.

    Raises:
        RuntimeError: If soccerdata is unavailable.
        Exception: Propagates soccerdata/browser errors (e.g. navigation
            timeouts) so the caller can decide how to handle them.

    Note:
        soccerdata's .get() returns a binary file-like object. We read it and
        decode to str. The downloaded bytes are cached at ``cache_path`` by
        soccerdata itself, so re-runs are fast and don't re-hit FBref.
    """
    cache_path = Path(cache_path)

    # DESIGN: short-circuit on our own cache first. If we already have the HTML,
    # we don't even need to spin up the browser reader.
    if cache_path.is_file():
        logger.debug("cache hit (local): %s -> %s", url, cache_path)
        return cache_path.read_text(encoding="utf-8", errors="replace")

    reader = _get_reader()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading via soccerdata browser: %s", url)
    # DESIGN: soccerdata.get() downloads through the headless browser (clearing
    # Cloudflare) and caches to `filepath`. It returns a binary file handle.
    handle = reader.get(url, filepath=cache_path, max_age=max_age_days)
    raw = handle.read()
    handle.close()

    html = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

    # DESIGN: ensure our cache_path holds the text (soccerdata may have written
    # bytes); normalize so our parser always reads utf-8 text from the same path.
    if not cache_path.is_file():
        cache_path.write_text(html, encoding="utf-8")

    return html