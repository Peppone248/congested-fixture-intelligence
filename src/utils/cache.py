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
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# DESIGN: module-level logger with the module's dotted name — this is what
# notebooks and Streamlit expect to configure, and it lets library users
# silence just the cache chatter without touching root logging.
logger = logging.getLogger(__name__)


# DESIGN: identify the scraper politely — a real UA plus a contact hint means
# site operators can reach out before rate-limiting or banning. Kept short so
# it does not look like an evasive spoof of a real browser.
DEFAULT_USER_AGENT: str = (
    "congested-fixture-intelligence/0.1 "
    "(+https://github.com/Peppone248/congested-fixture-intelligence)"
)

# DESIGN: bounded retries with exponential backoff — 3 attempts at 2s / 4s / 8s
# is enough to ride out a transient 429 or 5xx without turning a scrape into
# an unbounded stall.
MAX_RETRIES: int = 3
INITIAL_BACKOFF_S: float = 2.0

# DESIGN: only retry classes of failure that a retry can actually fix.
# 403 is excluded on purpose — a 403 typically means the site blocked the
# UA / IP, and retrying just harvests more 403s without ever succeeding.
RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


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


def _fetch_with_retries(
    url: str,
    user_agent: str,
    timeout_s: float,
) -> str:
    """Issue an HTTP GET with polite retries; return the response body.

    Args:
        url: Request URL.
        user_agent: Value for the ``User-Agent`` request header.
        timeout_s: Per-request timeout.

    Returns:
        Response body text.

    Raises:
        requests.HTTPError: For non-retryable status codes (e.g. 403, 404)
            or if all retries are exhausted on a retryable status.
        requests.RequestException: For transport-level failures after
            retries are exhausted.
    """
    # DESIGN: identity through the UA is the polite move; a Retry-After
    # header, when present, wins over the exponential schedule so we
    # respect the server's own guidance rather than second-guessing it.
    headers = {"User-Agent": user_agent}
    backoff = INITIAL_BACKOFF_S
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout_s)
        except requests.RequestException as exc:
            # DESIGN: transport-level errors (DNS, TCP, TLS) are treated as
            # retryable — they are usually transient in real scrapes.
            last_exc = exc
            logger.warning(
                "GET %s failed on attempt %d/%d: %s", url, attempt, MAX_RETRIES, exc,
            )
        else:
            status = response.status_code
            if status < 400:
                return response.text
            if status not in RETRYABLE_STATUS:
                # DESIGN: fail fast on 403/404 — retrying will not fix a
                # blocked UA or a missing page and only wastes the budget.
                logger.error("GET %s returned %d (non-retryable)", url, status)
                response.raise_for_status()
            # Retryable HTTP status: honour Retry-After if present.
            retry_after = response.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else backoff
            logger.warning(
                "GET %s returned %d on attempt %d/%d, backing off %.1fs",
                url, status, attempt, MAX_RETRIES, sleep_s,
            )
            last_exc = requests.HTTPError(f"{status} for {url}", response=response)

        if attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2

    # DESIGN: reraising the last observed error preserves the status /
    # transport context the caller needs to decide what to do next.
    assert last_exc is not None
    raise last_exc


def get_cached_or_fetch(
    url: str,
    cache_dir: Path | str,
    delay: float = 5.0,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout_s: float = 30.0,
) -> str:
    """Return the HTML for ``url``, using the on-disk cache when available.

    Args:
        url: Request URL.
        cache_dir: Root directory for this source's cache.
        delay: Seconds to sleep **after** a live fetch, before the function
            returns. Cache hits do not sleep — this is a courtesy delay for
            the *source*, not an artificial pause for the caller.
        user_agent: ``User-Agent`` header for live requests.
        timeout_s: Per-request timeout for live fetches.

    Returns:
        The HTML content (from cache or a fresh fetch).

    Raises:
        ValueError: If ``url`` is empty.
        requests.HTTPError: If the live fetch returns a non-retryable error
            or exhausts its retry budget.
        OSError: If the cache path is not writable.
    """
    path = cache_path_for_url(url, cache_dir)

    if path.is_file():
        # DESIGN: cache hits skip the delay entirely — notebooks re-run cells
        # often and paying the polite delay on a hit would slow interactive
        # work with no benefit to the source we are being polite towards.
        logger.debug("cache hit: %s -> %s", url, path)
        return path.read_text(encoding="utf-8")

    logger.info("cache miss: fetching %s", url)
    content = _fetch_with_retries(url, user_agent=user_agent, timeout_s=timeout_s)

    # DESIGN: create the fanout directory on first write so the cache tree
    # grows organically — no pre-seeding, no empty directories in git.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.debug("cached %s -> %s (%d bytes)", url, path, len(content))

    # DESIGN: sleep *after* the write so a Ctrl-C during the delay still
    # leaves the freshly-fetched page on disk, saving the next run a hit
    # against the source.
    if delay > 0:
        time.sleep(delay)

    return content


def clear_cache(cache_dir: Path | str, older_than_days: int | None = None) -> int:
    """Delete cached files under ``cache_dir``.

    Args:
        cache_dir: Root directory for the source's cache.
        older_than_days: If given, only delete files whose modification time
            is at least this many days in the past. ``None`` deletes every
            cached file under the directory.

    Returns:
        Number of files removed.

    Raises:
        FileNotFoundError: If ``cache_dir`` does not exist.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"cache directory not found: {root}")

    # DESIGN: threshold in absolute time, not per-file — this way all
    # eligible files agree on the same cutoff even if the walk takes a
    # while on a big cache.
    cutoff: datetime | None = None
    if older_than_days is not None:
        cutoff = datetime.now() - timedelta(days=older_than_days)

    removed = 0
    # DESIGN: match the exact filename pattern the cache writes so a stray
    # ``.gitkeep`` or a partially downloaded ``.tmp`` file is never touched.
    for path in root.rglob("*.html"):
        if not path.is_file():
            continue
        if cutoff is not None:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if mtime > cutoff:
                continue
        path.unlink()
        removed += 1

    logger.info(
        "cleared %d file(s) from %s (older_than_days=%s)",
        removed, root, older_than_days,
    )
    return removed
