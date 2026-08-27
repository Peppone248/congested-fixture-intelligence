"""FBref scraper: match logs, per-match minutes, and per-90 performance rates.

FBref is the primary source for per-match minutes played and per-90 rate stats
that feed the fatigue and performance-decay models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# DESIGN: FBref rate-limits aggressively; keep a hard floor between requests
# and expose it here so it can be tuned without touching call sites.
DEFAULT_REQUEST_INTERVAL_S: float = 3.0


def fetch_team_match_log(
    fbref_id: str,
    season: str,
    competition: str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch a team's match-by-match log for one season.

    Args:
        fbref_id: FBref team identifier (8-char hex slug from the team URL).
        season: Season label in FBref format, e.g. ``"2023-2024"``.
        competition: Optional competition filter (e.g. ``"Serie A"``).
        cache_dir: Directory holding cached HTML pages; when set, cached
            responses short-circuit the HTTP call.

    Returns:
        A dataframe with one row per match containing at minimum
        ``date``, ``competition``, ``opponent``, ``venue``, ``result``, ``xg``, ``xga``.

    Raises:
        NotImplementedError: Scraper not yet implemented.
    """
    raise NotImplementedError


def fetch_player_match_log(
    fbref_id: str,
    season: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch per-match minutes and rate stats for every player on a team.

    Args:
        fbref_id: FBref team identifier.
        season: Season label in FBref format.
        cache_dir: Directory holding cached HTML pages.

    Returns:
        Long-format dataframe keyed on ``(player_id, match_date)`` with
        columns for minutes, position, and per-90 output metrics.

    Raises:
        NotImplementedError: Scraper not yet implemented.
    """
    raise NotImplementedError


def parse_match_log_html(html: str) -> pd.DataFrame:
    """Parse an FBref match-log HTML fragment into a dataframe.

    Args:
        html: Raw HTML string as returned by FBref for a match-log page.

    Returns:
        Structured dataframe. Columns follow the FBref stat-table schema
        with lowercase snake_case names.

    Raises:
        ValueError: If the HTML does not contain the expected match-log table.
    """
    # DESIGN: FBref wraps many stat tables in HTML comments to defeat naive
    # scrapers; the real implementation strips the comment wrapper before
    # handing the fragment to lxml/BeautifulSoup.
    raise NotImplementedError


def load_metadata(_config: dict[str, Any]) -> dict[str, Any]:
    """Placeholder metadata loader used until the full scraper lands.

    Args:
        _config: Parsed contents of ``config/teams.yaml``.

    Returns:
        Empty dictionary. Real implementation will return per-team endpoint
        descriptors.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError
