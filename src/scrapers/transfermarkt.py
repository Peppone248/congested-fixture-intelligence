"""Transfermarkt scraper: squad valuations, injuries, and roster status.

Transfermarkt supplies the market-value signal for player importance and the
injury / suspension calendar the roster module intersects with match dates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# DESIGN: Transfermarkt blocks scripted UAs by default; the real client sets a
# browser-like User-Agent and honours a delay similar to FBref's floor.
DEFAULT_REQUEST_INTERVAL_S: float = 2.0


def fetch_squad_valuations(
    tm_id: int,
    season: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch the squad list with market values for one team-season.

    Args:
        tm_id: Transfermarkt club identifier (integer, from the club URL).
        season: Season label in Transfermarkt format, e.g. ``"2023"`` for 2023-24.
        cache_dir: Directory holding cached HTML pages.

    Returns:
        Dataframe with ``player_name``, ``position``, ``age``,
        ``market_value_eur``, and ``contract_until``.

    Raises:
        NotImplementedError: Scraper not yet implemented.
    """
    raise NotImplementedError


def fetch_injury_history(
    tm_id: int,
    season: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch the injury and suspension timeline for one team-season.

    Args:
        tm_id: Transfermarkt club identifier.
        season: Season label in Transfermarkt format.
        cache_dir: Directory holding cached HTML pages.

    Returns:
        Dataframe with ``player_name``, ``injury_type``, ``from_date``,
        ``until_date``, ``games_missed``.

    Raises:
        NotImplementedError: Scraper not yet implemented.
    """
    raise NotImplementedError


def parse_valuation_html(html: str) -> pd.DataFrame:
    """Parse a Transfermarkt squad-valuation HTML page into a dataframe.

    Args:
        html: Raw HTML string as returned by the Transfermarkt kader page.

    Returns:
        Structured dataframe with valuation rows.

    Raises:
        ValueError: If the HTML does not contain the expected roster table.
    """
    # DESIGN: Transfermarkt renders market values as strings like "€45.00m";
    # parsing normalises these to integer euros in a single conversion pass.
    raise NotImplementedError
