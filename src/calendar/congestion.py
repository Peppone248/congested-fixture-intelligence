"""Classify each fixture into congestion tiers (NORMAL/MODERATE/HEAVY/EXTREME).

Tier assignment combines a rolling match-count window with per-fixture rest-day
overrides; both come from ``config/congestion_tiers.yaml`` so the definitions
stay auditable.
"""

from __future__ import annotations

import pandas as pd

# DESIGN: encoding tier levels as an ordered categorical avoids fragile
# string comparisons and lets pandas group / plot in the right order.
CONGESTION_LEVELS: tuple[str, ...] = ("NORMAL", "MODERATE", "HEAVY", "EXTREME")


def classify_congestion(
    calendar: pd.DataFrame,
    tier_config: dict,
) -> pd.Series:
    """Assign a congestion tier to every fixture in the calendar.

    Args:
        calendar: Fixture calendar with a datetime index or ``match_date`` column.
        tier_config: Parsed contents of ``config/congestion_tiers.yaml``.

    Returns:
        Categorical series (ordered) with one label per fixture drawn from
        :data:`CONGESTION_LEVELS`.

    Raises:
        KeyError: If ``tier_config`` lacks the required ``tiers`` section.
        ValueError: If ``calendar`` has no datetime information.
    """
    # DESIGN: compute rolling match count first, then let per-fixture rest-day
    # overrides *upgrade* the tier — never downgrade — so we never mask a
    # short-turnaround fixture with a long-window low count.
    raise NotImplementedError


def rolling_match_count(dates: pd.Series, window_days: int) -> pd.Series:
    """Rolling count of matches (including current) within ``window_days``.

    Args:
        dates: Ascending series of fixture dates.
        window_days: Lookback window size in days.

    Returns:
        Integer series aligned to ``dates`` giving the match count in the
        preceding window (inclusive of the current fixture).

    Raises:
        ValueError: If ``dates`` is not sorted ascending.
    """
    raise NotImplementedError
