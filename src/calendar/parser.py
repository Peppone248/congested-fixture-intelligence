"""Turn raw match-log rows into a canonical fixture calendar.

The fixture calendar is the temporal spine of the whole pipeline: every
downstream module (congestion, fatigue, recovery, optimizer) joins on
``(team, match_date)``.
"""

from __future__ import annotations

import pandas as pd


def build_fixture_calendar(
    match_logs: pd.DataFrame,
    team_name: str,
) -> pd.DataFrame:
    """Normalise raw match logs into a per-team fixture calendar.

    Args:
        match_logs: Concatenated match logs from one or more competitions.
        team_name: Canonical team name used to label the output rows.

    Returns:
        Dataframe indexed by ``match_date`` (ascending, deduplicated) with
        columns ``team``, ``competition``, ``opponent``, ``venue``, ``kickoff_utc``.

    Raises:
        ValueError: If ``match_logs`` lacks the required date/competition columns.
    """
    # DESIGN: sorting + deduplicating here (rather than in each scraper) means
    # rescraping does not re-order the canonical calendar downstream models
    # cache against.
    raise NotImplementedError


def compute_rest_days(calendar: pd.DataFrame) -> pd.Series:
    """Compute days of rest between consecutive fixtures.

    Args:
        calendar: Fixture calendar as returned by :func:`build_fixture_calendar`.

    Returns:
        Series aligned to ``calendar.index`` giving the number of full days
        between each fixture and its immediate predecessor (``NaN`` for the
        first fixture of the season).

    Raises:
        ValueError: If ``calendar`` is not sorted by date.
    """
    raise NotImplementedError
