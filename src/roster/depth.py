"""Compute per-position depth charts from the canonical roster.

Depth is derived from realised minutes and adjusted for injury absence so a
short-term absentee is not misread as a low-depth position.
"""

from __future__ import annotations

import pandas as pd


def compute_depth_chart(
    roster: pd.DataFrame,
    player_minutes: pd.DataFrame,
) -> pd.DataFrame:
    """Rank every player at every canonical position.

    Args:
        roster: Canonical roster from :func:`roster.builder.build_roster`.
        player_minutes: Long-format per-match minutes played.

    Returns:
        Dataframe with columns ``position``, ``player_id``, ``rank``,
        ``minutes_share``. ``rank == 1`` marks the nominal starter at that
        position on total minutes to date.

    Raises:
        ValueError: If ``player_minutes`` lacks a match-date index.
    """
    # DESIGN: rank on cumulative minutes rather than on appearances so a
    # regular starter is not overtaken by a super-sub with more cameos.
    raise NotImplementedError


def depth_score(rank: int) -> float:
    """Convert a discrete depth rank into a continuous quality proxy.

    Args:
        rank: Positional depth rank (``1`` = starter, ``2`` = backup, ...).

    Returns:
        Score in ``(0, 1]`` decaying as rank increases.

    Raises:
        ValueError: If ``rank < 1``.
    """
    # DESIGN: exponential decay penalises drop-off past the second-choice
    # player, which matches how coaches actually feel the depth cliff.
    raise NotImplementedError
