"""Build per-(player, match) fatigue features.

Features are derived from the fixture calendar and per-match minutes, and are
the shared input to both the performance-decay model and the recovery estimator.
"""

from __future__ import annotations

import pandas as pd


# DESIGN: the feature vector is centralised so training, inference, and the
# Streamlit app all read the same schema — a mismatch here silently breaks the
# whole downstream chain, so it lives in one place.
FEATURE_COLUMNS: tuple[str, ...] = (
    "rest_days",
    "minutes_last_7d",
    "minutes_last_14d",
    "matches_last_21d",
    "travel_km_last_7d",
    "consecutive_starts",
    "age",
    "position",
)


def build_fatigue_features(
    player_minutes: pd.DataFrame,
    calendar: pd.DataFrame,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-(player, match) fatigue feature frame.

    Args:
        player_minutes: Long-format per-match minutes with position labels.
        calendar: Fixture calendar including rest_days and congestion tier.
        roster: Canonical roster carrying age and canonical position.

    Returns:
        Dataframe keyed on ``(player_id, match_date)`` with columns matching
        :data:`FEATURE_COLUMNS`.

    Raises:
        ValueError: If any required key column is missing from the inputs.
    """
    raise NotImplementedError


def rolling_minutes(minutes: pd.Series, window_days: int) -> pd.Series:
    """Rolling sum of minutes played over a lookback window.

    Args:
        minutes: Player minutes indexed by match date (ascending).
        window_days: Lookback window in days.

    Returns:
        Series of rolling minutes aligned to ``minutes.index``.

    Raises:
        ValueError: If ``minutes.index`` is not a monotonic DatetimeIndex.
    """
    raise NotImplementedError
