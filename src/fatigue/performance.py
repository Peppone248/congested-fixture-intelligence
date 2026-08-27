"""Model per-match performance as a function of accumulated load.

The target is a composite per-90 performance index built from position-appropriate
FBref rate stats; the model is trained per position group.
"""

from __future__ import annotations

import pandas as pd


def build_performance_index(
    per90_stats: pd.DataFrame,
    position_weights: dict[str, dict[str, float]],
) -> pd.Series:
    """Combine per-90 stats into a single performance index per appearance.

    Args:
        per90_stats: Wide dataframe of per-90 rate stats keyed by
            ``(player_id, match_date)``.
        position_weights: Mapping from position group to ``{stat: weight}``,
            expressing which stats matter per role (e.g. tackles for CB,
            xG for CF).

    Returns:
        Series of performance-index values in ``[0, 1]`` aligned to
        ``per90_stats.index``.

    Raises:
        KeyError: If a required per-90 stat column is missing.
    """
    # DESIGN: weights differ by position because a CB and a CF do not share
    # a meaningful "performance" definition — one composite scored on shared
    # weights would drown the CB signal in the CF shot volume.
    raise NotImplementedError


def fit_decay_curve(features: pd.DataFrame, target: pd.Series) -> object:
    """Fit a gradient-boosted model of performance on fatigue features.

    Args:
        features: Feature frame from :func:`fatigue.features.build_fatigue_features`.
        target: Performance-index series aligned to ``features``.

    Returns:
        Trained model (LightGBM Booster).

    Raises:
        ValueError: If ``features`` and ``target`` do not share an index.
    """
    raise NotImplementedError
