"""Estimate the recovery-days required to restore a player to baseline performance.

The estimator inverts the decay model: given a target performance level and a
current fatigue state, it returns the rest days needed to reach that level.
"""

from __future__ import annotations

import pandas as pd


# DESIGN: age and position modulate recovery in the literature — a 33-year-old
# CB does not recover on the same curve as a 22-year-old winger — so the
# estimator conditions on both.
BASELINE_TARGET: float = 0.95


def estimate_recovery_days(
    decay_model: object,
    current_features: pd.DataFrame,
    target_level: float = BASELINE_TARGET,
    max_days: int = 10,
) -> pd.Series:
    """Days of rest needed to reach ``target_level`` predicted performance.

    Args:
        decay_model: Trained decay model from :mod:`fatigue.decay_model`.
        current_features: Present-day fatigue features per player.
        target_level: Performance-index level considered "recovered".
        max_days: Cap for the search — beyond this the player is flagged
            as unable to recover in-window.

    Returns:
        Series aligned to ``current_features.index`` giving the estimated
        recovery days, capped at ``max_days``.

    Raises:
        ValueError: If ``target_level`` is outside ``(0, 1]``.
    """
    # DESIGN: search recovery days by simulating what the feature vector
    # looks like after N rest days, then querying the decay model — this
    # reuses the trained model rather than fitting a separate inverse model.
    raise NotImplementedError
