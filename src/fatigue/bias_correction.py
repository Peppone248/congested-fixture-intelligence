"""Correct for selection bias in observational fatigue data.

Coaches rest tired players, so the observed set of appearances is *not* a
random sample of (player, fatigue-level) combinations. We use doubly-robust
causal estimators (EconML / DoWhy) to recover the underlying fatigue effect.
"""

from __future__ import annotations

import pandas as pd


def build_treatment_frame(
    features: pd.DataFrame,
    treatment_col: str = "matches_last_21d",
) -> pd.DataFrame:
    """Shape features + performance into an EconML-ready frame.

    Args:
        features: Fatigue feature frame.
        treatment_col: The fatigue variable treated as the causal driver.

    Returns:
        Dataframe with columns ``T`` (treatment), ``Y`` (outcome), and
        ``X`` (confounders) suitable for a doubly-robust learner.

    Raises:
        KeyError: If ``treatment_col`` is missing from ``features``.
    """
    # DESIGN: keeping the treatment column configurable lets us re-run the
    # same identification pipeline for different fatigue proxies without
    # duplicating the estimator wiring.
    raise NotImplementedError


def fit_doubly_robust(frame: pd.DataFrame) -> object:
    """Fit an EconML doubly-robust learner on the treatment frame.

    Args:
        frame: Output of :func:`build_treatment_frame`.

    Returns:
        Fitted causal estimator exposing ``effect`` / ``ate`` methods.

    Raises:
        NotImplementedError: Estimator wiring not yet implemented.
    """
    raise NotImplementedError
