"""Continuous decay model: how much does expected performance drop per unit load?

Wraps the boosted tree from :mod:`fatigue.performance` with a monotonic
constraint on the fatigue axis and exposes SHAP-based attribution helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# DESIGN: LightGBM's monotone constraints let us enforce "more load ⇒ lower or
# equal predicted performance" on the fatigue features — this is the whole
# claim of the model, so it should be a hard constraint, not a soft prior.
MONOTONE_FEATURES: tuple[str, ...] = (
    "minutes_last_7d",
    "minutes_last_14d",
    "matches_last_21d",
    "consecutive_starts",
)


def train_decay_model(features: pd.DataFrame, target: pd.Series) -> object:
    """Train the fatigue-constrained decay model.

    Args:
        features: Fatigue feature frame.
        target: Performance index aligned to ``features``.

    Returns:
        Trained LightGBM Booster with monotone constraints applied to the
        columns listed in :data:`MONOTONE_FEATURES`.

    Raises:
        ValueError: If any monotone feature is absent from ``features``.
    """
    raise NotImplementedError


def predict_decay(model: object, features: pd.DataFrame) -> np.ndarray:
    """Predict per-appearance performance under given fatigue features.

    Args:
        model: Trained model from :func:`train_decay_model`.
        features: Feature frame with the same schema used at training time.

    Returns:
        Array of predicted performance indices.

    Raises:
        ValueError: If ``features`` schema does not match the trained model.
    """
    raise NotImplementedError


def shap_attribution(model: object, features: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions per feature for each row of ``features``.

    Args:
        model: Trained decay model.
        features: Feature frame to attribute.

    Returns:
        Dataframe of SHAP values with the same shape as ``features``.

    Raises:
        NotImplementedError: Attribution not yet implemented.
    """
    raise NotImplementedError
