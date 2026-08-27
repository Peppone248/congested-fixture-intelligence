"""Schema and integrity checks for pipeline dataframes.

All cross-module handoffs go through these checks so schema drift in one
scraper doesn't silently corrupt downstream models.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Assert that every named column is present.

    Args:
        frame: Dataframe to check.
        columns: Required column names.

    Returns:
        None. The function only raises on failure.

    Raises:
        ValueError: If any required column is missing, listing the missing set.
    """
    raise NotImplementedError


def require_monotonic_date_index(frame: pd.DataFrame) -> None:
    """Assert the dataframe has a strictly monotonic ascending datetime index.

    Args:
        frame: Dataframe to check.

    Returns:
        None.

    Raises:
        ValueError: If the index is not a DatetimeIndex or not sorted ascending.
    """
    # DESIGN: many rolling-window features silently return wrong numbers when
    # given a shuffled index — catching that here beats debugging it in
    # downstream metrics.
    raise NotImplementedError
