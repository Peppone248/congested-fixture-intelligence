"""Player-importance scoring: how much does a specific player raise the ceiling?

Value combines Transfermarkt market value (as a prior on ceiling talent),
observed minutes share (as a revealed-preference signal from the coach), and
positional depth (a star at a shallow position is more critical).
"""

from __future__ import annotations

import pandas as pd


def compute_player_value(
    roster: pd.DataFrame,
    depth_chart: pd.DataFrame,
    minutes_share: pd.Series,
) -> pd.Series:
    """Score each player's importance to the squad.

    Args:
        roster: Canonical roster with ``market_value_eur``.
        depth_chart: Depth chart from :mod:`roster.depth`.
        minutes_share: Series of minutes-share per player (0-1).

    Returns:
        Positive-valued series indexed by ``player_id`` — larger is more
        important to the squad.

    Raises:
        ValueError: If any input series index does not align with the roster.
    """
    # DESIGN: log-transform the market value first — the Transfermarkt scale
    # is heavy-tailed, and a raw multiplication makes the top three names
    # dominate the optimizer's objective.
    raise NotImplementedError
