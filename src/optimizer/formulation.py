"""Mixed-integer program formulation for multi-match rotation.

We optimise player-to-match assignments over a rolling window, maximising
expected weighted performance subject to formation, availability, and
minimum-rest constraints.
"""

from __future__ import annotations

import pandas as pd


# DESIGN: modelling the whole window (not one match at a time) is what lets
# the optimizer trade a mid-week rest for a stronger weekend XI — a single-
# match objective would always start the best XI and never rotate.
DEFAULT_WINDOW_MATCHES: int = 3


def build_lp(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    expected_performance: pd.DataFrame,
    match_weights: pd.Series,
    formation_slots: list[str],
) -> object:
    """Construct the PuLP problem for one rotation window.

    Args:
        players: Roster rows with canonical position and availability flag.
        fixtures: Fixture calendar for the window, in order.
        expected_performance: Predicted per-90 performance per (player, match)
            from the fatigue decay model.
        match_weights: Importance weight per fixture (see :mod:`importance`).
        formation_slots: Position slots the formation requires, e.g. from
            ``config/positions.yaml``.

    Returns:
        A ``pulp.LpProblem`` with variables, objective, and constraints attached.

    Raises:
        ValueError: If any fixture lacks 11 fillable slots given availability.
    """
    # DESIGN: binary decision var x[p, m] = 1 iff player p starts match m.
    # Constraints enforce: exactly one player per slot per match, no double
    # assignments within a match, and minimum rest between consecutive
    # starts for the same player.
    raise NotImplementedError
