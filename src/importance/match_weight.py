"""Weight fixtures by strategic importance.

The optimizer objective is expected-performance summed *weighted by match
importance* — a coach rests players ahead of a knockout, not ahead of a mid-
table cup tie.
"""

from __future__ import annotations

import pandas as pd


# DESIGN: baseline weights per competition are configurable and combined
# multiplicatively with a stage multiplier so a UCL group-stage away tie and
# a UCL semifinal are not treated the same.
COMPETITION_BASE_WEIGHTS: dict[str, float] = {
    "UEFA Champions League": 1.4,
    "UEFA Europa League": 1.2,
    "Serie A": 1.0,
    "Coppa Italia": 0.7,
}


def compute_match_weights(
    calendar: pd.DataFrame,
    league_position: pd.Series | None = None,
) -> pd.Series:
    """Compute a scalar importance weight per fixture.

    Args:
        calendar: Fixture calendar with a ``competition`` column and, where
            available, ``stage`` (e.g. group, R16, QF).
        league_position: Optional series of team league positions on each
            match-date; the closer to a promotion / relegation / title cutoff,
            the higher the derived multiplier.

    Returns:
        Positive-valued series aligned to ``calendar.index``.

    Raises:
        KeyError: If ``calendar`` lacks the ``competition`` column.
    """
    raise NotImplementedError
