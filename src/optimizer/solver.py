"""Solve the rotation LP and unpack the result into a lineup schedule."""

from __future__ import annotations

import pandas as pd


# DESIGN: CBC ships with PuLP and needs no external install, which keeps the
# project runnable in CI and on the Streamlit sandbox without licenses.
DEFAULT_SOLVER: str = "PULP_CBC_CMD"


def solve(problem: object, time_limit_s: int = 60) -> dict[str, pd.DataFrame]:
    """Solve the rotation LP and return the lineups per match.

    Args:
        problem: PuLP problem from :func:`optimizer.formulation.build_lp`.
        time_limit_s: Wall-clock cap for the solver.

    Returns:
        Dictionary with two dataframes:
        ``"lineups"``: rows of (match, position_slot, player_id);
        ``"summary"``: objective value, gap, wall-time.

    Raises:
        RuntimeError: If the solver returns an infeasible or unbounded status.
    """
    raise NotImplementedError
