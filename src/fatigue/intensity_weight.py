"""Intensity weight — how physically taxing a match was for a given player.

The ideal intensity measure is physical (distance covered, sprints, high-speed
running), but that data is tracking-only and not public for Serie A. We build a
multi-source PROXY instead, from data we do have, combining three components
(additive, weighted — transparent and robust to a single anomalous input):

    1. MATCH CONTEXT (team level, same for all players in a match):
       - total match xG (open, end-to-end games are more taxing)
       - Atalanta PPDA (low = high pressing = high physical output)
       - deep completions (attacking volume / actions created)
       - possession (low possession = chasing the ball = more running)

    2. PLAYER CONTRIBUTION (per player in the match):
       - tackles won + interceptions + fouls per 90 (defensive work done)

    3. OPPONENT STRENGTH (team level):
       - the dynamic match_weight (harder opponents → more running)

The output multiplies match minutes in the fatigue trend: 90 minutes in a
high-intensity match weigh more than 90 in a passive one.

HONEST LIMITATION (see METHODOLOGY.md): this is a proxy. Player heatmaps and
true distance/sprint data require positional/tracking data (SkillCorner,
Catapult, StatsBomb 360) — commercial, club-held. With the club's GPS data this
proxy becomes a direct physical-load measure.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DESIGN: component weights for the additive blend. Match context is the
# largest share because it's the richest signal (four sub-inputs from Understat);
# player contribution and opponent strength refine it. Documented and tunable.
W_CONTEXT: float = 0.45
W_PLAYER: float = 0.30
W_OPPONENT: float = 0.25


def _normalize(s: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalize a series to [0, 1], optionally inverted.

    Args:
        s: Numeric series.
        invert: If True, low raw values map to high normalized values (used for
            PPDA, where LOW ppda means HIGH pressing intensity).

    Returns:
        Series in [0, 1]. Constant input maps to 0.5 (neutral).
    """
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    z = (s - lo) / (hi - lo)
    return 1.0 - z if invert else z


def compute_match_context_intensity(context: pd.DataFrame) -> pd.Series:
    """Team-level match intensity from Understat context + possession.

    Args:
        context: Per-match frame with columns ata_xg, opp_xg, ata_ppda,
            ata_deep, and optionally poss (possession %).

    Returns:
        Series in [0, 1]: how intense the match was, team level.

    Note:
        # DESIGN: four sub-signals, each normalized then averaged:
        #   - total xG (higher = more open = more taxing)
        #   - PPDA (LOWER = more pressing = more taxing → inverted)
        #   - deep completions (higher = more attacking volume)
        #   - possession (LOWER = more chasing → inverted); optional
    """
    total_xg = context["ata_xg"] + context["opp_xg"]
    parts = [
        _normalize(total_xg),
        _normalize(context["ata_ppda"], invert=True),  # low ppda = high pressing
        _normalize(context["ata_deep"]),
    ]
    if "poss" in context.columns:
        parts.append(_normalize(context["poss"], invert=True))  # low poss = chasing
    return pd.concat(parts, axis=1).mean(axis=1)


def compute_player_contribution_intensity(
    player_matches: pd.DataFrame,
) -> pd.Series:
    """Per-player defensive workload intensity from summary stats.

    Args:
        player_matches: Per-player per-match rows with minutes and the summary
            defensive columns (performance_tklw, performance_int, performance_fls).

    Returns:
        Series in [0, 1] aligned to player_matches.index: how much defensive
        work the player did per 90 in that match, normalized across the season.

    Note:
        # DESIGN: tackles won + interceptions + fouls, per 90, is a crude but
        # real proxy for how much running/dueling a player did. Normalized
        # across all player-matches so it's comparable. Players with < 20
        # minutes are left as NaN (too noisy per-90).
    """
    df = player_matches.copy()
    for c in ("performance_tklw", "performance_int", "performance_fls", "minutes"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    work = (df.get("performance_tklw", 0).fillna(0)
            + df.get("performance_int", 0).fillna(0)
            + df.get("performance_fls", 0).fillna(0))
    per90 = np.where(df["minutes"] >= 20, work / df["minutes"] * 90, np.nan)
    per90 = pd.Series(per90, index=df.index)
    return _normalize(per90.fillna(per90.median()))


def compute_intensity_weight(
    player_matches: pd.DataFrame,
    context: pd.DataFrame,
    match_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the three components into a per-player-per-match intensity weight.

    Args:
        player_matches: Atalanta per-player per-match rows (with player_id,
            date, minutes, summary defensive stats).
        context: Understat per-match context (ata_xg, opp_xg, ata_ppda, ata_deep),
            keyed by date.
        match_weights: Per-match dynamic weights (with date, match_weight).

    Returns:
        player_matches with added columns:
            - context_intensity   : team-level match intensity (0-1)
            - player_intensity    : player defensive workload (0-1)
            - opponent_intensity  : normalized dynamic match_weight (0-1)
            - intensity_weight    : additive blend (0-1)

    Note:
        Matches without Understat context (European / cup) get a neutral 0.5
        context_intensity, so the weight still computes (documented limitation).
    """
    df = player_matches.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # --- Component 1: match context (team level), merged by date ---
    ctx = context.copy()
    ctx["date"] = pd.to_datetime(ctx["date"]).dt.date
    ctx["context_intensity"] = compute_match_context_intensity(ctx)
    df = df.merge(ctx[["date", "context_intensity"]], on="date", how="left")
    # DESIGN: matches with no Understat data (cups/Europe) → neutral 0.5.
    df["context_intensity"] = df["context_intensity"].fillna(0.5)

    # --- Component 2: player contribution ---
    df["player_intensity"] = compute_player_contribution_intensity(df)

    # --- Component 3: opponent strength (normalized match_weight) ---
    mw = match_weights.copy()
    mw["date"] = pd.to_datetime(mw["date"]).dt.date
    mw["opponent_intensity"] = _normalize(mw["match_weight"])
    df = df.merge(mw[["date", "opponent_intensity"]], on="date", how="left")
    df["opponent_intensity"] = df["opponent_intensity"].fillna(0.5)

    # --- Additive weighted blend ---
    df["intensity_weight"] = (
        W_CONTEXT * df["context_intensity"]
        + W_PLAYER * df["player_intensity"]
        + W_OPPONENT * df["opponent_intensity"]
    )

    logger.info(
        "Intensity weight computed: %d rows, mean %.2f, range %.2f–%.2f",
        len(df), df["intensity_weight"].mean(),
        df["intensity_weight"].min(), df["intensity_weight"].max(),
    )
    return df