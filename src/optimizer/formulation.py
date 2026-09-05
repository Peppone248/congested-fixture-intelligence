"""Rotation optimizer - MILP formulation with PARTIAL-MINUTES levels (PuLP).

v2 redesign: instead of a binary "start / rest", each player can be assigned one
of four MINUTE LEVELS per match, capturing real squad management:

    FULL     ~90 min   (plays the whole match)
    EARLY    ~65 min   (starts, taken off to manage load)
    SUB      ~25 min   (comes off the bench)
    REST     0 min

This lets the optimizer keep a key player available three matches in a row by
managing his minutes (e.g. de Roon FULL, EARLY, FULL) instead of the rigid
binary model that had to bench him entirely - which previously forced weak
out-of-role fillers (Bakker at CM). The DOUBLE EFFECT is modelled: fewer minutes
= less performance contribution in the match, but less fatigue accumulated
(relevant once the v3 dynamic-fatigue refinement lands; here it shapes the
minutes-cap trade-off).

WHY MILP still: decisions are discrete (which level), the objective and
constraints are linear in the level indicators, CBC returns the guaranteed
optimum, fully explainable.
"""

from __future__ import annotations

import logging

import pandas as pd
import pulp

logger = logging.getLogger(__name__)

# DESIGN: Gasperini's 3-4-2-1 / 3-4-3.
FORMATION_3421: dict[str, int] = {"GK": 1, "CB": 3, "WB": 2, "CM": 2, "AM": 2, "FW": 1}
# DESIGN: Gasperini's alternative 3-5-2.
FORMATION_352: dict[str, int] = {"GK": 1, "CB": 3, "WB": 2, "CM": 3, "FW": 2}

# DESIGN: three minute levels (Strada A - no substitute modelling). FULL plays
# the whole match, EARLY starts but is managed off (~65'), REST doesn't play.
# The "who comes on" detail is deliberately NOT modelled: it adds complexity and
# little strategic value (the rotation decision is who STARTS and how their
# minutes are managed). EARLY already captures managed load; a 65' start accrues
# less fatigue than a full 90', preserving the double effect.
LEVEL_MINUTES: dict[str, float] = {"FULL": 90.0, "EARLY": 65.0, "REST": 0.0}
STARTING_LEVELS = ("FULL", "EARLY")   # both are starters for coverage
PLAYING_LEVELS = ("FULL", "EARLY")    # levels that accrue minutes/contribution

# DESIGN: performance fraction by level - a 65' managed start delivers most of a
# full game's value; REST contributes nothing.
LEVEL_PERF_FRACTION: dict[str, float] = {"FULL": 1.0, "EARLY": 0.78, "REST": 0.0}

DEFAULT_LAMBDA: float = 0.6
DEFAULT_MINUTES_CAP: float = 220.0
DEFAULT_MINUTES_PENALTY: float = 0.02
LAMBDA_CHRONIC: float = 0.25

HIGH_STAKES_IMPORTANCE: float = 0.90
MIN_QUALITY_HIGH_STAKES: float = 0.35
MIN_SEASON_MINUTES_HIGH_STAKES: float = 600.0
FATIGUE_DAMPING_MIN: float = 0.3
FATIGUE_DAMPING_MAX: float = 1.3
KEY_PLAYER_QUALITY: float = 0.65

# DESIGN: secondary-role quality discount. A STRONG player stays strong out of
# position - Koopmeiners at CM is a quality option, not a filler. So instead of
# the old flat 0.5 role weight, we discount only MILDLY (0.85) and only the part
# of quality above a floor, so top players keep most of their quality in a
# secondary role while genuine specialists still rank first in their own role.
SECONDARY_ROLE_QUALITY_KEEP: float = 0.85


def build_problem(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    lam: float = DEFAULT_LAMBDA,
    minutes_cap: float = DEFAULT_MINUTES_CAP,
    minutes_penalty: float = DEFAULT_MINUTES_PENALTY,
    formation: dict[str, int] | None = None,
) -> tuple[pulp.LpProblem, dict]:
    """Build the partial-minutes MILP rotation problem.

    Args:
        players: One row per (player, role) with columns player_name, role,
            quality (already role-weighted upstream OR raw - see note),
            is_primary_role (bool), fatigue_role_norm, season_minutes, available.
        fixtures: match_id, importance, intensity.
        lam, minutes_cap, minutes_penalty, formation: tuning / formation.

    Returns:
        (problem, index) for the solver.

    Note:
        Decision vars y[player, role, match, level] in {0,1}, exactly one level
        per (player, match) via an assignment constraint. Coverage counts
        starting levels; minutes/contribution scale by level.
    """
    formation = formation or FORMATION_3421
    P = players[players.get("available", True)].copy()
    match_ids = list(fixtures["match_id"])
    roles = list(formation.keys())
    levels = list(LEVEL_MINUTES.keys())

    importance = dict(zip(fixtures["match_id"], fixtures["importance"]))
    intensity = dict(zip(fixtures["match_id"], fixtures["intensity"]))

    # quality / fatigue / minutes maps
    qmap = {(r.player_name, r.role): r.quality for r in P.itertuples()}
    prim = {(r.player_name, r.role): getattr(r, "is_primary_role", True) for r in P.itertuples()}
    fmap = {(r.player_name, r.role): getattr(r, "fatigue_role_norm", 0.0) for r in P.itertuples()}
    smap = {r.player_name: getattr(r, "season_minutes", 9999.0) for r in P.itertuples()}

    prob = pulp.LpProblem("rotation_partial", pulp.LpMaximize)

    # --- Decision variables y[player, role, match, level] ---
    # DESIGN: eligibility filter for high-stakes matches applies to STARTING a
    # big match; a weak/low-minutes player still can't be given FULL/EARLY there.
    eligible = {(r.player_name, r.role) for r in P.itertuples()}
    y = {}
    for (pl, role) in eligible:
        for m in match_ids:
            high_stakes = importance.get(m, 0.5) >= HIGH_STAKES_IMPORTANCE
            for lv in levels:
                # bar weak/low-minute players from STARTING (FULL/EARLY) big matches
                if high_stakes and lv in STARTING_LEVELS:
                    if qmap[(pl, role)] < MIN_QUALITY_HIGH_STAKES:
                        continue
                    if smap.get(pl, 9999.0) < MIN_SEASON_MINUTES_HIGH_STAKES:
                        continue
                y[(pl, role, m, lv)] = pulp.LpVariable(
                    f"y_{pl}_{role}_{m}_{lv}".replace(" ", "_"), cat="Binary")

    players_list = sorted({pl for (pl, _, _, _) in y})

    # --- Effective quality with mild secondary-role discount ---
    # DESIGN: keep strong players strong out of position. eff_q = quality for a
    # primary role; for a secondary role, discount only the part above 0 by
    # SECONDARY_ROLE_QUALITY_KEEP, so Koopmeiners-CM stays competitive.
    def eff_quality(pl, role):
        q = qmap[(pl, role)]
        if prim.get((pl, role), True):
            return q
        return q * SECONDARY_ROLE_QUALITY_KEEP

    # --- Precompute per-(player,role,match,level) net contribution ---
    net = {}
    for (pl, role, m, lv) in y:
        q = eff_quality(pl, role)
        perf = LEVEL_PERF_FRACTION[lv]
        imp = importance.get(m, 0.5)
        high_stakes = imp >= HIGH_STAKES_IMPORTANCE
        imp_c = min(max(imp, 0.0), 1.0)
        fatigue_mult = FATIGUE_DAMPING_MAX - imp_c * (FATIGUE_DAMPING_MAX - FATIGUE_DAMPING_MIN)
        if high_stakes and q >= KEY_PLAYER_QUALITY:
            fatigue_mult = 0.0
        # DESIGN: fatigue penalty scales with minutes played (perf fraction):
        # a 25' sub accrues little fatigue cost, a 90' start the full cost.
        acute = lam * fatigue_mult * fmap[(pl, role)] * intensity.get(m, 0.5) * perf
        chronic = LAMBDA_CHRONIC * fatigue_mult * (smap.get(pl, 0.0) / max(smap.values() or [1.0])) * perf
        # contribution = quality scaled by minutes fraction, minus fatigue costs
        net[(pl, role, m, lv)] = q * perf - acute - chronic

    # --- Minutes overage (soft cap) ---
    over = {pl: pulp.LpVariable(f"over_{pl}".replace(" ", "_"), lowBound=0) for pl in players_list}

    # ===== OBJECTIVE =====
    obj = pulp.lpSum(importance.get(m, 1.0) * net[(pl, role, m, lv)] * y[(pl, role, m, lv)]
                     for (pl, role, m, lv) in y)
    obj -= pulp.lpSum(minutes_penalty * over[pl] for pl in players_list)
    prob += obj

    # ===== HARD CONSTRAINTS =====
    # (H1) exactly one level per (player, match): the player is at exactly one of
    # FULL/EARLY/SUB/REST (REST always available as the fallback).
    for pl in players_list:
        for m in match_ids:
            pv = [y[(p, r, mm, lv)] for (p, r, mm, lv) in y if p == pl and mm == m]
            if pv:
                prob += pulp.lpSum(pv) == 1, f"one_level_{pl}_{m}"

    # (H2) formation coverage: exactly the required STARTERS (FULL/EARLY) per role.
    # DESIGN: with three levels, FULL+EARLY are the starters; summing the role
    # requirements gives exactly 11 starters per match. No SUB level exists, so
    # there is no way to add extra players beyond the XI.
    for m in match_ids:
        for role in roles:
            starters = [y[(pl, r, mm, lv)] for (pl, r, mm, lv) in y
                        if r == role and mm == m and lv in STARTING_LEVELS]
            if starters:
                prob += pulp.lpSum(starters) == formation[role], f"cover_{role}_{m}"

    # ===== SOFT CONSTRAINT LINK =====
    # (S1) minutes overage: over[p] >= total assigned minutes - cap.
    for pl in players_list:
        total_min = pulp.lpSum(LEVEL_MINUTES[lv] * y[(p, r, m, lv)]
                               for (p, r, m, lv) in y if p == pl)
        prob += over[pl] >= total_min - minutes_cap, f"overage_{pl}"

    index = {"y": y, "over": over, "net": net, "match_ids": match_ids,
             "roles": roles, "players": players_list, "levels": levels,
             "importance": importance, "intensity": intensity,
             "fatigue": fmap, "quality": qmap, "eff_quality": {(pl,role): eff_quality(pl,role) for (pl,role) in eligible},
             "level_minutes": LEVEL_MINUTES, "formation": formation}
    logger.info("Built partial-minutes MILP: %d vars, %d matches, %d roles, 4 levels",
                len(y), len(match_ids), len(roles))
    return prob, index