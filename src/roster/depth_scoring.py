"""Squad depth scoring — what it costs to rest a player, per role.

The question a coach faces in a congested run isn't "who is tired" but "what
does resting him cost?" Resting a starter whose backup is nearly as good costs
little; resting one whose only cover is a youth player costs a lot. This module
quantifies that, per functional role (from role_mapping), so the rotation
optimizer (M5) can trade freshness against quality loss.

APPROACH:
    1. Score each player's quality per role (offensive output + defensive work
       + minutes played as a trust/consistency proxy), role-normalized.
    2. Within each role, rank players; the top is the notional starter, the
       rest are backups (weighted by versatility from role_mapping).
    3. depth_score(role) = how well the best backup replaces the starter.
       rotation_cost(role) = 1 - depth_score: high when the drop-off is steep.

The quality score is intentionally simple and interpretable. With richer club
data the quality model becomes stronger; the architecture is the contribution,
the inputs are swappable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DESIGN: weights of the three quality ingredients. Minutes (manager trust)
# gets real weight because, absent rich performance data, sustained selection
# is itself a strong quality signal. Offensive and defensive output split rest.
W_MINUTES: float = 0.40
W_OFFENSIVE: float = 0.30
W_DEFENSIVE: float = 0.30

# DESIGN: empirical-Bayes shrinkage constant, in minutes. Small-sample per-90
# rates are unreliable (a defender who scores in one of his rare appearances
# gets an inflated off_p90 and wrongly outranks an ever-present starter). We
# shrink each player's per-90 rate toward his role mean, weighting the observed
# rate by his minutes and the role mean by K "pseudo-minutes":
#     shrunk = (minutes * observed + K * role_mean) / (minutes + K)
# K is ESTIMATED FROM THE DATA via the variance decomposition
#     K ≈ within-player variance / between-player variance  (in minutes)
# On Atalanta 2023-24 this gives K ≈ 619 min (~7 matches): noise is ~7x signal,
# so individual rates need substantial regression. A player with 4000 minutes
# keeps his real values; one with 164 (Bonfanti) is pulled hard to the role
# mean, restoring the correct ranking (Djimsiti above Bonfanti at CB).
SHRINKAGE_K: float = 619.0


def _shrink_to_role_mean(rate: pd.Series, minutes: pd.Series,
                         role_mean: float, k: float = SHRINKAGE_K) -> pd.Series:
    """Empirical-Bayes shrink a per-90 rate toward the role mean.

    Args:
        rate: Observed per-90 rate per player.
        minutes: Minutes played per player (sample size / reliability).
        role_mean: The role's minutes-weighted mean rate to shrink toward.
        k: Shrinkage constant in minutes (pseudo-observations at the mean).

    Returns:
        Shrunk rate: dominated by the observed value for high-minute players,
        pulled toward role_mean for low-minute players.
    """
    return (minutes * rate + k * role_mean) / (minutes + k)


def _normalize(s: pd.Series) -> pd.Series:
    """Min-max normalize to [0,1]; constant input -> 0.5."""
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def compute_player_quality(
    player_matches: pd.DataFrame,
    roles: pd.DataFrame,
) -> pd.DataFrame:
    """Compute a per-player quality score, role-normalized.

    Args:
        player_matches: Atalanta per-player per-match rows with minutes and
            summary stats (goals, assists, tackles won, interceptions).
        roles: Output of role_mapping.assign_roles.

    Returns:
        roles augmented with off_score, def_score, min_score (0-1,
        role-normalized) and quality (0-1), the weighted blend.
    """
    df = player_matches.copy()
    for c in ("minutes", "performance_gls", "performance_ast",
              "performance_tklw", "performance_int"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df[df["minutes"] >= 20]

    agg = df.groupby("player_name").agg(
        mins=("minutes", "sum"),
        gls=("performance_gls", "sum"),
        ast=("performance_ast", "sum"),
        tklw=("performance_tklw", "sum"),
        intc=("performance_int", "sum"),
    )
    agg["off_p90"] = (agg["gls"] + agg["ast"]) / agg["mins"] * 90
    agg["def_p90"] = (agg["tklw"] + agg["intc"]) / agg["mins"] * 90

    q = roles.merge(agg, left_on="player_name", right_index=True, how="left").fillna(0)

    # DESIGN: shrink per-90 rates toward the role mean BEFORE normalizing, so
    # small-sample players (few minutes) can't dominate on inflated rates. The
    # role mean is minutes-weighted (reliable players define the target). This
    # is empirical-Bayes shrinkage — the correct form for small-sample estimates
    # (the "cousin" of Ridge's L2 shrinkage, applied to the estimate itself
    # rather than to regression coefficients; Ridge is reserved for the fatigue
    # decay model, where correlated predictors need coefficient shrinkage).
    q["off_p90_shrunk"] = 0.0
    q["def_p90_shrunk"] = 0.0
    for role, idx in q.groupby("primary_role").groups.items():
        sub = q.loc[idx]
        # minutes-weighted role means (reliable players define the target)
        off_mean = np.average(sub["off_p90"], weights=sub["mins"]) if sub["mins"].sum() else 0.0
        def_mean = np.average(sub["def_p90"], weights=sub["mins"]) if sub["mins"].sum() else 0.0
        q.loc[idx, "off_p90_shrunk"] = _shrink_to_role_mean(sub["off_p90"], sub["mins"], off_mean)
        q.loc[idx, "def_p90_shrunk"] = _shrink_to_role_mean(sub["def_p90"], sub["mins"], def_mean)

    # DESIGN: normalize each ingredient WITHIN primary role, so quality is
    # role-appropriate (a striker judged against strikers). Uses SHRUNK rates.
    q["off_score"] = 0.0
    q["def_score"] = 0.0
    q["min_score"] = 0.0
    for role, idx in q.groupby("primary_role").groups.items():
        q.loc[idx, "off_score"] = _normalize(q.loc[idx, "off_p90_shrunk"])
        q.loc[idx, "def_score"] = _normalize(q.loc[idx, "def_p90_shrunk"])
        q.loc[idx, "min_score"] = _normalize(q.loc[idx, "mins"])

    q["quality"] = (W_MINUTES * q["min_score"]
                    + W_OFFENSIVE * q["off_score"]
                    + W_DEFENSIVE * q["def_score"])
    return q


def compute_depth_scores(quality: pd.DataFrame) -> pd.DataFrame:
    """Compute depth score and rotation cost per role.

    Args:
        quality: Output of compute_player_quality.

    Returns:
        DataFrame per role: role, starter, starter_quality, best_backup,
        backup_quality, depth_score, rotation_cost, n_options — ordered by
        rotation_cost (most vulnerable roles first).

    Note:
        A player contributes to a role's pool at his role weight (1.0 primary,
        discounted for secondary roles), so versatile players give partial
        cover away from their primary role. depth_score = backup_quality /
        starter_quality (capped at 1); rotation_cost = 1 - depth_score.
    """
    role_pools: dict[str, list[tuple[str, float]]] = {}
    for _, r in quality.iterrows():
        for role, w in r["role_weights"].items():
            eff_q = r["quality"] * w
            role_pools.setdefault(role, []).append((r["player_name"], eff_q))

    rows = []
    for role, pool in role_pools.items():
        pool = sorted(pool, key=lambda x: x[1], reverse=True)
        starter, sq = pool[0]
        backup, bq = (pool[1] if len(pool) > 1 else (None, 0.0))
        depth = min(bq / sq, 1.0) if sq > 0 else 0.0
        rows.append({
            "role": role,
            "starter": starter,
            "starter_quality": round(sq, 3),
            "best_backup": backup,
            "backup_quality": round(bq, 3),
            "depth_score": round(depth, 3),
            "rotation_cost": round(1 - depth, 3),
            "n_options": len(pool),
        })

    result = pd.DataFrame(rows).sort_values(
        "rotation_cost", ascending=False).reset_index(drop=True)
    logger.info("Depth scores computed for %d roles", len(result))
    return result