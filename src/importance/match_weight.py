"""Match / opponent weighting — how much each fixture 'matters'.

A rotation decision isn't only about fatigue; it's about WHICH matches to spend
freshness on. Resting a key player against a relegation side costs little;
resting him against a title rival, or in a European knockout, costs a lot. This
module scores each fixture on two axes and combines them:

    1. OPPONENT DIFFICULTY — how strong is the opponent?
    2. MATCH IMPORTANCE    — how much does this specific match matter
                             (competition stage, season context)?

The combined weight feeds two places downstream:
    - the rotation optimizer's objective (spend freshness where it matters);
    - the intensity model (harder matches tend to be more physically taxing).

DESIGN — why a-priori tiers, not final league position:
    Using an opponent's FINAL league position would leak the future: when
    Atalanta played Frosinone in September, nobody knew they'd finish 18th. A
    manager planning rotation uses PRIOR EXPECTATION of strength (Inter are
    strong, a promoted side less so). We therefore assign each opponent an
    a-priori strength tier reflecting pre-season expectation, which is what a
    coach would actually plan around. A future refinement can blend in
    pre-match form (rolling points) without ever using end-of-season data.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Opponent strength tiers (a-priori, 2023-24 Serie A)
# ---------------------------------------------------------------------------
# DESIGN: strength on a 0-1 scale by tier. These reflect pre-season expectation
# of quality, NOT final table position (see module docstring). Tiers:
#   ELITE (0.95)   — title/Champions-League contenders
#   STRONG (0.75)  — European-place contenders
#   MID (0.55)     — established mid-table
#   MODEST (0.40)  — lower-mid table
#   WEAK (0.25)    — promoted / relegation-battlers
#
# Documented per-team so the assignment is auditable by a club analyst.
SERIE_A_STRENGTH: dict[str, float] = {
    # Elite — title & top-4 contenders
    "Inter": 0.95, "Juventus": 0.90, "Milan": 0.88, "Napoli": 0.88,
    # Strong — Europa/Conference contenders
    "Roma": 0.78, "Lazio": 0.78, "Fiorentina": 0.70, "Bologna": 0.68,
    # Mid — established mid-table
    "Torino": 0.58, "Monza": 0.52, "Genoa": 0.50, "Lecce": 0.45,
    "Udinese": 0.50, "Sassuolo": 0.50,
    # Modest / weak — lower table, promoted, relegation battlers
    "Cagliari": 0.38, "Empoli": 0.38, "Hellas Verona": 0.38,
    "Frosinone": 0.30, "Salernitana": 0.25,
}

# DESIGN: European opponents scored by continental pedigree that season. These
# are coarse but defensible (Liverpool/Leverkusen elite; Marseille/Sporting
# strong; group-stage minnows weak). Leverkusen were the 2023-24 unbeaten
# Bundesliga champions; Liverpool a European heavyweight.
EUROPEAN_STRENGTH: dict[str, float] = {
    "Liverpool": 0.95, "Leverkusen": 0.95, "Marseille": 0.72,
    "Sporting CP": 0.70, "Sturm Graz": 0.45, "Raków": 0.40,
}

# DESIGN: fallback when an opponent isn't in the tables above — mid strength,
# so an unknown opponent never silently becomes trivially weak or elite.
DEFAULT_STRENGTH: float = 0.55

# ---------------------------------------------------------------------------
# Competition base weights
# ---------------------------------------------------------------------------
# DESIGN: how much a match in each competition/stage 'matters', 0-1. A European
# knockout is worth more than a group game; a Coppa Italia early round less than
# a league match. These are planning weights, tunable per club priorities.
COMPETITION_WEIGHT: dict[str, float] = {
    "Serie A": 0.80,
    "Europa Lg": 0.75,        # baseline; knockout bumps it up (see stage logic)
    "Coppa Italia": 0.55,
}

# DESIGN: stage multipliers for knockout competitions. A final matters far more
# than a group game. Matched loosely against the 'round' text FBref provides.
STAGE_MULTIPLIER: list[tuple[str, float]] = [
    ("final", 1.30),
    ("semi", 1.20),
    ("quarter", 1.12),
    ("round of 16", 1.06),
    ("knockout", 1.06),
    ("group", 0.90),
]


def compute_match_weight_dynamic(
    calendar: pd.DataFrame,
    long_results: pd.DataFrame,
    home_advantage: float = 0.10,
    form_n: int = 5,
    form_weight: float = 0.40,
    opponent_ppda: dict | None = None,
    tactical_weight: float = 0.25,
) -> pd.DataFrame:
    """Compute match weights using DYNAMIC opponent strength (position + form)
    and OPPONENT TACTICAL STYLE (pressing intensity from PPDA).

    Opponent strength is computed as of each fixture date from the live league
    table and recent form (blended with an a-priori seed early season). On top
    of strength, an opponent's TACTICAL STYLE is factored in: a high-pressing
    opponent (low PPDA) makes a match more taxing to PLAY — especially for
    Gasperini's Atalanta, whose man-to-man press and build-from-the-back are
    disrupted by an aggressive presser. Style is separate from strength (a weak
    side can press hard; a strong side can sit back), so it enters as its own
    multiplier on difficulty.

    Args:
        calendar: Atalanta club calendar with 'date', 'competition',
            'opponent', 'venue', 'round'.
        long_results: Long per-team-per-match results for standings/form.
        home_advantage: Downward difficulty adjustment for home matches.
        form_n: Form window length (matches).
        form_weight: Weight of form vs position in the live component.
        opponent_ppda: Optional dict mapping fixture date (datetime.date) to the
            opponent's PPDA in that match (low = high pressing). If None, the
            tactical component is neutral (no effect).
        tactical_weight: How much opponent pressing style lifts difficulty
            (0.25 = up to +25% for the highest-pressing opponent).

    Returns:
        The calendar with added columns:
            - opp_position, opp_strength, opp_blend
            - opp_press_intensity : 0-1, opponent pressing intensity (1 = presses
              hardest across the season); neutral 0.5 when PPDA is unavailable
            - difficulty          : opp_strength adjusted for home/away
            - tactical_multiplier : 1 + tactical_weight * (press_intensity - 0.5)*2
              clamped so a passive opponent slightly lowers, a high-press one
              raises the effective difficulty
            - competition_weight, stage_mult, match_importance
            - match_weight        : difficulty * tactical_multiplier * importance

    Note:
        PPDA (from Understat) is Serie-A-only, so European/cup opponents get the
        neutral 0.5 tactical intensity — the same competition-coverage boundary
        as opponent strength.
    """
    from src.importance.standings import opponent_strength_dynamic

    df = calendar.copy()
    df["date"] = pd.to_datetime(df["date"])

    positions, strengths, blends = [], [], []
    for _, r in df.iterrows():
        s = opponent_strength_dynamic(
            long_results, r["opponent"], r["date"],
            form_n=form_n, form_weight=form_weight,
        )
        positions.append(s["position"])
        strengths.append(s["strength"])
        blends.append(s["blend"])

    df["opp_position"] = positions
    df["opp_strength"] = strengths
    df["opp_blend"] = blends

    # Home advantage adjustment
    is_home = df["venue"].astype(str).str.lower().str.startswith("home")
    df["difficulty"] = (df["opp_strength"] - is_home * home_advantage).clip(0.0, 1.0)

    # --- Opponent tactical style: pressing intensity from PPDA ---
    # DESIGN: PPDA (passes allowed per defensive action) is LOW for a high press.
    # We invert and normalize it across the season's opponents to a 0-1 pressing
    # intensity (1 = presses hardest). Missing PPDA (Europe/cup) -> neutral 0.5.
    if opponent_ppda:
        df["_ppda"] = df["date"].dt.date.map(opponent_ppda)
        valid = df["_ppda"].dropna()
        if len(valid) > 1:
            lo, hi = valid.min(), valid.max()
            # invert: low ppda -> high intensity
            df["opp_press_intensity"] = df["_ppda"].apply(
                lambda p: (1.0 - (p - lo) / (hi - lo)) if pd.notna(p) and hi > lo else 0.5)
        else:
            df["opp_press_intensity"] = 0.5
        df["opp_press_intensity"] = df["opp_press_intensity"].fillna(0.5)
        df = df.drop(columns=["_ppda"])
    else:
        df["opp_press_intensity"] = 0.5

    # DESIGN: tactical multiplier centered on 1.0. A neutral opponent (0.5
    # intensity) leaves difficulty unchanged; the highest presser lifts it by
    # +tactical_weight, the most passive lowers it by -tactical_weight. This is
    # the Gasperini-specific nuance: facing a high press is more taxing to play.
    df["tactical_multiplier"] = 1.0 + tactical_weight * (df["opp_press_intensity"] - 0.5) * 2

    # Match importance (same as before)
    df["competition_weight"] = df["competition"].map(COMPETITION_WEIGHT).fillna(0.60)
    df["stage_mult"] = df["round"].apply(stage_multiplier)
    df["match_importance"] = df["competition_weight"] * df["stage_mult"]

    # DESIGN: match_weight now folds in the tactical multiplier alongside
    # difficulty and importance.
    df["match_weight"] = df["difficulty"] * df["tactical_multiplier"] * df["match_importance"]

    logger.info(
        "Dynamic match weights (with tactical style): %d fixtures, "
        "range %.2f–%.2f, mean %.2f",
        len(df), df["match_weight"].min(), df["match_weight"].max(),
        df["match_weight"].mean(),
    )
    return df
    """Return an a-priori strength score (0-1) for an opponent.

    Args:
        opponent: Opponent team name (country-prefix already stripped).
        competition: Competition name, to pick the right strength table.

    Returns:
        Strength in [0, 1]; DEFAULT_STRENGTH if the opponent is unknown.
    """
    if competition == "Serie A":
        return SERIE_A_STRENGTH.get(opponent, DEFAULT_STRENGTH)
    if competition == "Europa Lg":
        return EUROPEAN_STRENGTH.get(opponent, DEFAULT_STRENGTH)
    # Coppa Italia opponents are Serie A sides — reuse that table
    return SERIE_A_STRENGTH.get(opponent, DEFAULT_STRENGTH)


def stage_multiplier(round_label: str) -> float:
    """Return a knockout-stage multiplier from the round label.

    Args:
        round_label: FBref 'round' text (e.g. 'Group stage', 'Final').

    Returns:
        A multiplier; 1.0 if no stage keyword matches (e.g. league matchweeks).
    """
    if not isinstance(round_label, str):
        return 1.0
    low = round_label.lower()
    for keyword, mult in STAGE_MULTIPLIER:
        if keyword in low:
            return mult
    return 1.0


def compute_match_weight(
    calendar: pd.DataFrame,
    home_advantage: float = 0.10,
) -> pd.DataFrame:
    """Compute opponent-difficulty and match-importance weights per fixture.

    Args:
        calendar: Atalanta club calendar with 'competition', 'opponent',
            'venue', 'round'.
        home_advantage: How much easier a home match is treated as, as a
            downward adjustment to difficulty (0.10 = 10% easier at home).

    Returns:
        The calendar with added columns:
            - opp_strength        : a-priori opponent strength (0-1)
            - difficulty          : opp_strength adjusted for home/away
            - competition_weight  : base weight for the competition
            - stage_mult          : knockout-stage multiplier
            - match_importance    : competition_weight * stage_mult (0-1+)
            - match_weight        : combined difficulty * importance, the single
              number the optimizer/intensity model consume

    Note:
        `difficulty` and `match_importance` are kept as separate columns on
        purpose — they answer different questions ("how hard?" vs "how much does
        it matter?") and a club may want to weight them differently. `match_
        weight` is the default combination for convenience.
    """
    df = calendar.copy()

    # --- Opponent difficulty ---
    df["opp_strength"] = df.apply(
        lambda r: opponent_strength(r["opponent"], r["competition"]), axis=1
    )
    # DESIGN: home matches are easier — reduce difficulty by home_advantage.
    # Away/neutral keep full difficulty. Clip to [0,1].
    is_home = df["venue"].astype(str).str.lower().str.startswith("home")
    df["difficulty"] = (df["opp_strength"] - is_home * home_advantage).clip(0.0, 1.0)

    # --- Match importance ---
    df["competition_weight"] = df["competition"].map(COMPETITION_WEIGHT).fillna(0.60)
    df["stage_mult"] = df["round"].apply(stage_multiplier)
    df["match_importance"] = df["competition_weight"] * df["stage_mult"]

    # --- Combined weight ---
    # DESIGN: multiply difficulty and importance. A hard AND important match
    # (away to Inter, or a European knockout) scores highest; an easy, low-
    # stakes match (home Coppa Italia vs a weak side) scores lowest. This is
    # the single knob the optimizer uses to decide where to spend freshness.
    df["match_weight"] = df["difficulty"] * df["match_importance"]

    logger.info(
        "Match weights computed for %d fixtures. Range: %.2f–%.2f, mean %.2f",
        len(df), df["match_weight"].min(), df["match_weight"].max(),
        df["match_weight"].mean(),
    )
    return df