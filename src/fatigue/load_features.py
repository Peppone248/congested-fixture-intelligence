"""Family 1 — windowed load features (player workload carried into each match).

These features describe the accumulated physical load a player brings INTO each
Atalanta fixture. They are the raw material for the fatigue model (M3) and for
the propensity model that corrects survivorship bias.

KEY DESIGN DECISIONS (see METHODOLOGY.md for the full rationale):

1. CALENDAR CROSS-JOIN. A player's FBref match log lists only matches he played.
   To know when he RESTED, we expand his log onto the full Atalanta club
   calendar: any Atalanta fixture missing from his log is a 0-minute rest.
   Without this cross-join, rests are invisible.

2. WEIGHTED REST, NOT BINARY. Rest is a continuous weight from minutes, not a
   0/1 flag. 0 min -> full rest (1.0); 90+ min -> full load (0.0); a 30-minute
   cameo -> partial rest (~0.67). Playing <45 minutes is NOT full rest — a
   substitute still accumulates fatigue. This replaces the earlier, cruder
   "<45 min = rest" rule.

3. REST IS MATCH-BASED, NOT DAY-BASED. We have no training-session data, so we
   cannot measure day-to-day load. We count matches, not calendar days. This
   avoids the temporal saturation the day-based version suffered (a season-long
   ever-starter accrued an uninformative 200+ "days since rest").

4. ACCUMULATED-FATIGUE TREND. An exponentially-weighted running sum of per-match
   load gives a smooth fatigue trajectory where recent matches weigh more than
   old ones — a trend, not just a point-in-time window.

5. INJURIES (v1 LIMITATION). A player injured for weeks also shows as 0-minute
   "rests" here. In v1 we treat these generically as "did not play". A later
   pass will cross-reference Transfermarkt injury data to distinguish planned
   turnover from injury absence — an important distinction, because a player
   RETURNING from injury is MORE fatigued than a rested healthy one, the
   opposite of what a plain rest weight implies.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DESIGN: a "full match" cap for the rest-weight scale. 90 minutes = full load.
# Minutes beyond 90 (stoppage/extra time) still map to 0 rest.
FULL_MATCH_MINUTES: float = 90.0

# DESIGN: half-life (in matches) for the accumulated-fatigue EWMA. After this
# many matches, a given match's contribution to the fatigue trend halves. 4 is
# roughly two congested weeks — long enough to accumulate, short enough to decay.
FATIGUE_HALFLIFE_MATCHES: float = 4.0

# DESIGN: window (in matches) for the rolling average of minutes — the "usage
# rhythm". 4 matches ~ a typical congested fortnight.
USAGE_WINDOW_MATCHES: int = 4


def rest_weight(minutes: float) -> float:
    """Continuous rest weight from minutes played in a match.

    Args:
        minutes: Minutes the player played (0 if absent / unused sub).

    Returns:
        1.0 for a full rest (0 min), 0.0 for a full load (>=90 min), linearly
        interpolated in between. A 30-minute cameo returns ~0.67.
    """
    return float(np.clip(1.0 - minutes / FULL_MATCH_MINUTES, 0.0, 1.0))


def expand_on_calendar(
    player_log: pd.DataFrame,
    calendar: pd.DataFrame,
    active_window: bool = True,
) -> pd.DataFrame:
    """Expand one player's match log onto the full club calendar.

    Args:
        player_log: One player's rows, with at least 'date' and 'minutes'.
        calendar: The full Atalanta club calendar, with 'date', 'competition',
            'opponent' (one row per fixture).
        active_window: If True (default), restrict the expansion to the
            player's ACTIVE WINDOW — fixtures between his first and last actual
            appearance, inclusive. If False, expand across the whole calendar.

    Returns:
        One row per club fixture in the player's active window, sorted by date,
        with:
            - minutes: minutes the player played (0 for fixtures he missed)
            - rest_weight: continuous rest weight for that fixture
            - played: bool, whether he appeared at all (minutes > 0)

    Note:
        The active-window bound is what prevents PHANTOM RESTS. A January
        signing (e.g. Hien, first Atalanta match 2024-01-03) or a departed
        player (e.g. Muriel, last match 2024-02-04) would otherwise accrue
        0-minute "rests" for every fixture outside their time at the club,
        badly distorting their load features. We only count a fixture as a
        real rest if it falls between the player's first and last appearance.
    """
    p = player_log[["date", "minutes"]].copy()

    cal = calendar[["date", "competition", "opponent"]]
    if active_window and not p.empty:
        # DESIGN: bound the calendar to [first appearance, last appearance].
        # Fixtures before a player joined or after he left are not rests —
        # he simply wasn't at the club. This is the fix for §5.2 and §5.4
        # in METHODOLOGY.md (departures and winter signings).
        first, last = p["date"].min(), p["date"].max()
        cal = cal[(cal["date"] >= first) & (cal["date"] <= last)]

    merged = cal.merge(p, on="date", how="left")
    # DESIGN: a fixture the player has no row for = he didn't play = 0 minutes.
    merged["minutes"] = merged["minutes"].fillna(0.0)
    merged["played"] = merged["minutes"] > 0
    merged["rest_weight"] = merged["minutes"].apply(rest_weight)
    return merged.sort_values("date").reset_index(drop=True)


def compute_load_features(
    player_matches: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Family-1 load features for every player, across the club calendar.

    Args:
        player_matches: Long-format Atalanta-only player match logs, with
            'player_id', 'player_name', 'date', 'minutes'.
        calendar: Full Atalanta club calendar ('date', 'competition',
            'opponent').

    Returns:
        One row per player per CLUB FIXTURE (not just matches played), with the
        load feature set:
            - minutes, played, rest_weight
            - minutes_7d, minutes_14d : minutes in the trailing 7 / 14 days
              (strictly before the fixture — no leakage)
            - starts_7d              : starts (>=45 min) in the trailing 7 days
            - matches_since_rest     : consecutive fixtures since the last full
              rest (0-minute fixture); resets on a full rest
            - weighted_load_since_rest : sum of (1 - rest_weight) since last full
              rest — partial appearances count partially
            - avg_weekly_minutes     : rolling mean of minutes over the last
              USAGE_WINDOW_MATCHES fixtures (usage rhythm)
            - fatigue_trend          : EWMA of per-match load (1 - rest_weight),
              recent matches weighted more (accumulated-fatigue trajectory)

    Note:
        Features are computed per player on the calendar-expanded frame, so
        rests are included. All windows are backward-looking and exclude the
        current fixture.
    """
    required = {"player_id", "date", "minutes"}
    missing = required - set(player_matches.columns)
    if missing:
        raise ValueError(f"player_matches missing columns: {sorted(missing)}")

    pm = player_matches.copy()
    pm["date"] = pd.to_datetime(pm["date"], errors="coerce")
    pm["minutes"] = pd.to_numeric(pm["minutes"], errors="coerce").fillna(0)
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
    cal = cal.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # DESIGN: EWMA alpha from the half-life. alpha = 1 - 2^(-1/halflife).
    alpha = 1.0 - 2 ** (-1.0 / FATIGUE_HALFLIFE_MATCHES)

    name_map = (
        pm.dropna(subset=["player_id"])
        .groupby("player_id")["player_name"].first().to_dict()
        if "player_name" in pm.columns else {}
    )

    out_frames = []
    for player_id, g in pm.groupby("player_id"):
        # Expand onto the full calendar so rests appear
        exp = expand_on_calendar(g, cal)
        exp["player_id"] = player_id
        exp["player_name"] = name_map.get(player_id, player_id)

        dates = exp["date"]
        minutes = exp["minutes"]

        # --- Trailing time-window minutes (strictly before current fixture) ---
        m7, m14, s7 = [], [], []
        for i, d in enumerate(dates):
            lo7, lo14 = d - pd.Timedelta(days=7), d - pd.Timedelta(days=14)
            prior = exp[(dates >= lo14) & (dates < d)]
            prior7 = prior[prior["date"] >= lo7]
            m7.append(prior7["minutes"].sum())
            m14.append(prior["minutes"].sum())
            s7.append(int((prior7["minutes"] >= 45).sum()))
        exp["minutes_7d"] = m7
        exp["minutes_14d"] = m14
        exp["starts_7d"] = s7

        # --- Match-based rest counters (value BEFORE the current fixture) ---
        # DESIGN: a "full rest" is a 0-minute fixture. matches_since_rest counts
        # consecutive appearances since the last full rest. weighted_load_since_
        # rest sums per-match load (1 - rest_weight) so partial games count
        # partially — a truer fatigue accumulator than a raw match count.
        matches_since, wload_since = [], []
        m_counter, w_counter = 0, 0.0
        for i in range(len(exp)):
            matches_since.append(m_counter)
            wload_since.append(w_counter)
            if minutes.iloc[i] == 0:  # full rest -> reset
                m_counter, w_counter = 0, 0.0
            else:
                m_counter += 1
                w_counter += (1.0 - exp["rest_weight"].iloc[i])
        exp["matches_since_rest"] = matches_since
        exp["weighted_load_since_rest"] = wload_since

        # --- Usage rhythm: rolling mean minutes over last N fixtures ---
        # DESIGN: shift(1) so the current fixture is excluded (no leakage).
        exp["avg_weekly_minutes"] = (
            minutes.shift(1).rolling(USAGE_WINDOW_MATCHES, min_periods=1).mean()
        )

        # --- Accumulated-fatigue trend: EWMA of per-match load ---
        # DESIGN: per-match load = (1 - rest_weight) in [0,1]. EWMA (shifted so
        # current fixture excluded) gives a smooth trajectory, recent-weighted.
        per_match_load = (1.0 - exp["rest_weight"]).shift(1)
        exp["fatigue_trend"] = per_match_load.ewm(alpha=alpha, adjust=False).mean()

        out_frames.append(exp)

    result = pd.concat(out_frames, ignore_index=True)
    # First-fixture NaNs from shifts -> 0 (no prior history)
    for col in ["avg_weekly_minutes", "fatigue_trend"]:
        result[col] = result[col].fillna(0.0)

    logger.info(
        "Family-1 load features: %d players, %d player-fixture rows",
        result["player_id"].nunique(), len(result),
    )
    return result