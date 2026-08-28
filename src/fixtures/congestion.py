"""Compute the congestion index: team-level and player-level.

This is the analytical heart of module M2. It transforms a plain fixture
calendar into a structured description of scheduling pressure.

THE KEY CONCEPTUAL DISTINCTION — team vs player congestion:

- **Team-level congestion** describes the SCHEDULING PRESSURE around a match:
  how many matches the team plays in a 7/14/21-day window, how many recovery
  hours separate them, how many competition switches. It uses windows CENTRED
  on each match because it describes the density of the fixture environment.

- **Player-level congestion** describes the WORKLOAD A SPECIFIC PLAYER CARRIES
  INTO a match: minutes accumulated in the last 7/14 days, consecutive full
  matches, days since a rest. It uses BACKWARD-LOOKING windows because it
  describes fatigue that has already accumulated.

Why this matters: a team can face EXTREME scheduling congestion (3 matches in
7 days) while an individual player experiences only MODERATE workload, because
the manager rotated him. That gap between team pressure and player workload is
exactly what creates the survivorship bias the fatigue model has to correct
for. If we only measured team congestion, we'd miss it entirely.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# DESIGN: encoding tier levels as an ordered categorical avoids fragile
# string comparisons and lets pandas group / plot in the right order.
CONGESTION_LEVELS: tuple[str, ...] = ("NORMAL", "MODERATE", "HEAVY", "EXTREME")

# DESIGN: a "full match" threshold below 90 absorbs stoppage time and minor
# data imprecision. A player recorded at 88 minutes started and finished; we
# don't want to miss that as a "full match" because of a 2-minute rounding.
FULL_MATCH_THRESHOLD: int = 85

# DESIGN: a "start" threshold — 45 minutes means the player played at least a
# half. Below this we treat the appearance as a substitute cameo, not a start.
START_THRESHOLD: int = 45


# ===================================================================
# TEAM-LEVEL CONGESTION
# ===================================================================

def rolling_match_count(dates: pd.Series, window_days: int) -> pd.Series:
    """Count matches within a centred window of +/- window_days/2 around each match.

    Args:
        dates: Ascending series of fixture dates (datetime64).
        window_days: Total window width in days. The window is centred on each
            match, extending +/- window_days/2 on each side.

    Returns:
        Integer series aligned to ``dates`` giving the match count within the
        centred window (inclusive of the current fixture).

    Raises:
        ValueError: If ``dates`` is not sorted ascending.

    Note:
        We use a CENTRED window (not backward-looking) for team congestion
        because scheduling pressure is symmetric: a match sandwiched between
        two others is congested regardless of which side the neighbours are on.
    """
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending.")

    half = pd.Timedelta(days=window_days / 2)
    counts = []
    for d in dates:
        lo, hi = d - half, d + half
        counts.append(int(((dates >= lo) & (dates <= hi)).sum()))
    return pd.Series(counts, index=dates.index)


def compute_team_congestion(
    calendar: pd.DataFrame,
    tier_config: dict | None = None,
) -> pd.DataFrame:
    """Compute all team-level congestion features for each match.

    Args:
        calendar: Fixture calendar from ``parser.build_fixture_calendar``,
            sorted by date, with columns: date, competition, kickoff_time,
            travel_load.
        tier_config: Parsed ``config/congestion_tiers.yaml``. If None, uses
            built-in default thresholds.

    Returns:
        The calendar with these columns added:
            - matches_in_7d, matches_in_14d, matches_in_21d: int
            - hours_since_last_match: float (NaN for first match)
            - hours_to_next_match: float (NaN for last match)
            - min_recovery_hours: float — min of the two above
            - short_recovery: bool — True if min_recovery_hours < 72
            - consecutive_short_recovery: int — running streak of short-recovery matches
            - competition_switches_7d: int — distinct competitions in the 7-day window
            - travel_load_14d: float — sum of travel_load in the trailing 14 days
            - congestion_tier: ordered categorical from CONGESTION_LEVELS

    Raises:
        ValueError: If the calendar isn't sorted or lacks a date column.
    """
    if "date" not in calendar.columns:
        raise ValueError("calendar must have a 'date' column.")
    if not calendar["date"].is_monotonic_increasing:
        raise ValueError("calendar must be sorted by date.")

    df = calendar.copy().reset_index(drop=True)
    dates = df["date"]

    # ---- Rolling match counts (centred windows) ----
    df["matches_in_7d"] = rolling_match_count(dates, 7)
    df["matches_in_14d"] = rolling_match_count(dates, 14)
    df["matches_in_21d"] = rolling_match_count(dates, 21)

    # ---- Recovery hours (uses kickoff times via the parser helper) ----
    from src.fixtures.parser import compute_rest_hours
    df["hours_since_last_match"] = compute_rest_hours(df)
    # hours_to_next is just the next match's hours_since_last, shifted up
    df["hours_to_next_match"] = df["hours_since_last_match"].shift(-1)

    # DESIGN: min_recovery_hours captures the tightest turnaround touching this
    # match — whether the squeeze is before or after. This is what actually
    # limits recovery around a given fixture.
    df["min_recovery_hours"] = df[
        ["hours_since_last_match", "hours_to_next_match"]
    ].min(axis=1)

    # DESIGN: sanity check for data quality. Recovery hours below ~40h are
    # physically implausible for two competitive matches (even a Sat 20:45 →
    # Mon 18:45 turnaround is ~46h). Negative or near-zero values signal a
    # data problem: duplicate fixtures, a same-day collision, or a parsing
    # error. We warn loudly rather than silently modelling on bad data.
    implausible = df["hours_since_last_match"].notna() & (df["hours_since_last_match"] < 40)
    if implausible.any():
        bad = df.loc[implausible, ["date", "opponent", "hours_since_last_match"]]
        logger.warning(
            "%d matches have implausibly short recovery (<40h) — possible "
            "duplicate or same-day fixtures. Inspect:\n%s",
            implausible.sum(), bad.to_string(index=False),
        )

    # ---- Short-recovery flag and consecutive streak ----
    df["short_recovery"] = df["hours_since_last_match"] < 72

    # DESIGN: consecutive_short_recovery counts how many short-turnaround
    # matches the team has played in an unbroken run ending at this match.
    # A run of 3 means the team has been in "back-to-back-to-back" mode —
    # cumulative fatigue that a single-match count doesn't capture.
    streak = 0
    streaks = []
    for is_short in df["short_recovery"]:
        streak = streak + 1 if is_short else 0
        streaks.append(streak)
    df["consecutive_short_recovery"] = streaks

    # ---- Competition switches in the trailing 7 days ----
    # DESIGN: switching between Serie A and Champions League within a week
    # adds cognitive/tactical load beyond the physical. Count distinct
    # competitions in the trailing 7-day window.
    comp_switches = []
    for i, d in enumerate(dates):
        lo = d - pd.Timedelta(days=7)
        window = df[(dates >= lo) & (dates <= d)]
        comp_switches.append(window["competition"].nunique())
    df["competition_switches_7d"] = comp_switches

    # ---- Travel load in the trailing 14 days ----
    if "travel_load" in df.columns:
        travel_14d = []
        for i, d in enumerate(dates):
            lo = d - pd.Timedelta(days=14)
            window = df[(dates >= lo) & (dates <= d)]
            travel_14d.append(window["travel_load"].sum())
        df["travel_load_14d"] = travel_14d
    else:
        df["travel_load_14d"] = 0.0

    # ---- Tier classification ----
    df["congestion_tier"] = _classify_tiers(df, tier_config)

    logger.info(
        "Team congestion computed: %s",
        df["congestion_tier"].value_counts().to_dict(),
    )

    return df


def _classify_tiers(df: pd.DataFrame, tier_config: dict | None) -> pd.Categorical:
    """Assign a congestion tier to each match.

    Args:
        df: Calendar with matches_in_7d and min_recovery_hours computed.
        tier_config: Tier threshold config, or None for defaults.

    Returns:
        Ordered categorical series with tier labels.

    Note:
        The logic: start from the match-count tier, then let a short recovery
        UPGRADE the tier (never downgrade). A match can look calm by 7-day
        count but still be a brutal 48-hour turnaround — we never mask that.
    """
    # DESIGN: default thresholds encode the recovery science — full recovery
    # needs ~72–96h, so <72h between matches is the "insufficient recovery"
    # line, and 3 matches in 7 days is the "extreme" line.
    if tier_config is None:
        tier_config = {
            "tiers": {
                "NORMAL": {"matches_in_7d": {"max": 1}, "min_recovery_hours": {"min": 120}},
                "MODERATE": {"matches_in_7d": {"max": 2}, "min_recovery_hours": {"min": 72}},
                "HEAVY": {"matches_in_7d": {"max": 2}, "min_recovery_hours": {"max": 72}},
                "EXTREME": {"matches_in_7d": {"min": 3}},
            }
        }

    tiers = tier_config.get("tiers", {})

    def classify_row(row) -> str:
        m7 = row["matches_in_7d"]
        rec = row["min_recovery_hours"]
        # DESIGN (recalibrated after EDA): tier is driven PRIMARILY by recovery
        # hours, not raw match count. Rationale: in elite football with European
        # competition, "2 matches in 7 days" (Thu EL + Sun league) is the NORMAL
        # rhythm, not congestion. Classifying it as HEAVY made >50% of the
        # season HEAVY, destroying the tier's discriminative power. What
        # actually stresses players is SHORT RECOVERY between matches (<72h)
        # and TRIPLE weeks (3 matches / 7 days), so those drive the tiers.
        #
        #   EXTREME  : 3+ matches in 7 days (triple week — unavoidable overload)
        #   HEAVY    : short recovery (<72h) into or out of this match
        #   MODERATE : 2 matches in 7 days with adequate recovery (72–96h)
        #   NORMAL   : isolated match, or 96h+ recovery on both sides
        if m7 >= 3:
            return "EXTREME"
        if pd.notna(rec) and rec < 72:
            return "HEAVY"
        if m7 >= 2 and pd.notna(rec) and rec < 96:
            return "MODERATE"
        return "NORMAL"

    labels = df.apply(classify_row, axis=1)
    return pd.Categorical(labels, categories=CONGESTION_LEVELS, ordered=True)


# Keep the original stub's function name as an alias for backward-compat
def classify_congestion(calendar: pd.DataFrame, tier_config: dict) -> pd.Series:
    """Assign a congestion tier to every fixture (thin wrapper).

    See :func:`compute_team_congestion` for the full feature computation.
    This wrapper exists for the original module interface: it returns just
    the tier series.
    """
    enriched = compute_team_congestion(calendar, tier_config)
    return enriched["congestion_tier"]


# ===================================================================
# PLAYER-LEVEL CONGESTION
# ===================================================================

def compute_player_congestion(
    player_match_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-player, per-match workload features (backward-looking).

    Args:
        player_match_logs: Long-format player match logs with columns:
            player_id, date, minutes. One row per player per match played.
            (As produced by fbref.fetch_all_player_match_logs, summary stat.)

    Returns:
        The same rows with these workload features added:
            - player_minutes_7d: minutes played in the 7 days BEFORE this match
            - player_minutes_14d: minutes in the 14 days before
            - player_starts_7d: matches started (>= 45 min) in the last 7 days
            - player_consecutive_full: current streak of full matches (>= 85 min)
            - player_days_since_rest: days since the player last played < 45 min
              (or since season start if they've never rested)

    Raises:
        ValueError: If required columns are missing.

    Note:
        All windows are BACKWARD-LOOKING and EXCLUSIVE of the current match:
        we're measuring the load the player brings INTO this match, not
        including the match itself. This is the opposite of team congestion's
        centred windows — and the difference is intentional.
    """
    required = {"player_id", "date", "minutes"}
    missing = required - set(player_match_logs.columns)
    if missing:
        raise ValueError(
            f"player_match_logs missing columns: {sorted(missing)}. "
            f"Got: {sorted(player_match_logs.columns)}"
        )

    df = player_match_logs.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)

    # Process each player independently, then recombine
    out_frames = []
    for player_id, g in df.groupby("player_id"):
        g = g.sort_values("date").reset_index(drop=True)
        dates = g["date"]
        minutes = g["minutes"]

        minutes_7d = []
        minutes_14d = []
        starts_7d = []
        for i, d in enumerate(dates):
            # DESIGN: strictly-before window (d - 7d <= x < d). The current
            # match is excluded so the feature is "load carried in", usable
            # as a predictor of this match's performance without leakage.
            lo7 = d - pd.Timedelta(days=7)
            lo14 = d - pd.Timedelta(days=14)
            prior = g[(dates >= lo14) & (dates < d)]
            prior7 = prior[prior["date"] >= lo7]
            minutes_7d.append(prior7["minutes"].sum())
            minutes_14d.append(prior["minutes"].sum())
            starts_7d.append(int((prior7["minutes"] >= START_THRESHOLD).sum()))

        g["player_minutes_7d"] = minutes_7d
        g["player_minutes_14d"] = minutes_14d
        g["player_starts_7d"] = starts_7d

        # DESIGN: consecutive full matches — streak of >= 85 min appearances
        # ending at (but not including) the current match. High values mean
        # the player has been carrying the team without rotation.
        consec = []
        streak = 0
        for i in range(len(g)):
            consec.append(streak)  # value BEFORE this match
            if minutes.iloc[i] >= FULL_MATCH_THRESHOLD:
                streak += 1
            else:
                streak = 0
        g["player_consecutive_full"] = consec

        # DESIGN: days since last rest — a "rest" is any match where the player
        # played < 45 min (benched, cameo, or absent). Measures how long
        # they've gone without a breather.
        days_since_rest = []
        last_rest_date = dates.iloc[0] - pd.Timedelta(days=7)  # assume rested before season
        for i in range(len(g)):
            days_since_rest.append((dates.iloc[i] - last_rest_date).days)
            if minutes.iloc[i] < START_THRESHOLD:
                last_rest_date = dates.iloc[i]
        g["player_days_since_rest"] = days_since_rest

        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    logger.info(
        "Player congestion computed for %d players, %d player-match rows",
        result["player_id"].nunique(), len(result),
    )
    return result


# ===================================================================
# SUMMARY / REPORTING
# ===================================================================

def congestion_summary(team_congestion: pd.DataFrame) -> dict:
    """Produce summary statistics for a team-season's congestion profile.

    Args:
        team_congestion: Output of compute_team_congestion.

    Returns:
        Dict with headline numbers: total matches, matches per tier, longest
        short-recovery streak, tightest turnaround, busiest calendar month.
    """
    df = team_congestion
    tier_counts = df["congestion_tier"].value_counts().to_dict()

    tightest_idx = df["min_recovery_hours"].idxmin() if df["min_recovery_hours"].notna().any() else None
    tightest = None
    if tightest_idx is not None:
        row = df.loc[tightest_idx]
        tightest = {
            "date": row["date"].date().isoformat(),
            "opponent": row.get("opponent", "?"),
            "recovery_hours": round(row["min_recovery_hours"], 1),
        }

    busiest_month = (
        df.assign(month=df["date"].dt.to_period("M"))
        .groupby("month")
        .size()
        .idxmax()
    )

    return {
        "total_matches": len(df),
        "matches_per_tier": {str(k): int(v) for k, v in tier_counts.items()},
        "longest_short_recovery_streak": int(df["consecutive_short_recovery"].max()),
        "tightest_turnaround": tightest,
        "busiest_month": str(busiest_month),
        "matches_in_busiest_month": int(
            df.assign(month=df["date"].dt.to_period("M")).groupby("month").size().max()
        ),
    }