"""Turn raw match-log rows into a canonical fixture calendar.

The fixture calendar is the temporal spine of the whole pipeline: every
downstream module (congestion, fatigue, recovery, optimizer) joins on
``(team, match_date)``.

This module does three jobs:

1. **Normalize** raw scraper output into a fixed, validated schema. If FBref
   renames a column or changes date formats, we fail loudly here rather than
   letting bad data leak into the models.

2. **Enrich** the calendar with derived scheduling context that isn't in the
   raw data: travel load (home / away / European away) and kickoff slots
   (which affect the *effective* recovery window between matches).

3. **Order** the calendar chronologically and deduplicate, so downstream
   caches are stable across re-scrapes.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
# DESIGN: this is the canonical schema every downstream module relies on.
# The parser's job is to guarantee these columns exist and are correctly
# typed. If the raw data can't produce them, we raise — no silent NaN.

REQUIRED_INPUT_COLUMNS: set[str] = {"date", "competition", "opponent", "venue"}

# DESIGN: European competitions get a higher travel-load weight. This list is
# matched case-insensitively against the competition name. It's a heuristic —
# we don't know actual travel distances, but "played a European away leg"
# is a reasonable proxy for the flight + time-zone + logistics burden that
# domestic away matches don't carry.
EUROPEAN_COMPETITIONS: tuple[str, ...] = (
    "champions league",
    "europa league",
    "conference league",
    "uefa",
)


def build_fixture_calendar(
    match_logs: pd.DataFrame,
    team_name: str,
) -> pd.DataFrame:
    """Normalise raw match logs into a per-team fixture calendar.

    Args:
        match_logs: Concatenated match logs from one or more competitions,
            as returned by ``fbref.fetch_team_match_log``. Must contain at
            least: date, competition, opponent, venue.
        team_name: Canonical team name used to label the output rows.

    Returns:
        DataFrame sorted by date (ascending, deduplicated) with columns:
            - team: str — the canonical team name
            - date: datetime64 — match date
            - kickoff_time: str — kickoff time if available, else NaN
            - competition: str
            - round: str — matchday / round label if available
            - venue: str — "Home" or "Away"
            - opponent: str
            - goals_for, goals_against: Int64 (nullable)
            - xg, xga: float
            - is_european: bool — True for UEFA competitions
            - travel_load: float — 0 (home) / 1 (away domestic) / 2 (away European)
            - kickoff_slot: str — early_afternoon / late_afternoon / evening / night / unknown
            - match_report_url: str (if available)

    Raises:
        ValueError: If required columns are missing, or if no rows survive
            date parsing (which usually means the date format changed).
    """
    df = match_logs.copy()

    # ---- Normalize raw FBref column names ----
    # DESIGN: the parser accepts either already-normalized columns (from the
    # scraper) OR raw FBref abbreviations (comp, gf, ga). We apply the rename
    # map defensively so the parser works standalone in tests and notebooks,
    # not only when fed through the scraper's normalization.
    raw_rename = {
        "comp": "competition",
        "gf": "goals_for",
        "ga": "goals_against",
        "match_report": "match_report_url",
    }
    df = df.rename(columns={k: v for k, v in raw_rename.items() if k in df.columns})

    # ---- Validation: fail loudly if the input schema is wrong ----
    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"match_logs is missing required columns: {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}. "
            f"Check that the scraper output matches the expected FBref schema."
        )

    # ---- Date parsing ----
    # DESIGN: coerce to datetime, then drop unparseable rows. FBref match log
    # tables include future (unplayed) fixtures with valid dates but empty
    # results, plus occasional separator rows. We keep future fixtures (they
    # matter for the optimizer) but drop anything without a real date.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["date"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d rows with unparseable dates", n_dropped)

    if df.empty:
        raise ValueError(
            "No rows survived date parsing. The date column format may have "
            "changed, or the input was empty."
        )

    # ---- Team label ----
    df["team"] = team_name

    # ---- Optional columns: fill with sensible defaults if absent ----
    # DESIGN: kickoff_time, round, and match_report_url are nice-to-have but
    # not guaranteed. We create them if missing so the output schema is stable.
    for opt_col in ("kickoff_time", "time", "round", "match_report_url"):
        if opt_col not in df.columns:
            df[opt_col] = pd.NA

    # Normalize "time" → "kickoff_time" if the scraper used "time"
    if "time" in df.columns and df["kickoff_time"].isna().all():
        df["kickoff_time"] = df["time"]

    # ---- Numeric columns ----
    for col in ("goals_for", "goals_against"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = pd.Series([pd.NA] * len(df), dtype="Int64")

    for col in ("xg", "xga"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    # ---- Enrichment: European flag ----
    # DESIGN: computed once here and reused by travel_load and (later) by
    # match-importance weighting.
    df["is_european"] = (
        df["competition"]
        .astype(str)
        .str.lower()
        .apply(lambda c: any(euro in c for euro in EUROPEAN_COMPETITIONS))
    )

    # ---- Enrichment: venue normalization ----
    # DESIGN: FBref uses "Home"/"Away"/"Neutral". We normalize casing and
    # treat "Neutral" (cup finals) as "Away" for travel purposes — the team
    # still travels, it's just not the opponent's ground.
    df["venue"] = df["venue"].astype(str).str.strip().str.capitalize()

    # ---- Enrichment: travel load ----
    df["travel_load"] = df.apply(_compute_travel_load, axis=1)

    # ---- Enrichment: kickoff slot ----
    df["kickoff_slot"] = df["kickoff_time"].apply(_classify_kickoff_slot)

    # ---- Enrichment: clean opponent names ----
    # DESIGN: FBref prefixes foreign opponents with a 2-letter country code in
    # European competitions ("pl Raków", "de Leverkusen", "eng Liverpool").
    # We strip a leading lowercase 2-3 letter token so the opponent name joins
    # cleanly with other sources (Transfermarkt, standings). Domestic Serie A
    # opponents have no prefix and are unaffected.
    df["opponent"] = (
        df["opponent"]
        .astype(str)
        .str.replace(r"^[a-z]{2,3}\s+", "", regex=True)
        .str.strip()
    )

    # ---- Sort and deduplicate ----
    # DESIGN: dedup on (date, opponent, competition) — the same match can
    # appear twice if a team's schedule is scraped from multiple sources.
    # Sorting by date makes the calendar the stable temporal spine.
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date", "opponent", "competition"])
        .reset_index(drop=True)
    )

    # ---- Final column ordering ----
    ordered_cols = [
        "team", "date", "kickoff_time", "kickoff_slot", "competition", "round",
        "venue", "opponent", "goals_for", "goals_against", "xg", "xga",
        "is_european", "travel_load", "match_report_url",
    ]
    present = [c for c in ordered_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in present]
    df = df[present + remaining]

    logger.info(
        "Built calendar for %s: %d matches, %s to %s, competitions=%s",
        team_name, len(df),
        df["date"].min().date(), df["date"].max().date(),
        sorted(df["competition"].unique()),
    )

    return df


def _compute_travel_load(row: pd.Series) -> float:
    """Compute a travel-load proxy for a single match.

    Args:
        row: A calendar row with 'venue' and 'is_european' fields.

    Returns:
        0.0 for home matches, 1.0 for domestic away, 2.0 for European away.

    Note:
        This is a deliberately coarse proxy. We don't have actual travel
        distances or routes. The signal we're capturing is ordinal: home =
        no travel, domestic away = some travel, European away = significant
        travel (flight, time zone, logistics). The fatigue model can learn
        whether this ordinal matters; if it doesn't, the feature drops out.
    """
    venue = str(row.get("venue", "")).lower()
    is_home = venue.startswith("home")

    if is_home:
        return 0.0
    # Away (or neutral) match
    if row.get("is_european", False):
        return 2.0
    return 1.0


def _classify_kickoff_slot(kickoff_time) -> str:
    """Classify a kickoff time into a coarse slot.

    Args:
        kickoff_time: A time string like "20:45" or "15:00", or NA.

    Returns:
        One of: early_afternoon (before 15:00), late_afternoon (15:00–17:59),
        evening (18:00–20:29), night (20:30+), or unknown (unparseable).

    Note:
        Kickoff slot matters because the *effective* recovery window between
        two matches depends on kickoff times, not just calendar days. A 20:45
        Saturday match followed by a 12:30 Tuesday match gives less real
        recovery than two 15:00 matches the same number of days apart. This
        feature lets the congestion index and fatigue model account for that.
    """
    if pd.isna(kickoff_time):
        return "unknown"

    time_str = str(kickoff_time).strip()
    # DESIGN: parse just the hour. FBref times look like "20:45" or "20:45 (19:45)"
    # for matches with local + venue time. We take the first HH:MM we find.
    import re
    match = re.search(r"(\d{1,2}):(\d{2})", time_str)
    if not match:
        return "unknown"

    hour = int(match.group(1))
    minute = int(match.group(2))
    total_minutes = hour * 60 + minute

    if total_minutes < 15 * 60:            # before 15:00
        return "early_afternoon"
    elif total_minutes < 18 * 60:          # 15:00–17:59
        return "late_afternoon"
    elif total_minutes < 20 * 60 + 30:     # 18:00–20:29
        return "evening"
    else:                                  # 20:30+
        return "night"


def compute_rest_days(calendar: pd.DataFrame) -> pd.Series:
    """Compute days of rest between consecutive fixtures.

    Args:
        calendar: Fixture calendar as returned by :func:`build_fixture_calendar`.

    Returns:
        Series aligned to ``calendar.index`` giving the number of full days
        between each fixture and its immediate predecessor (``NaN`` for the
        first fixture of the season).

    Raises:
        ValueError: If ``calendar`` is not sorted by date.
    """
    if "date" not in calendar.columns:
        raise ValueError("calendar must have a 'date' column.")

    # DESIGN: verify sorting rather than silently re-sorting. If the calendar
    # arrives unsorted, that's a bug upstream we want to know about.
    if not calendar["date"].is_monotonic_increasing:
        raise ValueError(
            "calendar must be sorted by date before computing rest days. "
            "Call build_fixture_calendar first."
        )

    # Difference in days between consecutive matches
    rest = calendar["date"].diff().dt.total_seconds() / (24 * 3600)
    return rest


def compute_rest_hours(calendar: pd.DataFrame) -> pd.Series:
    """Compute hours of rest between consecutive fixtures, using kickoff times.

    This is more precise than :func:`compute_rest_days` because it accounts
    for kickoff times. Two matches three calendar days apart give 72 hours if
    both kick off at the same time, but only ~64 hours if the first is at
    20:45 and the second at 12:30.

    Args:
        calendar: Fixture calendar with 'date' and optionally 'kickoff_time'.

    Returns:
        Series of rest hours between each match and its predecessor. Falls
        back to date-only (assuming 15:00 kickoffs) when kickoff_time is
        missing.
    """
    if not calendar["date"].is_monotonic_increasing:
        raise ValueError("calendar must be sorted by date.")

    # DESIGN: build a full datetime by combining date + kickoff_time. When
    # kickoff_time is missing, assume 15:00 (a neutral mid-afternoon default)
    # so the estimate is reasonable rather than absent.
    def _to_datetime(row) -> pd.Timestamp:
        base = row["date"]
        kt = row.get("kickoff_time")
        if pd.isna(kt):
            return base + pd.Timedelta(hours=15)
        import re
        m = re.search(r"(\d{1,2}):(\d{2})", str(kt))
        if not m:
            return base + pd.Timedelta(hours=15)
        return base + pd.Timedelta(hours=int(m.group(1)), minutes=int(m.group(2)))

    kickoff_dt = calendar.apply(_to_datetime, axis=1)
    rest_hours = kickoff_dt.diff().dt.total_seconds() / 3600
    return rest_hours