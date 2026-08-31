"""FBref scraper: match logs, per-match minutes, and per-90 performance rates.

FBref is the primary data source for the congested-fixture-intelligence pipeline.
It provides two things we can't get anywhere else for free:

1. **Team match schedules** across all competitions (Serie A + CL + Coppa Italia)
   in a single, consistent format — this becomes the fixture calendar.

2. **Player-level per-match statistics** (minutes, xG, xA, progressive actions,
   defensive actions) — this feeds the fatigue-performance decay model.

FBref has two scraping quirks that this module handles:

- **Comment-wrapped tables:** many stat tables are inside HTML comments
  (``<!-- <table>...</table> -->``). JavaScript renders them in the browser,
  but ``requests.get()`` + ``pandas.read_html()`` can't see them.
  We strip the comment markers before parsing.

- **Multi-level headers:** stat tables often have two header rows
  (a group row and a column row). ``pandas.read_html()`` creates a
  MultiIndex; we flatten it into clean single-level column names.

Rate limiting: FBref enforces aggressive rate limits. This module does NOT
manage delays directly — it delegates to ``src.utils.cache.get_cached_or_fetch``
which handles caching + polite delays in one place.
"""

from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Comment

from src.scrapers.downloader import download_fbref_html
from src.utils.cache import cache_path_for_url
from src.utils.constants import FBREF_CACHE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FBref URL patterns
# ---------------------------------------------------------------------------
# DESIGN: URL templates live here as module constants, not buried in function
# bodies. When FBref changes their URL scheme (happens ~once a year), there's
# exactly one place to update.

FBREF_BASE = "https://fbref.com"

# Team match log — all competitions for one season
# Example: https://fbref.com/en/squads/cd051869/2023-2024/matchlogs/all_comps/schedule/Atalanta-Scores-and-Fixtures-All-Competitions
TEAM_MATCH_LOG_URL = (
    "{base}/en/squads/{team_id}/{season}/matchlogs/all_comps/schedule/"
)

# Player match log — one stat type for one season
# Example: https://fbref.com/en/players/553fef74/matchlogs/2023-2024/summary/Ademola-Lookman-Match-Logs
PLAYER_MATCH_LOG_URL = (
    "{base}/en/players/{player_id}/matchlogs/{season}/{stat_type}/"
)

# Team roster page — squad list with season stats
# Example: https://fbref.com/en/squads/cd051869/2023-2024/Atalanta-Stats
TEAM_ROSTER_URL = "{base}/en/squads/{team_id}/{season}/"

# DESIGN: delay is set at 5 seconds — FBref returns 429s if you go faster.
# This is only used if the cache module's default isn't appropriate.
DEFAULT_DELAY_S: float = 5.0


# ---------------------------------------------------------------------------
# UTILITY: Strip HTML comments to expose hidden tables
# ---------------------------------------------------------------------------

def _strip_comments(html: str) -> str:
    """Remove HTML comment markers to expose FBref's hidden stat tables.

    FBref wraps many data tables inside ``<!-- ... -->`` to prevent naive
    scraping. The tables are real HTML — they just need the comment delimiters
    stripped so that a parser can see them.

    Args:
        html: Raw HTML string from a FBref page.

    Returns:
        The same HTML with all comment-wrapped ``<table>`` blocks unwrapped.
        Non-table comments (like copyright notices) are left untouched.

    Note:
        This is the single most important utility in the scraper. Without it,
        ``pandas.read_html()`` finds only 1-2 tables on a page that actually
        has 10+. Every FBref scraper on GitHub solves this problem; the ones
        that don't are broken.
    """
    # DESIGN: use BeautifulSoup to find Comment nodes, check if they contain
    # a <table> tag, and if so, replace the Comment with its contents.
    # This is more robust than regex because comments can be nested or span
    # multiple lines unpredictably.
    soup = BeautifulSoup(html, "lxml")

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment_text = str(comment)
        # DESIGN: only unwrap comments that contain a <table> — we don't want
        # to expose random HTML fragments or JavaScript that FBref also hides
        # in comments.
        if "<table" in comment_text:
            # Replace the comment node with the actual HTML content
            new_soup = BeautifulSoup(comment_text, "lxml")
            comment.replace_with(new_soup)
            logger.debug("Unwrapped a comment-hidden table")

    return str(soup)


# ---------------------------------------------------------------------------
# UTILITY: Flatten multi-level headers
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column header into single-level snake_case names.

    FBref stat tables often have two header rows:

        |          Performance          |   Expected    |
        | Gls | Ast | PK | PKatt | ... | xG  | xAG | ...

    ``pandas.read_html()`` reads these as a MultiIndex. This function
    collapses them into ``performance_gls``, ``expected_xg``, etc.

    Args:
        df: DataFrame with potentially multi-level columns.

    Returns:
        DataFrame with single-level, lowercase, underscore-separated columns.
        Duplicate column names (which happen when FBref uses the same name
        in different groups) get a numeric suffix.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        # DESIGN: if columns are already flat, just normalize to snake_case
        df.columns = [
            re.sub(r"[^\w]+", "_", str(c)).strip("_").lower()
            for c in df.columns
        ]
        return df

    # DESIGN: join the levels with underscore, but skip generic group names
    # like "Unnamed: 0_level_0" that pandas generates for ungrouped columns.
    flat = []
    for parts in df.columns:
        cleaned = []
        for p in parts:
            s = str(p)
            # Skip pandas-generated filler names
            if s.startswith("Unnamed") or s.startswith("level_"):
                continue
            cleaned.append(s)
        name = "_".join(cleaned) if cleaned else f"col_{len(flat)}"
        flat.append(re.sub(r"[^\w]+", "_", name).strip("_").lower())

    # DESIGN: handle duplicates by appending a counter — this happens when
    # FBref uses "Gls" in both "Performance" and "Expected" groups.
    seen: dict[str, int] = {}
    deduped = []
    for name in flat:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)

    df.columns = deduped
    return df


# ---------------------------------------------------------------------------
# UTILITY: Find and parse a specific table by ID or position
# ---------------------------------------------------------------------------

def _extract_table(html: str, table_id: str | None = None,
                   table_index: int = 0) -> pd.DataFrame:
    """Extract and parse a single table from FBref HTML.

    Args:
        html: HTML string (already comment-stripped).
        table_id: If given, find the table by its HTML ``id`` attribute.
            If None, use ``table_index`` instead.
        table_index: Zero-based index into all ``<table>`` elements.
            Only used when ``table_id`` is None.

    Returns:
        Parsed DataFrame with flattened column names.

    Raises:
        ValueError: If the specified table is not found.
    """
    if table_id:
        # DESIGN: use regex match parameter in read_html to target a specific
        # table by its id attribute. This is more reliable than positional
        # indexing, which breaks when FBref adds or removes tables.
        try:
            # DESIGN: attrs={"id": ...} alone selects the table — no need for
            # the match parameter. Passing match=None would cause a TypeError
            # in pandas because it tries to compile None as a regex.
            tables = pd.read_html(
                StringIO(html),
                attrs={"id": table_id},
            )
        except ValueError:
            raise ValueError(
                f"No table found with id='{table_id}'. "
                f"FBref may have changed their page structure."
            )
        if not tables:
            raise ValueError(f"Table id='{table_id}' matched but returned empty.")
        df = tables[0]
    else:
        try:
            tables = pd.read_html(StringIO(html))
        except ValueError:
            raise ValueError("No tables found in the HTML.")
        if table_index >= len(tables):
            raise ValueError(
                f"Requested table_index={table_index} but only "
                f"{len(tables)} tables found."
            )
        df = tables[table_index]

    # Flatten multi-level headers
    df = _flatten_columns(df)

    # DESIGN: FBref inserts separator rows with the column headers repeated
    # in the data. These rows have the same values as the column names.
    # Drop them by checking if the first column's value equals its name.
    first_col = df.columns[0]
    df = df[df[first_col] != first_col].reset_index(drop=True)

    return df


# ===================================================================
# PUBLIC API: Team match log (the fixture calendar)
# ===================================================================

def fetch_team_match_log(
    team_id: str,
    season: str,
    competition: str | None = None,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Fetch a team's match-by-match log for one season, all competitions.

    This is the **fixture calendar** — the temporal spine of the pipeline.
    Every downstream module joins on (team, match_date).

    Args:
        team_id: FBref team identifier (8-char hex slug from the team URL).
            Example: ``"cd051869"`` for Atalanta.
        season: Season label in FBref format, e.g. ``"2023-2024"``.
        competition: Optional filter. If given (e.g. ``"Serie A"``), only
            matches in that competition are returned. If None, all
            competitions are included.
        cache_dir: Override for the HTML cache directory. Defaults to
            the project's standard FBref cache path.

    Returns:
        DataFrame with one row per match, columns:
            - date: datetime — match date
            - time: str — kickoff time (local)
            - competition: str — competition name (Serie A, UEFA Europa League, ...)
            - round: str — matchday or round name
            - venue: str — "Home" or "Away"
            - opponent: str — opponent team name
            - result: str — "W", "D", or "L"
            - goals_for: int — goals scored
            - goals_against: int — goals conceded
            - xg: float — expected goals
            - xga: float — expected goals against
            - match_report_url: str — link to detailed match report

    Raises:
        ValueError: If the expected table is not found in the HTML.
    """
    _cache = Path(cache_dir) if cache_dir else FBREF_CACHE_DIR

    # DESIGN: the "all_comps/schedule" URL gives us every match across
    # Serie A, Champions League, Coppa Italia, etc. in a single table.
    # This is better than scraping each competition separately because:
    # 1. One HTTP request instead of 3-4
    # 2. No deduplication needed
    # 3. The congestion index needs the COMPLETE calendar, not per-competition
    url = TEAM_MATCH_LOG_URL.format(
        base=FBREF_BASE, team_id=team_id, season=season,
    )

    logger.info("Fetching team match log: team=%s season=%s", team_id, season)
    cache_file = cache_path_for_url(url, _cache)
    raw_html = download_fbref_html(url, cache_path=cache_file)

    # Strip HTML comments to expose hidden tables
    clean_html = _strip_comments(raw_html)

    # DESIGN: the match log table has id "matchlogs_for" on team schedule pages.
    # This ID has been stable for years but could change — the ValueError
    # from _extract_table will tell us immediately if it does.
    df = _extract_table(clean_html, table_id="matchlogs_for")

    # --- Column normalization ---
    # DESIGN: FBref column names vary slightly across seasons and locales.
    # We normalize to a fixed schema so downstream code never breaks on
    # a renamed column.
    rename_map = {
        "date": "date",
        "time": "time",
        "comp": "competition",
        "round": "round",
        "venue": "venue",
        "opponent": "opponent",
        "result": "result",
        "gf": "goals_for",
        "ga": "goals_against",
        "xg": "xg",
        "xga": "xga",
        "match_report": "match_report_url",
    }

    # DESIGN: only rename columns that actually exist — FBref sometimes
    # omits columns (e.g., xG is missing for some lower leagues).
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    # Parse date to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Drop rows without a valid date (separator rows, future fixtures)
        df = df.dropna(subset=["date"])

    # Parse numeric columns
    for col in ["goals_for", "goals_against", "xg", "xga"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply competition filter if requested
    if competition and "competition" in df.columns:
        df = df[df["competition"].str.contains(competition, case=False, na=False)]

    df = df.reset_index(drop=True)

    logger.info(
        "Parsed %d matches for team=%s season=%s (competitions: %s)",
        len(df), team_id, season,
        df["competition"].unique().tolist() if "competition" in df.columns else "unknown",
    )

    return df


# ===================================================================
# PUBLIC API: Player match log (per-player per-match stats)
# ===================================================================

def fetch_player_match_log(
    player_id: str,
    season: str,
    stat_type: str = "summary",
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Fetch per-match statistics for a single player in one season.

    Args:
        player_id: FBref player identifier (8-char hex slug from player URL).
            Example: ``"553fef74"`` for Ademola Lookman.
        season: Season label in FBref format, e.g. ``"2023-2024"``.
        stat_type: Which stat table to fetch. Options:
            - ``"summary"``: minutes, goals, assists, xG, xA
            - ``"passing"``: pass completion, progressive passes, key passes
            - ``"defense"``: tackles, interceptions, blocks, clearances
            - ``"possession"``: touches, carries, progressive carries, take-ons
        cache_dir: Override for the HTML cache directory.

    Returns:
        DataFrame with one row per match, columns vary by stat_type.
        All stat types include: date, competition, opponent, venue,
        result, minutes.

    Raises:
        ValueError: If stat_type is not recognized or table not found.
    """
    valid_stat_types = {"summary", "passing", "defense", "possession",
                        "gca", "misc", "keeper"}
    if stat_type not in valid_stat_types:
        raise ValueError(
            f"stat_type='{stat_type}' not recognized. "
            f"Valid options: {sorted(valid_stat_types)}"
        )

    _cache = Path(cache_dir) if cache_dir else FBREF_CACHE_DIR

    url = PLAYER_MATCH_LOG_URL.format(
        base=FBREF_BASE, player_id=player_id,
        season=season, stat_type=stat_type,
    )

    logger.info(
        "Fetching player match log: player=%s season=%s stat=%s",
        player_id, season, stat_type,
    )
    cache_file = cache_path_for_url(url, _cache)
    raw_html = download_fbref_html(url, cache_path=cache_file)
    clean_html = _strip_comments(raw_html)

    # DESIGN: player match log tables use id "matchlogs_all" for the
    # all-competitions view. This is different from the team table ID.
    df = _extract_table(clean_html, table_id="matchlogs_all")

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    # Parse minutes — FBref sometimes uses "90" or "90'" or has empty cells
    if "min" in df.columns:
        df["min"] = (
            df["min"]
            .astype(str)
            .str.replace("'", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
        df = df.rename(columns={"min": "minutes"})

    # Parse all plausibly numeric columns
    # DESIGN: coerce errors rather than raising — FBref occasionally has
    # non-numeric values in stat columns (e.g., "-" for a keeper who didn't
    # face a shot). Coercion gives us NaN, which downstream code handles.
    skip_cols = {"date", "competition", "comp", "opponent", "venue",
                 "result", "round", "match_report", "pos", "started"}
    for col in df.columns:
        if col not in skip_cols and df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only replace if at least some values converted successfully
            if converted.notna().any():
                df[col] = converted

    # DESIGN: normalize column names to the canonical schema so player match
    # logs use the same names as the team calendar ('competition', not 'comp').
    # Downstream congestion/fatigue code joins on these names.
    player_rename = {
        "comp": "competition",
        "squad": "team",
        "opponent": "opponent",
        "round": "round",
        "venue": "venue",
        "result": "result",
        "pos": "position",
        "gls": "goals",
        "ast": "assists",
    }
    df = df.rename(columns={k: v for k, v in player_rename.items() if k in df.columns})

    df = df.reset_index(drop=True)

    logger.info(
        "Parsed %d match entries for player=%s (%s)",
        len(df), player_id, stat_type,
    )

    return df


# ===================================================================
# PUBLIC API: Team roster (squad list with player IDs)
# ===================================================================

def fetch_team_roster(
    team_id: str,
    season: str,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Fetch the squad roster with FBref player IDs.

    This function extracts both the statistical table AND the player URLs
    from the team page. The player URLs contain the player_id needed for
    ``fetch_player_match_log``.

    Args:
        team_id: FBref team identifier.
        season: Season label in FBref format.
        cache_dir: Override for the HTML cache directory.

    Returns:
        DataFrame with columns:
            - player_name: str
            - player_id: str — the 8-char hex ID from the player's FBref URL
            - player_url: str — full FBref player URL
            - position: str — primary position
            - age: int
            - minutes: int — total minutes in the season
            + additional season-total stats (goals, assists, etc.)
    """
    _cache = Path(cache_dir) if cache_dir else FBREF_CACHE_DIR

    url = TEAM_ROSTER_URL.format(
        base=FBREF_BASE, team_id=team_id, season=season,
    )

    logger.info("Fetching team roster: team=%s season=%s", team_id, season)
    cache_file = cache_path_for_url(url, _cache)
    raw_html = download_fbref_html(url, cache_path=cache_file)
    clean_html = _strip_comments(raw_html)

    # DESIGN: we need BOTH the table data AND the hyperlinks (which contain
    # player IDs). pandas.read_html() gives us the data but strips the links.
    # So we use BeautifulSoup to extract links, then join with the table data.

    soup = BeautifulSoup(clean_html, "lxml")

    # DESIGN: the roster table's id is "stats_standard_{league_id}" — the
    # numeric suffix is FBref's league id (e.g. 11 for Serie A) and differs
    # per league, so we can't hardcode it. Match by prefix instead. We also
    # keep "combined" (multi-competition team pages) and the bare name as
    # fallbacks, covering season-complete, in-progress, and combined views.
    stats_table = None
    for candidate in soup.find_all("table"):
        tid = candidate.get("id", "")
        if tid.startswith("stats_standard"):
            stats_table = candidate
            logger.info("Found roster table: id=%r", tid)
            break

    if stats_table is None:
        # Explicit fallbacks for older/variant page layouts
        for fallback_id in ("stats_standard_combined", "stats_standard"):
            stats_table = soup.find("table", {"id": fallback_id})
            if stats_table is not None:
                break

    if stats_table is None:
        # DESIGN: fail loudly with the ids we DID find, so a future FBref
        # layout change is diagnosable from the error alone.
        available = [t.get("id", "(none)") for t in soup.find_all("table")]
        raise ValueError(
            f"Could not find roster table for team={team_id} season={season}. "
            f"Expected an id starting with 'stats_standard'. "
            f"Tables present: {available}"
        )

    # Extract player links: <a href="/en/players/{id}/Player-Name">
    player_links = {}
    for link in stats_table.find_all("a", href=True):
        href = link["href"]
        if "/en/players/" in href:
            # DESIGN: extract the 8-char hex ID from the URL path.
            # URL format: /en/players/553fef74/Ademola-Lookman
            parts = href.split("/")
            player_idx = parts.index("players") + 1
            if player_idx < len(parts):
                pid = parts[player_idx]
                player_links[link.get_text(strip=True)] = {
                    "player_id": pid,
                    "player_url": FBREF_BASE + href,
                }

    # Now parse the table data
    df = _extract_table(str(stats_table), table_index=0)

    # DESIGN: the "player" column in the stats table contains the player name.
    # We join with the extracted links on player name to add player_id.
    player_col = None
    for candidate in ["player", "player_name"]:
        if candidate in df.columns:
            player_col = candidate
            break

    if player_col and player_links:
        df["player_id"] = df[player_col].map(
            lambda name: player_links.get(name, {}).get("player_id")
        )
        df["player_url"] = df[player_col].map(
            lambda name: player_links.get(name, {}).get("player_url")
        )

    # Filter out non-player rows (totals, averages)
    if player_col:
        df = df[df[player_col].notna() & (df[player_col] != "")]

    logger.info(
        "Parsed roster: %d players for team=%s season=%s",
        len(df), team_id, season,
    )

    return df


# ===================================================================
# PUBLIC API: Batch scrape — all players on a team
# ===================================================================

def fetch_all_player_match_logs(
    team_id: str,
    season: str,
    stat_types: list[str] | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch match logs for every player on a team, for one or more stat types.

    This is the main entry point for building the fatigue feature matrix.
    It first fetches the roster to discover player IDs, then fetches
    each player's match logs.

    Args:
        team_id: FBref team identifier.
        season: Season label in FBref format.
        stat_types: List of stat types to fetch per player.
            Defaults to ``["summary"]``.
        cache_dir: Override for the HTML cache directory.

    Returns:
        Dict mapping stat_type to a single DataFrame with ALL players'
        match logs concatenated, with a ``player_id`` column to identify
        each player.

    Note:
        For a 25-player squad with 1 stat type, this makes ~26 HTTP requests
        (1 roster + 25 players). At 5 seconds each, that's ~2 minutes.
        With caching, subsequent runs are instant.
    """
    if stat_types is None:
        stat_types = ["summary"]

    # Step 1: Get the roster to discover player IDs
    roster = fetch_team_roster(team_id, season, cache_dir=cache_dir)

    if "player_id" not in roster.columns or roster["player_id"].isna().all():
        raise ValueError(
            "Could not extract player IDs from roster. "
            "Cannot proceed with player match log scraping."
        )

    player_ids = roster[["player_id"]].dropna()["player_id"].unique()
    player_name_map = {}
    if "player" in roster.columns:
        player_name_map = dict(zip(roster["player_id"], roster["player"]))

    logger.info(
        "Scraping match logs for %d players × %d stat types = %d requests",
        len(player_ids), len(stat_types), len(player_ids) * len(stat_types),
    )

    results: dict[str, list[pd.DataFrame]] = {st: [] for st in stat_types}

    for pid in player_ids:
        pname = player_name_map.get(pid, pid)
        for stat_type in stat_types:
            try:
                df = fetch_player_match_log(
                    player_id=pid, season=season,
                    stat_type=stat_type, cache_dir=cache_dir,
                )
                # DESIGN: tag each row with the player's identity so the
                # concatenated DataFrame is still traceable.
                df["player_id"] = pid
                df["player_name"] = pname
                results[stat_type].append(df)

            except Exception as exc:
                # DESIGN: log and skip, don't crash the batch. Some players
                # (youth/reserve) may not have match log pages.
                logger.warning(
                    "Failed to fetch %s for player %s (%s): %s",
                    stat_type, pid, pname, exc,
                )

    # Concatenate per stat type
    output = {}
    for stat_type, dfs in results.items():
        if dfs:
            output[stat_type] = pd.concat(dfs, ignore_index=True)
            logger.info(
                "Collected %d total rows for stat_type='%s'",
                len(output[stat_type]), stat_type,
            )
        else:
            logger.warning("No data collected for stat_type='%s'", stat_type)
            output[stat_type] = pd.DataFrame()

    return output