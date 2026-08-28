"""Acquire the REAL, all-competitions Atalanta 2023-24 dataset from FBref.

RUN THIS ON YOUR OWN MACHINE — not in a sandbox. FBref must be reachable.

ARCHITECTURE (why it's built this way):

    Our custom scraper (src/scrapers/fbref.py) is the PRIMARY engine, because
    only the FBref "all competitions" pages give us the full congested
    calendar (Serie A + Europa League + Coppa Italia) and per-match player
    stats across every competition in a single table. soccerdata cannot do
    this — it only covers the Big-5 domestic leagues, so it has no cup or
    European data at all.

    soccerdata is used as an optional VALIDATOR for the Serie A slice: an
    independent second source for league minutes/xG. If our scraper and
    soccerdata agree on the Serie A numbers, we trust the parser. If they
    diverge, we've caught a bug before it reaches the model.

OUTPUTS (all under data/raw/):
    {team}_{season}_calendar.csv        <- full multi-competition calendar
    {team}_{season}_roster.csv          <- squad list with FBref player IDs
    {team}_{season}_player_matches.csv  <- per-player per-match stats, ALL comps
    {team}_{season}_validation.csv      <- soccerdata-vs-custom Serie A check (optional)

USAGE:
    python scripts/acquire_data.py                 # uses config/teams.yaml
    python scripts/acquire_data.py --no-validate   # skip the soccerdata cross-check

BEFORE YOU RUN — verify the FBref team id:
    Open the team's FBref page and copy the 8-char hex id from the URL:
        https://fbref.com/en/squads/<THIS_PART>/2023-2024/Atalanta-Stats
    Put it in config/teams.yaml as `fbref_id`. The default in this repo is a
    placeholder and MUST be confirmed, or every request 404s.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# DESIGN: make the package importable no matter where the script is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.fbref import (  # noqa: E402
    fetch_team_match_log,
    fetch_team_roster,
    fetch_all_player_match_logs,
)
from src.fixtures.parser import build_fixture_calendar  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("acquire_data")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_PATH = PROJECT_ROOT / "config" / "teams.yaml"


def load_target(config_path: Path) -> dict:
    """Load the first team target from config/teams.yaml.

    Args:
        config_path: Path to teams.yaml.

    Returns:
        Dict with keys: name, fbref_id, season (first configured season).

    Raises:
        SystemExit: If the config is missing or malformed.
    """
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        raise SystemExit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    teams = cfg.get("teams", [])
    if not teams:
        logger.error("No teams defined in %s", config_path)
        raise SystemExit(1)

    team = teams[0]
    seasons = team.get("seasons", [])
    if not seasons:
        logger.error("No seasons defined for team %s", team.get("name"))
        raise SystemExit(1)

    return {
        "name": team["name"],
        "fbref_id": team["fbref_id"],
        "season": seasons[0],  # e.g. "2023-2024"
    }


def acquire_calendar(fbref_id: str, season: str, team_name: str) -> pd.DataFrame:
    """Fetch and normalize the full multi-competition calendar.

    Args:
        fbref_id: FBref team hex id.
        season: FBref season label, e.g. "2023-2024".
        team_name: Canonical team name.

    Returns:
        Normalized calendar (from build_fixture_calendar).
    """
    logger.info("[1/3] Fetching full calendar (all competitions)...")
    raw = fetch_team_match_log(fbref_id, season)
    calendar = build_fixture_calendar(raw, team_name=team_name)
    logger.info(
        "  -> %d matches across %d competitions: %s",
        len(calendar), calendar["competition"].nunique(),
        sorted(calendar["competition"].unique()),
    )
    return calendar


def acquire_roster(fbref_id: str, season: str) -> pd.DataFrame:
    """Fetch the squad roster with FBref player IDs.

    Args:
        fbref_id: FBref team hex id.
        season: FBref season label.

    Returns:
        Roster DataFrame including a 'player_id' column.
    """
    logger.info("[2/3] Fetching roster with player IDs...")
    roster = fetch_team_roster(fbref_id, season)
    n_ids = roster["player_id"].notna().sum() if "player_id" in roster.columns else 0
    logger.info("  -> %d players (%d with resolvable IDs)", len(roster), n_ids)
    return roster


def acquire_player_matches(fbref_id: str, season: str) -> pd.DataFrame:
    """Fetch per-player, per-match summary stats across ALL competitions.

    Args:
        fbref_id: FBref team hex id.
        season: FBref season label.

    Returns:
        Long-format per-player per-match DataFrame (summary stat type).

    Note:
        The player match-log page lists every competition in one table, so a
        single 'summary' pull per player captures Serie A + Europa League +
        Coppa Italia together — satisfying the v1 "all competitions" goal.
    """
    logger.info("[3/3] Fetching per-player match logs (all competitions)...")
    logger.info("  This is the slow step: ~1 request per player at 5s each.")
    logs = fetch_all_player_match_logs(fbref_id, season, stat_types=["summary"])
    summary = logs.get("summary", pd.DataFrame())
    logger.info(
        "  -> %d player-match rows for %d players",
        len(summary),
        summary["player_id"].nunique() if "player_id" in summary.columns else 0,
    )
    return summary


def validate_against_soccerdata(
    player_matches: pd.DataFrame, season: str, team_name: str,
) -> pd.DataFrame | None:
    """Cross-check the Serie A slice against soccerdata (optional).

    Args:
        player_matches: Our scraper's per-player per-match output.
        season: FBref season label (converted to soccerdata's "2324" form).
        team_name: Team to filter soccerdata to.

    Returns:
        A comparison DataFrame (player, our_minutes, sd_minutes, diff) for the
        Serie A rows, or None if soccerdata is unavailable.

    Note:
        This never blocks acquisition — if soccerdata isn't installed or the
        API differs, we log a warning and skip. The custom scraper's data is
        authoritative; soccerdata is only a sanity check.
    """
    try:
        import soccerdata as sd
    except ImportError:
        logger.warning("soccerdata not installed — skipping validation.")
        return None

    # DESIGN: convert "2023-2024" -> "2324" for soccerdata's season code.
    sd_season = season[2:4] + season[7:9] if "-" in season else season

    try:
        reader = sd.FBref(leagues="ITA-Serie A", seasons=sd_season)
        sd_pm = reader.read_player_match_stats(stat_type="summary").reset_index()
    except Exception as exc:
        logger.warning("soccerdata read failed — skipping validation: %s", exc)
        return None

    # Filter soccerdata to the team and to Serie A only
    if "team" in sd_pm.columns:
        sd_pm = sd_pm[sd_pm["team"].astype(str).str.contains(team_name, case=False, na=False)]

    # DESIGN: compare total Serie A minutes per player as a coarse but telling
    # check. If our all-comps scraper's Serie-A-only minutes match soccerdata's,
    # the parser is trustworthy.
    our_sa = player_matches[
        player_matches["competition"].astype(str).str.contains("Serie A", case=False, na=False)
    ] if "competition" in player_matches.columns else player_matches

    def _min_col(df):
        for c in ("minutes", "min", "Min"):
            if c in df.columns:
                return c
        return None

    our_c, sd_c = _min_col(our_sa), _min_col(sd_pm)
    if our_c is None or sd_c is None:
        logger.warning("Could not locate minutes columns for validation.")
        return None

    our_tot = our_sa.groupby("player_name")[our_c].sum() if "player_name" in our_sa.columns else None
    sd_tot = sd_pm.groupby("player")[sd_c].sum() if "player" in sd_pm.columns else None
    if our_tot is None or sd_tot is None:
        return None

    comp = pd.DataFrame({"our_minutes": our_tot}).join(
        pd.DataFrame({"sd_minutes": sd_tot}), how="outer"
    )
    comp["diff"] = (comp["our_minutes"] - comp["sd_minutes"]).abs()
    n_agree = (comp["diff"].fillna(999) <= 5).sum()
    logger.info(
        "  Validation: %d/%d players agree within 5 minutes on Serie A totals",
        n_agree, len(comp),
    )
    return comp.reset_index()


def main() -> None:
    """Run the full acquisition and write outputs to data/raw/."""
    ap = argparse.ArgumentParser(description="Acquire real all-competitions FBref data.")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip the soccerdata Serie A cross-check.")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = load_target(Path(args.config))
    name, fbref_id, season = target["name"], target["fbref_id"], target["season"]
    slug = f"{name.lower()}_{season.replace('-', '')}"

    logger.info("=== Acquiring %s %s (FBref id=%s) ===", name, season, fbref_id)

    # 1. Calendar
    calendar = acquire_calendar(fbref_id, season, name)
    calendar.to_csv(RAW_DIR / f"{slug}_calendar.csv", index=False)

    # 2. Roster
    roster = acquire_roster(fbref_id, season)
    roster.to_csv(RAW_DIR / f"{slug}_roster.csv", index=False)

    # 3. Player match logs (all competitions)
    player_matches = acquire_player_matches(fbref_id, season)
    player_matches.to_csv(RAW_DIR / f"{slug}_player_matches.csv", index=False)

    # 4. Optional validation
    if not args.no_validate:
        comp = validate_against_soccerdata(player_matches, season, name)
        if comp is not None:
            comp.to_csv(RAW_DIR / f"{slug}_validation.csv", index=False)

    logger.info("=== Done. Real dataset written to %s ===", RAW_DIR)
    logger.info("Files:")
    for suffix in ("calendar", "roster", "player_matches", "validation"):
        p = RAW_DIR / f"{slug}_{suffix}.csv"
        if p.exists():
            logger.info("  %s", p.name)


if __name__ == "__main__":
    main()