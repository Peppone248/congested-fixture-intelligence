"""Acquire detailed FBref stat tables (passing, possession, defense) + team xG.

Extends the base acquisition with the stat types needed for:
  - Family 2 (rolling performance): xAG, progressive passes/carries
  - Intensity weighting: pressures, tackles, carries distance (physical proxies)
  - Team-level xG per match (more reliable than player-level xG)

RUN ON YOUR MACHINE (venv active, FBref reachable via soccerdata):
    python scripts/acquire_detailed_stats.py

Outputs to data/raw/:
    {slug}_player_passing.csv
    {slug}_player_possession.csv
    {slug}_player_defense.csv
    {slug}_team_xg.csv          (team-level xG/xGA per match, if available)

The script also PRINTS a column inventory for each table, so we can see exactly
what FBref returns before building features on top of it.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.fbref import (  # noqa: E402
    fetch_all_player_match_logs,
    fetch_team_match_log,
    TEAM_MATCH_LOG_URL, FBREF_BASE, _strip_comments, _extract_table,
)
from src.scrapers.downloader import download_fbref_html  # noqa: E402
from src.utils.cache import cache_path_for_url  # noqa: E402
from src.utils.constants import FBREF_CACHE_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("acquire_detailed")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_PATH = PROJECT_ROOT / "config" / "teams.yaml"

# DESIGN: the detailed per-match stat types we need. summary is already pulled.
DETAILED_STAT_TYPES = ["passing", "possession", "defense"]


def load_target() -> dict:
    """Load the first team target from config/teams.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    team = cfg["teams"][0]
    return {"name": team["name"], "fbref_id": team["fbref_id"],
            "season": team["seasons"][0]}


def inventory(df: pd.DataFrame, label: str) -> None:
    """Print a compact column inventory + xG/xA detection for a table."""
    print(f"\n{'='*66}\n{label}: {len(df)} rows, {len(df.columns)} cols\n{'='*66}")
    print("Columns:", list(df.columns))
    xg_cols = [c for c in df.columns if re.search(r"xg|xa|expected", str(c), re.I)]
    print(f">>> xG/xA columns: {xg_cols if xg_cols else 'NONE'}")


def acquire_team_xg(fbref_id: str, season: str) -> pd.DataFrame | None:
    """Try to pull team-level xG/xGA per match from the team schedule page.

    The team's all_comps schedule table often carries xG and xGA columns —
    more reliably than the player-level summary. We already fetch this page
    for the calendar; here we re-parse it specifically for xG.
    """
    url = TEAM_MATCH_LOG_URL.format(base=FBREF_BASE, team_id=fbref_id, season=season)
    cache_file = cache_path_for_url(url, FBREF_CACHE_DIR)
    html = download_fbref_html(url, cache_path=cache_file)
    clean = _strip_comments(html)
    df = _extract_table(clean, table_id="matchlogs_for")
    xg_cols = [c for c in df.columns if re.search(r"xg|xga", str(c), re.I)]
    if not xg_cols:
        logger.warning("No team-level xG columns found on schedule page.")
        return None
    keep = ["date", "comp", "opponent", "venue"] + xg_cols
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t = load_target()
    name, fbref_id, season = t["name"], t["fbref_id"], t["season"]
    slug = f"{name.lower()}_{season.replace('-', '')}"
    logger.info("Acquiring detailed stats for %s %s", name, season)

    # --- Player detailed stats (passing, possession, defense) ---
    logs = fetch_all_player_match_logs(fbref_id, season, stat_types=DETAILED_STAT_TYPES)
    for stat_type, df in logs.items():
        if df.empty:
            logger.warning("No data for %s", stat_type)
            continue
        out = RAW_DIR / f"{slug}_player_{stat_type}.csv"
        df.to_csv(out, index=False)
        inventory(df, f"PLAYER {stat_type.upper()}")
        logger.info("Wrote %s", out)

    # --- Team-level xG ---
    team_xg = acquire_team_xg(fbref_id, season)
    if team_xg is not None:
        out = RAW_DIR / f"{slug}_team_xg.csv"
        team_xg.to_csv(out, index=False)
        inventory(team_xg, "TEAM xG")
        logger.info("Wrote %s", out)

    logger.info("Done. Inspect the column inventories above.")


if __name__ == "__main__":
    main()