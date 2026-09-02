"""Acquire full Serie A results (all teams) using OUR scraper, not soccerdata's
native read_schedule (which fails when read_leagues comes back empty).

The match/opponent weight needs each opponent's league position and recent form
AT THE TIME of each Atalanta fixture. That requires every Serie A result through
the season. We fetch the league's Scores & Fixtures page directly via our
Cloudflare-clearing downloader and parse it ourselves.

RUN ON YOUR MACHINE (venv active):
    python scripts/acquire_league_results.py

Output:
    data/raw/serie_a_2324_results.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.downloader import download_fbref_html  # noqa: E402
from src.scrapers.fbref import _strip_comments, _extract_table  # noqa: E402
from src.utils.cache import cache_path_for_url  # noqa: E402
from src.utils.constants import FBREF_CACHE_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("acquire_league")

RAW = PROJECT_ROOT / "data" / "raw"

# DESIGN: Serie A on FBref is competition id 11. The Scores & Fixtures page for
# a season lists every match of every team — exactly what we need. This is a
# normal FBref page our downloader can fetch through the headless browser.
SERIE_A_SCHEDULE_URL = (
    "https://fbref.com/en/comps/11/2023-2024/schedule/"
    "2023-2024-Serie-A-Scores-and-Fixtures"
)

# DESIGN: the fixtures table id on a competition schedule page follows the
# pattern sched_{season}_{comp_id}_1, e.g. sched_2023-2024_11_1. We match by
# prefix to be robust to the exact suffix.
SCHEDULE_TABLE_PREFIX = "sched_"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Serie A 2023-24 full schedule via headless browser...")
    cache_file = cache_path_for_url(SERIE_A_SCHEDULE_URL, FBREF_CACHE_DIR)
    html = download_fbref_html(SERIE_A_SCHEDULE_URL, cache_path=cache_file)
    clean = _strip_comments(html)

    # Find the schedule table by prefix
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(clean, "lxml")
    table = None
    for t in soup.find_all("table"):
        tid = t.get("id", "")
        if tid.startswith(SCHEDULE_TABLE_PREFIX):
            table = t
            logger.info("Found schedule table: id=%r", tid)
            break

    if table is None:
        available = [t.get("id", "(none)") for t in soup.find_all("table")]
        raise ValueError(
            f"Could not find schedule table (prefix '{SCHEDULE_TABLE_PREFIX}'). "
            f"Tables present: {available}"
        )

    df = _extract_table(str(table), table_index=0)

    # DESIGN: the schedule table has columns like: wk, day, date, home, score,
    # away, attendance, venue, referee. We normalize to what standings needs.
    logger.info("Raw columns: %s", list(df.columns))

    rename = {"wk": "week", "home": "home_team", "away": "away_team"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Parse the score column "2–1" (note: FBref uses an en-dash)
    if "score" in df.columns:
        scores = df["score"].astype(str).str.extract(r"(\d+)\D+(\d+)")
        df["home_score"] = pd.to_numeric(scores[0], errors="coerce")
        df["away_score"] = pd.to_numeric(scores[1], errors="coerce")

    # Keep only rows with two team names (drop separator/empty rows)
    df = df[df["home_team"].notna() & df["away_team"].notna()]
    df = df[df["home_team"].astype(str).str.strip() != ""]
    # DESIGN: FBref repeats the header row periodically inside the table body;
    # those rows have home_team == "Home". Drop them explicitly.
    df = df[df["home_team"] != "Home"]

    keep = [c for c in ["week", "date", "home_team", "away_team",
                        "home_score", "away_score"] if c in df.columns]
    out = df[keep].copy()

    out.to_csv(RAW / "serie_a_2324_results.csv", index=False)
    logger.info("Wrote %s (%d matches, %d played)",
                RAW / "serie_a_2324_results.csv", len(out),
                out["home_score"].notna().sum() if "home_score" in out.columns else 0)
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()