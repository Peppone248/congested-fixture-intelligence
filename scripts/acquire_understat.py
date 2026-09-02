"""Acquire xG data from Understat (Serie A) — team-match and per-shot.

Understat provides xG for the top-5 leagues including Serie A, for free, via
plain HTTP (no Cloudflare, no headless browser). soccerdata wraps it. This
gives us the xG that FBref does not expose per-match for Serie A:
    - team xG / xGA per match  (read_team_match_stats)
    - per-shot xG              (read_shot_events)  [optional, for later]

RUN ON YOUR MACHINE (venv active):
    python scripts/acquire_understat_xg.py

Outputs to data/raw/:
    understat_seriea_2324_team_xg.csv     team xG per Serie A match (all teams)

NOTE — Europa League / Coppa Italia:
    Understat covers league play only. Atalanta's Serie A matches get real xG;
    European and cup matches do not (a documented v1 limitation, like the
    dynamic opponent strength).

NOTE — joining to our data:
    Understat match_ids differ from FBref's. We join Understat xG to the
    Atalanta calendar on (date, home/away teams), not on a shared id.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("acquire_understat")

RAW = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        import soccerdata as sd
    except ImportError:
        logger.error("soccerdata not installed. pip install soccerdata")
        raise SystemExit(1)

    logger.info("Reading Understat Serie A 2023-24 team match stats (xG)...")
    # DESIGN: Understat uses plain requests (BaseRequestsReader), so this is
    # fast and needs no browser. Season "2324" matches soccerdata's convention.
    understat = sd.Understat(leagues="ITA-Serie A", seasons="2324")

    team_stats = understat.read_team_match_stats().reset_index()
    logger.info("Got %d team-match rows. Columns: %s",
                len(team_stats), list(team_stats.columns))

    out = RAW / "understat_seriea_2324_team_xg.csv"
    team_stats.to_csv(out, index=False)
    logger.info("Wrote %s", out)

    # Show a sample so we can see the xG columns and team naming
    xg_cols = [c for c in team_stats.columns if "xg" in c.lower()]
    print(f"\nxG columns: {xg_cols}")
    print(team_stats.head(10).to_string(index=False))


if __name__ == "__main__":
    main()