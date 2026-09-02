"""Crawl Atalanta's 56 match pages for advanced player stats + team xG.

RUN ON YOUR MACHINE (venv active). This is the slow crawl: ~56 match pages
through the headless browser. Per-match caching makes it RESUMABLE — if it
stops at match 40, re-running picks up from where it left off.

    python scripts/acquire_match_pages.py

Outputs to data/raw/:
    {slug}_match_possession.csv   Atalanta players' per-match possession stats
    {slug}_match_passing.csv      Atalanta players' per-match passing (incl xAG)
    {slug}_team_xg.csv            team xG/xGA per match
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.match_pages import discover_match_urls, scrape_match_page  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("acquire_match_pages")

RAW = PROJECT_ROOT / "data" / "raw"
CONFIG = PROJECT_ROOT / "config" / "teams.yaml"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with open(CONFIG) as f:
        team = yaml.safe_load(f)["teams"][0]
    name, fbref_id, season = team["name"], team["fbref_id"], team["seasons"][0]
    slug = f"{name.lower()}_{season.replace('-', '')}"

    # --- Phase 1: discover the 56 match URLs ---
    logger.info("Discovering match URLs for %s %s...", name, season)
    matches = discover_match_urls(fbref_id, season)
    logger.info("Found %d matches to crawl", len(matches))

    # --- Phase 2: crawl each match page (resumable via cache) ---
    poss_rows, pass_rows, xg_rows = [], [], []
    for i, m in enumerate(matches, 1):
        tag = f"[{i}/{len(matches)}] {m['date']} vs {m['opponent']}"
        try:
            data = scrape_match_page(m["match_url"], fbref_id)

            # DESIGN: tag every extracted row with match context so the
            # concatenated frames remain joinable back to the calendar.
            for key, bucket in (("possession", poss_rows), ("passing", pass_rows)):
                df = data.get(key)
                if df is not None and not df.empty:
                    df = df.copy()
                    df["match_date"] = m["date"]
                    df["competition"] = m["competition"]
                    df["opponent"] = m["opponent"]
                    bucket.append(df)

            xg_rows.append({
                "date": m["date"], "competition": m["competition"],
                "opponent": m["opponent"],
                "team_xg": data.get("team_xg"), "opp_xg": data.get("opp_xg"),
            })
            logger.info("%s  ok (team_xg=%s)", tag, data.get("team_xg"))

        except Exception as exc:
            # DESIGN: never let one bad page kill the crawl. Log and continue;
            # a re-run will retry this match (its page won't be cached on failure).
            logger.warning("%s  FAILED: %s", tag, exc)
            continue

    # --- Phase 3: write outputs ---
    if poss_rows:
        pd.concat(poss_rows, ignore_index=True).to_csv(
            RAW / f"{slug}_match_possession.csv", index=False)
        logger.info("Wrote %s_match_possession.csv", slug)
    if pass_rows:
        pd.concat(pass_rows, ignore_index=True).to_csv(
            RAW / f"{slug}_match_passing.csv", index=False)
        logger.info("Wrote %s_match_passing.csv", slug)
    if xg_rows:
        pd.DataFrame(xg_rows).to_csv(RAW / f"{slug}_team_xg.csv", index=False)
        logger.info("Wrote %s_team_xg.csv", slug)

    logger.info("Done. Inspect the outputs; some columns may need mapping.")


if __name__ == "__main__":
    main()