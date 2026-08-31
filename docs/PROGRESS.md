# Progress Log

Chronological log of implementation work. Each entry names the commit(s)
that landed the change and the intent behind them.

## 2026-08-27 — Project scaffolding
- `a4ec71c` — Full package layout, config YAMLs, module stubs with docstrings,
  type hints, and `# DESIGN:` markers.

## 2026-08-27 — On-disk cache for scraped HTML
Building `src/utils/cache.py` so both scrapers (FBref, Transfermarkt) share
one caching layer instead of each rolling its own.

Slicing:
1. Path helpers (`cache_path_for_url`, `is_cached`) — the deterministic
   md5 → 2-char fanout → `.html` layout, and a cheap existence check.
2. HTTP fetch with polite backoff (`get_cached_or_fetch`) + maintenance
   (`clear_cache`).

Design choices worth remembering:
- **Fanout by first 2 hex chars of the hash** keeps any single directory
  under ~256 sibling files even after thousands of pages — matters on
  ext4 where directory reads slow down at high fanout.
- **The `cache_dir` argument names the leaf** (e.g. `data/raw/fbref/`),
  so the caller decides which source's cache to write into. The 2-char
  fanout is created underneath it.
- **Delay is per-fetch, not per-call**: cached hits do not sleep, only
  live HTTP requests do. This matters because notebooks re-run cells
  and would otherwise pay the delay on every cache hit.
- **Retries are exponential and bounded** (3 attempts, 2s → 4s → 8s).
  429/5xx retry; 403 does not — a 403 usually means the site blocked
  the UA and retrying will just get another 403.
- **Logging instead of print**: notebooks and Streamlit both capture the
  `logging` output; `print` gets lost in the Streamlit process.

## 2026-08-31 — Real data acquired; Family-1 load features
Cloudflare blocked the custom scraper (403); routed downloads through
soccerdata's headless browser (`src/scrapers/downloader.py`) while keeping our
parser for the all-competitions pages. Acquired real Atalanta 2023-24: 56-match
calendar, 42-player roster, 1,497 player-match rows.

Congestion tiers recalibrated to be recovery-hours driven (count-driven
over-classified >50% of the season as HEAVY). On real data: 61% NORMAL / 14%
MODERATE / 14% HEAVY / 11% EXTREME, EXTREME localising to the Apr–May Europa
run-in.

Family-1 load features (`src/fatigue/load_features.py`):
- Calendar cross-join makes rests visible (2,184 player-fixture rows).
- Weighted rest (continuous, from minutes) replaces the "<45 min = rest" rule.
- Match-based counters replace day-based (avoids temporal saturation).
- Added `fatigue_trend` (EWMA of per-match load) and `avg_weekly_minutes`.
- Anti-leakage verified: every player's first fixture has zero load.

Descriptive-stats script + 4 figures: `scripts/describe_family1.py`.
Documented all decisions and 4 dataset edge cases in `docs/METHODOLOGY.md`.
