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
