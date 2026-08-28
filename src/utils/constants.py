"""Project-wide constants and file-system anchors."""

from __future__ import annotations

from pathlib import Path


# DESIGN: expose the project root as a single anchor so scripts, notebooks,
# and the Streamlit app do not each carry their own guess about layout.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR: Path = DATA_DIR / "synthetic"
MAPPINGS_DIR: Path = DATA_DIR / "mappings"
# DESIGN: per-source cache directories live under data/raw/ so each source's
# cached HTML is isolated. The cache module creates subdirectories (2-char
# hex fanout) underneath these paths on first write.
FBREF_CACHE_DIR: Path = RAW_DATA_DIR / "fbref"
TRANSFERMARKT_CACHE_DIR: Path = RAW_DATA_DIR / "transfermarkt"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

# DESIGN: canonical position groups match the keys in config/positions.yaml
# so a mismatch surfaces at import time, not deep in the depth chart.
CANONICAL_POSITIONS: tuple[str, ...] = ("GK", "CB", "FB", "DM", "CM", "AM", "W", "CF")

# DESIGN: a regulation match is 90 minutes; extra-time appearances are
# treated as 90+ in features so cup knockouts don't undercount load.
FULL_MATCH_MINUTES: int = 90