"""Descriptive statistics + visualizations for Family-1 load features.

Produces a readable, self-contained report on the workload feature set:
distributions, per-player load profiles, and the accumulated-fatigue trend.
Figures are written to outputs/figures/.

USAGE (venv active, from project root):
    python scripts/describe_family1.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.load_features import compute_load_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("describe_family1")

RAW = PROJECT_ROOT / "data" / "raw"
FIG = PROJECT_ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CLUB_COMPS = ["Serie A", "Europa Lg", "Coppa Italia"]

# DESIGN: a consistent, readable style across all figures — no default
# matplotlib grey. Colours chosen for print legibility.
PALETTE = {
    "primary": "#1f6feb", "accent": "#e67e22",
    "rest": "#2ecc71", "load": "#e74c3c", "grid": "#dddddd",
}


def _load_and_prepare(slug: str = "atalanta_20232024") -> pd.DataFrame:
    """Load raw CSVs, clean, filter to Atalanta, compute Family-1 features."""
    cal = pd.read_csv(RAW / f"{slug}_calendar.csv")
    pm = pd.read_csv(RAW / f"{slug}_player_matches.csv")

    pm["team"] = pm["team"].apply(
        lambda s: re.sub(r"^[a-z]{2,3}\s+", "", str(s)).strip()
    )
    pm = pm[
        pm["competition"].isin(CLUB_COMPS)
        & pm["team"].str.contains("Atalanta", case=False, na=False)
    ].copy()
    if "minutes" not in pm.columns and "min" in pm.columns:
        pm = pm.rename(columns={"min": "minutes"})

    cal_club = cal[cal["competition"].isin(CLUB_COMPS)].copy()
    return compute_load_features(pm, cal_club)


def _print_descriptive_table(lf: pd.DataFrame) -> None:
    """Print a formatted descriptive-statistics table to the console."""
    feats = [
        "minutes_7d", "minutes_14d", "starts_7d", "matches_since_rest",
        "weighted_load_since_rest", "avg_weekly_minutes", "fatigue_trend",
    ]
    desc = lf[feats].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).round(2)
    print("\n" + "=" * 70)
    print("FAMILY-1 LOAD FEATURES — DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(desc.to_string())
    print(f"\nRows: {len(lf)}  |  Players: {lf['player_id'].nunique()}  |  "
          f"Fixtures/player: {len(lf)//max(lf['player_id'].nunique(),1)}")


def fig_distributions(lf: pd.DataFrame) -> Path:
    """Histogram grid of the core load features."""
    feats = [
        ("minutes_7d", "Minutes in last 7 days"),
        ("minutes_14d", "Minutes in last 14 days"),
        ("matches_since_rest", "Matches since last full rest"),
        ("weighted_load_since_rest", "Weighted load since rest"),
        ("avg_weekly_minutes", "Avg minutes (last 4 fixtures)"),
        ("fatigue_trend", "Accumulated-fatigue trend (EWMA)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title) in zip(axes.flat, feats):
        data = lf[col].dropna()
        ax.hist(data, bins=25, color=PALETTE["primary"], edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color=PALETTE["accent"], ls="--", lw=2,
                   label=f"mean={data.mean():.1f}")
        ax.axvline(data.median(), color=PALETTE["load"], ls=":", lw=2,
                   label=f"median={data.median():.1f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle("Family-1 Load Features — Distributions (Atalanta 2023-24)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIG / "family1_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_player_load_ranking(lf: pd.DataFrame, top_n: int = 15) -> Path:
    """Horizontal bar chart: total club minutes per player (the workhorses)."""
    played = lf[lf["minutes"] > 0]
    totals = (
        played.groupby("player_name")["minutes"].sum()
        .sort_values(ascending=False).head(top_n).iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(totals.index, totals.values, color=PALETTE["primary"], edgecolor="white")
    for i, v in enumerate(totals.values):
        ax.text(v + 20, i, f"{int(v)}", va="center", fontsize=9)
    ax.set_title(f"Top {top_n} players by total club minutes — Atalanta 2023-24",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Total minutes (Serie A + Europa Lg + Coppa Italia)")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    out = FIG / "family1_player_ranking.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_fatigue_trajectories(lf: pd.DataFrame, player_ids: list[str] | None = None) -> Path:
    """Line chart of the accumulated-fatigue trend for a few key players."""
    if player_ids is None:
        # DESIGN: pick the 4 players with the most club minutes — the ones whose
        # fatigue trajectory matters most for rotation decisions.
        top = (
            lf[lf["minutes"] > 0].groupby("player_id")["minutes"].sum()
            .sort_values(ascending=False).head(4).index.tolist()
        )
        player_ids = top

    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ["#1f6feb", "#e67e22", "#e74c3c", "#2ecc71", "#9b59b6"]
    for pid, c in zip(player_ids, colors):
        g = lf[lf["player_id"] == pid].sort_values("date")
        if g.empty:
            continue
        name = g["player_name"].iloc[0]
        ax.plot(g["date"], g["fatigue_trend"], marker="o", ms=3, lw=1.6,
                color=c, label=name)
        # Mark full rests
        rests = g[g["minutes"] == 0]
        ax.scatter(rests["date"], rests["fatigue_trend"], marker="v", s=80,
                   color=c, edgecolors="black", zorder=5)

    ax.set_title("Accumulated-fatigue trend (▼ = full rest) — top-minutes players",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Fatigue trend (EWMA of per-match load)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = FIG / "family1_fatigue_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_rest_weight_illustration(lf: pd.DataFrame) -> Path:
    """Scatter showing how minutes map to the continuous rest weight."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sample = lf.sample(min(600, len(lf)), random_state=42)
    ax.scatter(sample["minutes"], sample["rest_weight"],
               alpha=0.4, color=PALETTE["primary"], s=25)
    ax.set_title("Weighted rest: minutes → rest weight\n(0 min = full rest, 90+ = full load)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Minutes played")
    ax.set_ylabel("Rest weight")
    ax.axhline(0.5, color=PALETTE["accent"], ls="--", alpha=0.6, label="half rest (45 min)")
    ax.axvline(45, color=PALETTE["accent"], ls="--", alpha=0.6)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = FIG / "family1_rest_weight.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    lf = _load_and_prepare()
    _print_descriptive_table(lf)

    outputs = [
        fig_distributions(lf),
        fig_player_load_ranking(lf),
        fig_fatigue_trajectories(lf),
        fig_rest_weight_illustration(lf),
    ]
    print("\nFigures written:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
