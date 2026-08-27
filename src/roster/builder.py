"""Assemble a canonical roster by reconciling FBref and Transfermarkt sources.

FBref supplies performance / minutes; Transfermarkt supplies valuations and
availability. A stable identity mapping between them is a prerequisite for
every downstream join.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_roster(
    fbref_players: pd.DataFrame,
    tm_players: pd.DataFrame,
    mapping_path: Path | None = None,
) -> pd.DataFrame:
    """Merge FBref and Transfermarkt player rows into a canonical roster.

    Args:
        fbref_players: Player rows from FBref (identity by ``fbref_id``).
        tm_players: Player rows from Transfermarkt (identity by ``tm_id``).
        mapping_path: Optional path to a persisted ``fbref_id -> tm_id`` mapping.

    Returns:
        Dataframe with one row per player carrying both source ids plus
        normalised position, age, market value, and contract length.

    Raises:
        ValueError: If the mapping resolves fewer than 90% of the FBref roster.
    """
    # DESIGN: persist the FBref-TM mapping under data/mappings/ so future runs
    # skip fuzzy name matching (accents, aliases, birth-year disambiguation).
    raise NotImplementedError


def normalise_position(raw_position: str, position_config: dict) -> str:
    """Map a source-specific position label to a canonical group.

    Args:
        raw_position: Position string as reported by a scraper.
        position_config: Parsed ``config/positions.yaml``.

    Returns:
        Canonical position group (e.g. ``"CB"``, ``"CM"``, ``"W"``, ``"CF"``).

    Raises:
        KeyError: If the raw position matches no alias in the config.
    """
    raise NotImplementedError
