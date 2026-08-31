"""Diagnostica: dove sono gli xG nelle pagine FBref già scaricate?

Lancia dalla cartella progetto col venv attivo:
    python diagnose_xg.py
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.utils.cache import cache_path_for_url
from src.utils.constants import FBREF_CACHE_DIR
from src.scrapers.fbref import (
    PLAYER_MATCH_LOG_URL, FBREF_BASE, _strip_comments, _extract_table,
)

# Un giocatore offensivo con xG sicuri: Éderson (a9202def) o cambia con Lookman
pid = "a9202def"
season = "2023-2024"

print("=" * 60)
print("DIAGNOSTICA xG — pagina summary match log")
print("=" * 60)

url = PLAYER_MATCH_LOG_URL.format(base=FBREF_BASE, player_id=pid,
                                  season=season, stat_type="summary")
cf = cache_path_for_url(url, FBREF_CACHE_DIR)
print(f"Cache path: {cf}")
print(f"Esiste: {cf.is_file()}")

if cf.is_file():
    raw = cf.read_text(encoding="utf-8", errors="replace")
    clean = _strip_comments(raw)

    # 1. Le colonne dopo il nostro parsing
    df = _extract_table(clean, table_id="matchlogs_all")
    print(f"\n--- Colonne dopo _extract_table: ---")
    print(list(df.columns))

    # 2. Cerca 'xg' nell'HTML grezzo (case-insensitive)
    import re
    xg_mentions = len(re.findall(r'xg', raw, re.IGNORECASE))
    print(f"\n--- Menzioni di 'xg' nell'HTML grezzo: {xg_mentions} ---")

    # 3. Prova a leggere la tabella con pandas SENZA il nostro post-processing
    from io import StringIO
    try:
        tables = pd.read_html(StringIO(clean), attrs={"id": "matchlogs_all"})
        if tables:
            raw_df = tables[0]
            print(f"\n--- Colonne grezze da pandas (MultiIndex): ---")
            print(list(raw_df.columns))
    except Exception as e:
        print(f"pandas.read_html fallito: {e}")

# Verifica anche se esistono altri stat_type scaricati
print("\n" + "=" * 60)
print("Altri stat_type in cache per questo giocatore?")
for st in ["summary", "passing", "defense", "possession", "gca", "misc"]:
    u = PLAYER_MATCH_LOG_URL.format(base=FBREF_BASE, player_id=pid, season=season, stat_type=st)
    c = cache_path_for_url(u, FBREF_CACHE_DIR)
    print(f"  {st:12s}: {'IN CACHE' if c.is_file() else 'non scaricato'}")