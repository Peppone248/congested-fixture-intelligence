"""Diagnostica mirata: come è strutturato l'xG nello scorebox della pagina-partita."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from bs4 import BeautifulSoup
from src.scrapers.match_pages import discover_match_urls
from src.scrapers.fbref import _strip_comments
from src.scrapers.downloader import download_fbref_html
from src.utils.cache import cache_path_for_url
from src.utils.constants import FBREF_CACHE_DIR

with open(PROJECT_ROOT / "config" / "teams.yaml") as f:
    team = yaml.safe_load(f)["teams"][0]
fbref_id, season = team["fbref_id"], team["seasons"][0]

matches = discover_match_urls(fbref_id, season)
m = matches[0]  # Sassuolo-Atalanta
print(f"Partita: {m['date']} vs {m['opponent']} (Atalanta era in TRASFERTA)\n")

cache_file = cache_path_for_url(m["match_url"], FBREF_CACHE_DIR)
html = download_fbref_html(m["match_url"], cache_path=cache_file)
clean = _strip_comments(html)
soup = BeautifulSoup(clean, "lxml")

# Lo scorebox intero
scorebox = soup.find("div", class_="scorebox")
if scorebox:
    print("=== Tutti i div dentro lo scorebox con 'xg' nella classe ===")
    for d in scorebox.find_all("div"):
        cls = " ".join(d.get("class", []))
        if "xg" in cls.lower():
            print(f"  class={cls!r}  testo={d.get_text(strip=True)!r}")

    print("\n=== I link /squads/ nello scorebox (per capire l'ordine casa/trasferta) ===")
    for a in scorebox.find_all("a", href=True):
        if "/squads/" in a["href"]:
            print(f"  {a['href']}  ->  {a.get_text(strip=True)}")

# Cerca qualsiasi elemento con class contenente 'xg' in tutta la pagina
print("\n=== Tutti gli elementi con 'score_xg' o simili ===")
for d in soup.select("[class*='xg']"):
    cls = " ".join(d.get("class", []))
    print(f"  <{d.name} class={cls!r}> {d.get_text(strip=True)[:20]!r}")