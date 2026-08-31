# Methodology

This document records the methodological decisions behind the Congested
Fixture Intelligence framework: what we measure, why we measure it that way,
and which edge cases we handle (or explicitly defer). It is the companion to
`PROGRESS.md` (which logs *what* was built when); this file explains *why*.

Every non-obvious modelling choice should be documented here so that a
reviewer — or Atalanta's data staff — can audit the reasoning, and so that
future work doesn't silently re-decide a settled question.

---

## 1. Data sources

| Source | Role | Access |
|--------|------|--------|
| FBref (via soccerdata's headless-browser downloader) | Match calendar, roster, per-player per-match stats | Public, scraped |
| Transfermarkt | Injury history, market value (planned, not yet integrated) | Public, scraped |

### 1.1 Why soccerdata downloads but our parser parses

FBref sits behind Cloudflare. A plain `requests.get()` — regardless of how
complete the headers are — receives a `403`, because Cloudflare requires a
real browser that executes its JavaScript challenge. `soccerdata`'s FBref
reader is a `BaseSeleniumReader` (headless browser via seleniumbase) that
clears Cloudflare, and it exposes a low-level `.get(url, filepath)` that
downloads *any* FBref URL to disk.

We therefore split responsibilities: **soccerdata is the download engine**
(clears Cloudflare, caches HTML), **our `fbref.py` is the parse engine** (it
handles the all-competitions pages that soccerdata does not parse at match
granularity). This gives us soccerdata's robustness *and* full multi-
competition coverage. See `src/scrapers/downloader.py`.

### 1.2 FBref quirks handled in parsing

- **Comment-wrapped tables.** Many FBref stat tables are hidden inside HTML
  comments (`<!-- <table>...</table> -->`). We strip the comment markers
  before parsing (`_strip_comments`).
- **Multi-level headers.** Stat tables use two header rows (a group row and a
  column row); we flatten them to snake_case (`_flatten_columns`).
- **League-suffixed table ids.** The roster table id is
  `stats_standard_{league_id}` (e.g. `stats_standard_11` for Serie A). The
  numeric suffix differs per league, so we match by prefix, not exact id.
- **Country-code prefixes.** In European competitions FBref prefixes foreign
  team names — and even the club's own name — with a 2–3 letter country code
  (`pl Raków`, `it Atalanta`). We strip a leading lowercase short token from
  team and opponent names.
- **Competition labels.** FBref writes `Europa Lg` and `Champions Lg`, not the
  full UEFA names. European-competition detection matches these actual labels.

---

## 2. The fixture calendar and congestion index (Module M2)

### 2.1 Team vs player congestion

We compute congestion at two levels, with deliberately different window shapes:

- **Team-level congestion** describes the *scheduling pressure* around a match:
  matches in a 7/14/21-day window, recovery hours between matches, competition
  switches. It uses windows **centred** on each match, because scheduling
  density is symmetric — a match sandwiched between two others is congested
  regardless of which side the neighbours fall on.

- **Player-level congestion** (Family 1, below) describes the *workload a
  specific player carries into* a match. It uses **backward-looking** windows
  that exclude the current match, because it measures fatigue already
  accumulated — and must not leak information from the match being predicted.

The gap between the two is the point: a team can face EXTREME scheduling
congestion while an individual player experiences only moderate workload,
because the manager rotated him. That gap is exactly what creates the
survivorship bias the fatigue model (M3) must correct for.

### 2.2 Congestion tiers are recovery-driven, not count-driven

**Decision.** Tiers are driven primarily by recovery hours, not raw match
count.

**Why.** An early version classified any "2 matches in 7 days" as HEAVY. In a
multi-competition season that is the *normal* rhythm (Thursday Europa League +
Sunday league), so >50% of the season came out HEAVY — an indicator that fires
almost always carries no information. Recalibrating on recovery hours (the
physiologically grounded driver, per the post-match recovery literature)
restored discriminative power. On real Atalanta 2023-24 data the tiers now come
out ≈61% NORMAL / 14% MODERATE / 14% HEAVY / 11% EXTREME, and the EXTREME weeks
correctly localise to the April–May Europa-League run-in.

Tier logic (see `src/fixtures/congestion.py`):
- **EXTREME** — 3+ matches in 7 days (a genuine triple week).
- **HEAVY** — short recovery (<72h) into or out of the match.
- **MODERATE** — 2 matches in 7 days with adequate recovery (72–96h).
- **NORMAL** — isolated match, or 96h+ recovery on both sides.

### 2.3 Data-quality guard: implausibly short recovery

Recovery below ~40h between two competitive matches is physically implausible
(even a Sat 20:45 → Mon 18:45 turnaround is ~46h). We warn loudly on such
values rather than modelling on them, because they usually indicate duplicate
or same-day fixtures — a data problem, not a real schedule.

---

## 3. Family 1 — windowed load features (Module M3 input)

Family 1 turns the raw "who played how many minutes" table into a quantitative
description of the load each player carries into each fixture. It is the raw
material for both the fatigue model and the propensity (survivorship-bias)
model. Implementation: `src/fatigue/load_features.py`.

### 3.1 Calendar cross-join: making rests visible

**Decision.** We expand each player's FBref match log onto the *full Atalanta
club calendar*. Any club fixture absent from a player's log is treated as a
0-minute appearance.

**Why.** A player's match log only lists matches he played. Without the cross-
join, the moments he *rested* are simply invisible, and any rest-based feature
is uncomputable. The cross-join is what lets us see rests at all. On real data
this expands the dataset from 1,266 appearance rows to 2,184 player-fixture
rows (39 players × 56 club fixtures, minus mid-season squad changes).

### 3.2 Weighted rest, not binary

**Decision.** Rest is a continuous weight derived from minutes played, not a
0/1 flag. `rest_weight = clip(1 − minutes/90, 0, 1)`: 0 minutes → full rest
(1.0); 90+ minutes → full load (0.0); a 30-minute cameo → ~0.67.

**Why.** Playing under 45 minutes is **not** a full rest — a substitute still
accumulates fatigue. The earlier crude rule ("<45 min = rest") mislabelled
cameos as rest. A continuous weight captures partial turnover correctly: a
45-minute appearance is half a rest, not a full one.

### 3.3 Rest is match-based, not day-based

**Decision.** Rest and fatigue counters are expressed in **matches**, not
calendar days.

**Why.** We have no training-session data and cannot infer day-to-day load, so
counting calendar days is misleading. An earlier day-based `days_since_rest`
saturated uninformatively — a season-long ever-starter accrued 200+ "days since
rest", a number that reflected *usage pattern*, not physiological fatigue.
Counting discrete match events (`matches_since_rest`) avoids that saturation and
is directly interpretable ("he has started the last N fixtures without a rest").

### 3.4 The Family-1 feature set

All windows are backward-looking and exclude the current fixture (no leakage;
verified: every player's first fixture has all-zero load features).

| Feature | Definition |
|---------|-----------|
| `minutes_7d`, `minutes_14d` | Minutes played in the trailing 7 / 14 days |
| `starts_7d` | Starts (≥45 min) in the trailing 7 days |
| `matches_since_rest` | Consecutive fixtures since the last full (0-minute) rest; resets on a full rest |
| `weighted_load_since_rest` | Sum of per-match load `(1 − rest_weight)` since the last full rest — partial appearances count partially |
| `avg_weekly_minutes` | Rolling mean of minutes over the last 4 fixtures (usage rhythm), shifted to exclude the current fixture |
| `fatigue_trend` | EWMA of per-match load `(1 − rest_weight)`, half-life 4 matches — a smooth accumulated-fatigue trajectory, recent matches weighted more |

**Why an EWMA fatigue trend.** A point-in-time window ("minutes in last 7
days") captures the acute spike but not the *trajectory*. An exponentially-
weighted moving average of per-match load gives a smooth curve where recent
matches weigh more than old ones, so we can see fatigue building across a
congested block and decaying after a rest. On real data it is the best-
distributed Family-1 feature (spread across [0,1] rather than spiking at zero),
which makes it a strong candidate predictor.

---

## 4. Scope and filtering decisions

### 4.1 v1 scope: Atalanta club matches only (Approach 1)

**Decision.** For v1 the modelling dataset is Atalanta matches in Serie A,
Europa League, and Coppa Italia only. Filtering requires **both** competition ∈
{Serie A, Europa Lg, Coppa Italia} **and** team = Atalanta (after country-prefix
normalisation).

**Why the team filter matters.** FBref's per-player match log includes *every*
match a player played that season — including matches for other clubs after a
transfer. Filtering by competition alone leaves in, e.g., Zapata's Torino
matches. The team filter removes 102 such rows on real data.

### 4.2 National-team load: deferred to post-v1 (Approach 2)

**Decision.** National-team and other non-Atalanta matches (Euro qualifiers,
AFCON, Copa América, friendlies, MLS, etc. — ~9% of raw player-match rows) are
excluded from v1.

**Why, and the plan.** These minutes are *real* load — a player back from AFCON
is genuinely fatigued — but they are not part of the Atalanta calendar, and
their summary data is heterogeneous. Post-v1 we will re-introduce them
(Approach 2) as accumulated load inside each player's time-windows *without*
modelling performance in those matches. This is a cleanly isolated increment
whose effect we can measure, consistent with the "one feature at a time"
methodology.

---

## 5. Edge cases in the dataset

FBref per-player match logs, cross-joined against the club calendar, surface
several edge cases. How each is handled in v1, and the planned refinement:

### 5.1 Injured players with prolonged "rest"

**Situation.** An injured player disappears from the match log for weeks. After
the calendar cross-join, those fixtures look like 0-minute rests.

**v1 handling.** Treated generically as "did not play" — i.e. as rest.

**Planned refinement.** Cross-reference Transfermarkt injury history to mark
injury absences distinctly, and exclude them from the *rest* interpretation.
This matters because a player **returning from injury** is **more** fatigued /
under-conditioned than a rested healthy player — the opposite of what a plain
rest weight implies. Injury-return should eventually enter the model as its own
signal, not be smoothed into "well rested".

### 5.2 Players who left Atalanta in January

**Situation.** A player transferred out mid-season (e.g. Zapata → Torino,
Muriel → Orlando) has later-season matches for another club in his FBref log.

**v1 handling.** Removed by the `team = Atalanta` filter (§4.1). Their
Atalanta-period matches remain; their post-transfer matches are dropped.

**Note.** After they leave, the calendar cross-join would otherwise generate
spurious 0-minute "rests" for every remaining Atalanta fixture. Because we
compute per player over the fixtures present in their (filtered) data, a
departed player's trajectory simply ends at his last Atalanta match rather than
accruing phantom rests. A future refinement may bound each player's active
window explicitly by first/last Atalanta appearance.

### 5.3 Suspended players (one or more matches)

**Situation.** A player serving a suspension is absent for a specific fixture,
which the cross-join records as a 0-minute rest.

**v1 handling.** Treated as rest. Physiologically a suspension *is* a rest (the
player doesn't play), so for the fatigue interpretation this is acceptable.

**Planned refinement.** For the propensity / availability model (M6), a
suspension is *unavailability*, not a rotation choice — the manager didn't
*choose* to rest the player, he *couldn't* pick him. Distinguishing suspension
from voluntary rest matters for modelling the selection mechanism (who the
manager chooses to rest), which is the core of the survivorship-bias
correction. Suspension dates are derivable from accumulated yellows / red cards
in the match log, or from Transfermarkt.

### 5.4 Players signed in January (no pre-January Atalanta matches)

**Situation.** A winter signing (e.g. Hien arriving from Verona) has no Atalanta
fixtures before January. The calendar cross-join would generate 0-minute
"rests" for every Atalanta match in Aug–Dec, badly distorting his load features.

**v1 handling.** Because features are computed per player over his own filtered
appearances, his trajectory begins at his first Atalanta match; the pre-arrival
fixtures do not generate phantom rests for him.

**Planned refinement.** Make the active window explicit: bound each player's
fixture expansion between his first and last Atalanta appearance, so the cross-
join never invents rests outside a player's time at the club. This generalises
the handling for both winter arrivals (§5.4) and departures (§5.2).

---

## 6. Known limitations (v1)

- **No training-session load.** We infer load from match minutes only. This is
  the fundamental limitation of a public-data approach; a club-internal version
  would add GPS/wearable session load (as in the companion injury-risk
  framework).
- **xG not yet available per match.** FBref's `summary` per-player match log
  does not carry Expected columns for all players. xG/xA acquisition (via the
  `passing`/`possession` stat types, or team-match xG) is pending, and Family-2
  (rolling performance) features that need xG are blocked on it.
- **Injuries, suspensions, and squad changes** are handled generically in v1
  (§5); the Transfermarkt cross-reference that distinguishes them is planned.
- **Single team-season.** The framework is validated on Atalanta 2023-24. Multi-
  team, multi-season validation is future work.
