# Games integration: method report (revision r2)

**No labeled evaluation and no supervised training were done.** No reference matches, labels, judge or
teacher were available or created. Every number below is a structural, label-free diagnostic computed
from the saved artifacts (`work/diagnostics.jsonl`). None of them is an accuracy, precision, recall, PC or F1 figure.

Pipeline: `work/s1_schema.py` … `work/s7_export_diag.py`, shared helpers in `work/norm.py`, and the
entrypoint `work/rebuild.sh`. The build is deterministic and has no random sampling. I ran `rebuild.sh` in a fresh copy of the
workspace with `submission/` and `work/state/` removed. It reproduced membership, correspondences,
fused, candidates and sm_mapping exactly (compared as sorted rows).

## Sources (profiled)
- **dbpedia** (46,580 rows) is a flattened **cross-product** of multi-valued DBpedia properties. One article
  becomes many rows: for example, "Oddworld: Stranger's Wrath" has 537 rows covering 12 systems × 7 dates × 7 studios × 3 genres.
  Studio values are noisy (some contain platform names). Titles carry Wikipedia disambiguation such as "(1988 video game)".
  The data has no scores, ESRB or publisher.
- **metacritic** (20,494 rows) and **sales** (7,877 rows) have about one row per game/platform. Their critic scores
  are nearly identical when titles match (both derive from Metacritic). Sales years are often 1 year off.

## 1. Schema matching (`sm_mapping.csv`)
I mapped columns by hand, based on meaning and example values. For example, dbpedia `system`, metacritic `console`
and sales `hw` all map to `platform`. `franchise` maps to `series`, and `dist` (sales only) maps to `publisher`.
Scores are 1.0 except the noisy dbpedia `studio` and `genre` columns (0.9). Two kinds of column are left unmapped: the id columns,
which are kept as provenance, and `sales.units_sold_mm`, which has no target attribute. The target `id` is filled with the fused `_id`.

## 2. Normalization (`work/norm.py`, `s2_normalize.py`; plan in `work/taxonomy_plan.json`)
- **Platform:** regex rules map every source spelling to a comparison key used for blocking and matching.
  Examples: PS4 / PlayStation 4 → ps4; Microsoft Windows / Windows / PC → pc; XOne / Xbox One → xone; iPhone / iPad / IOS → ios.
  - Metacritic and sales mapped 100% of values.
  - dbpedia mapped 84%. The rest are long-tail systems such as Amiga or MSX. These keep a folded raw key, so they still group
    within dbpedia but cannot match other sources, since those sources do not list such systems.
  - **Output spelling:** metacritic, sales and dbpedia already share one vocabulary ("PS4", "Switch", "PC", "Xbox 360"),
    so the fused table keeps the source-supported spelling rather than rewriting it to taxonomy names. The taxonomy is
    non-exhaustive, and the schema example itself uses "PC". The sales codes XOne, WiiU and DC become Xbox One, Wii U and Dreamcast.
  - **Uncertainty:** if the evaluator expects taxonomy names such as "PlayStation 4", platform values will not match.
- **Titles:** Wikipedia-style trailing disambiguation is stripped symmetrically in all sources, and any year in it is kept as a hint.
  The comparison key applies accent folding, `&`→and, removal of punctuation and articles, Roman numerals → digits, and versus→vs.
- **ESRB:** alias K-A→E (from the schema). The vocabulary is exhaustive, and all source values were valid after the alias.
- **Dates:** every source gives YYYY-01-01, so year parsing succeeded 100% on non-null values.

## 3. Blocking (`s3_blocking.py`; complete set in `blocking/candidates.csv`)
- **Matching units:** metacritic and sales rows are one unit each. dbpedia rows are grouped by identical (raw title
  including disambiguation, platform key, year), which gives 36,794 units averaging 1.27 rows.
- **Candidates:** for each source pair, only units with the **same platform key** are compared (the entity is a game on one platform).
  The candidate set is the union of:
  - exact title-key matches;
  - top-8 neighbours in both directions by token_set_ratio;
  - top-8 neighbours in both directions by plain ratio, with similarity ≥ 60.
- **Counts:** 184k unit pairs, expanding to **219,768 record pairs**. The reduction ratio against the full cross-source product is 0.99985.
  The transitive closure of matches added 0 pairs outside the candidates, so all correspondences are in the candidate set.
  Records without a platform (0.9% of dbpedia) cannot be blocked.

## 4. Entity matching (`s4_matching.py`, deterministic rules)
Throughout, `numok` means the digit sequences in the two titles are equal, which protects sequels and yearly editions.
The year gap uses the dbpedia unit's year plus its disambiguation year.

**metacritic–sales rules:**
- exact title key, with year gap ≤ 2 or critic score within ±1 (6,999 pairs);
- spelling variants (spacing, hyphenation, token order, one typo in a long token) with numok and year gap ≤ 1;
- brand prefix or edition suffix from a whitelist, plus critic within ±1 and year gap ≤ 1;
- token_set ≥ 90, plus equal critic score and user score, plus year gap ≤ 1.

**dbpedia–metacritic and dbpedia–sales rules** (no scores are available):
- exact title key with year gap ≤ 2;
- exact title key when there is no disambiguation year (dbpedia dates are a noisy cross-product);
- spelling variants, brand prefix or edition suffix with numok and year gap ≤ 1.

**Brand-prefix and edition-suffix whitelists:**
- Brand prefixes: Disney, DreamWorks, Sid Meier's, Lara Croft, Marvel's, WWE, Nickelodeon, and similar.
- Edition suffixes: "The Video Game", Remastered, HD, Definitive, Deluxe, Complete Edition, and similar.

**How the rules were derived.** I derived them by inspecting near-miss samples. Rules I tried and rejected:
- a generic token_sort ≥ 90 rule, which merged ESPN NFL 2K5 with NHL 2K5 and "Explorers of Time" with "Explorers of Sky";
- a single-token typo rule, which merged Hades with Haven and Catan with Conan;
- a nospace ratio ≥ 95 rule, which merged "Mugen Souls" with "Mugen Souls Z" and "Skyrim" with "Skyrim VR";
- a token_set + critic ±1 rule, which merged "Borderlands" with its DLC.

DLC and expansions (Xtreme Legends, Empires, and so on) are deliberately **not** matched.

**Accepted pairs by source pair:**

| Source pair | Accepted unit pairs |
|---|---|
| dbpedia–metacritic | 9,025 |
| dbpedia–sales | 5,087 |
| metacritic–sales | 7,096 |

## 5. Clustering (`s5_cluster.py`)
Accepted edges are processed greedily by descending score, with union-find. A merge is **refused if the cluster would contain two
units from the same source**, because each source holds at most one unit per entity. This refused 3,056 edges. Most of them were
alternative dbpedia date-units of the same title, where the closest year wins, or sequel and variant competitors. The refusals are listed in
`work/state/rejected_edges.csv`.

In revision r1, dbpedia units were whole (title, platform) groups. That produced clusters of up to 108 rows, and 7.2% of records sat in
clusters larger than 9, so `self_check` failed on chaining. r2 splits dbpedia units by release year as well. Rows of one title and platform
whose dates disagree are kept as separate units, and only the date-consistent unit joins a matched cluster.
This is a known trade-off: a game/platform that dbpedia lists with several dates may now be split across several fused rows, and
some dbpedia rows that describe the same game/platform are left out of the matched cluster.

Unmatched records are kept as singletons. Correspondences are all cross-source record pairs inside each final cluster (20,785).

## 6. Fusion (`s6_fusion.py`)
Each dbpedia unit contributes one vote per attribute: the mode of its rows. Votes are counted once per source on normalized values.
Ties go to source priority (metacritic > sales > dbpedia), then to the lowest record number.

| Attribute | Rule |
|---|---|
| name | Raw title from the highest-priority source (dbpedia with disambiguation stripped) |
| platform | Highest-priority source spelling |
| releaseYear | Year vote → `YYYY-01-01`, within 1960–2024 |
| developer | Vote |
| publisher | Sales only |
| criticScore, userScore, ESRB | Vote across metacritic and sales |
| series | dbpedia franchise mode, disambiguation stripped |
| genres | JSON list: union of all members' source labels (deduplicated), plus the taxonomy top-level Genre Name from keyword rules; at most 10, source labels first |

No value is invented. Missing stays missing.

## Diagnostics (r2, label-free)
- **Coverage:** 74,951 of 74,951 source ids are in membership, with no unknown ids. There are 50,718 clusters, equal to the 50,718 fused rows, with `_id` values matching in both directions.
- **Cluster composition:**

  | Sources in cluster | Clusters |
  |---|---|
  | dbpedia only | 29,395 |
  | metacritic only | 10,133 |
  | dbpedia + metacritic + sales | 3,712 |
  | metacritic + sales | 3,336 |
  | dbpedia + metacritic | 3,313 |
  | sales only | 455 |
  | dbpedia + sales | 374 |

- **Cluster structure:** singleton share is 0.69. The largest cluster has 29 rows (dbpedia duplicate rows of one title/platform/year).
  No cluster holds more than one metacritic or more than one sales row.
- **Correspondences:** none are same-source and none cross clusters.
- **Schema checks:** ESRB, date-pattern, score-range and genres ≤ 10 checks show 0 violations.
- **Output density:**

  | Attribute | Density |
  |---|---|
  | releaseYear | 0.98 |
  | developer | 0.98 |
  | genres | 0.997 |
  | platform | 0.993 |
  | criticScore | 0.42 |
  | ESRB | 0.38 |
  | series | 0.37 |
  | publisher | 0.16 |

  Critic score, ESRB and publisher density is limited by source coverage.
- **Conflicts:** among multi-source clusters, 15.5% have a year conflict and 35% have a developer-string conflict.
  Most developer conflicts are naming differences such as "EA Sports" vs "EA Canada" or wiki names like "Key (company)".
- I traced sample multi-source clusters back to their members (for example Reign of Fire GC, Lords of the Fallen PS4, Army Men: Air Attack 2 PS1).
  Every fused value comes from its own members.

## Known weaknesses and uncertainty
- **dbpedia structure:** because dbpedia is a cross-product, "which dbpedia rows belong to an entity" is ambiguous.
  The year-split choice favours precision and structural validity over including all rows.
- **Output vocabularies:** the platform and developer spellings follow the sources. If the gold expects taxonomy names or other developer
  naming, exact-match comparison will fail.
- **Untested whitelists:** the brand-prefix and edition-suffix whitelists are judgement calls. Editions and remasters may or may not count as the same entity.
- **Unreachable records:** dbpedia records whose platform is unmapped or missing cannot match other sources.
- **Fuzzy thresholds** (year gap ≤ 1 or 2, critic ±1) were chosen by inspecting examples, not tuned against labels.
