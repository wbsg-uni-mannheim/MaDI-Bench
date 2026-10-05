# Games integration — method report (label-free)

**No labeled evaluation, no supervised training, no label/teacher service was used.** Every number below is a
structural, label-free diagnostic computed from the saved artifacts (`work/diagnostics.jsonl`,
`work/state/*`). None of them is precision, recall, F1, pair completeness or fusion accuracy. Those can't be
measured without ground truth.

Entity = one video game on one specific platform. Pipeline: `work/s1_schema.py` … `work/s7_diagnostics.py`, and
`work/rebuild.sh` regenerates everything deterministically from `task/input` (verified: a fresh copy with no
`submission/` reproduced identical correspondences, membership, fused, mapping and candidate files).

## Source observations (drove every design choice)
* All three sources carry heavy synthetic noise:
  * keyboard-neighbour and OCR typos (`Xhox 36O`, `Nintcndo`, `OD`/`0F`/`LX` = "PC")
  * word shuffles (`Journey Savage the to Planet`)
  * truncation (`Playstation Portab`, `Call of Duty: World at`)
  * dropped words
  * abbreviations (`GTA IV`, `THPS2`, `FH4`)
  * wrong acronym expansions (`Deutsche Schule`/`Double Strike`/`DisplayPort` = DS)
  * multiple date formats, all of them `01-01`, so only the year carries information
  * `UNCHANGED` placeholders
  * scores such as `7 9.0`/`ie.p`
* **dbpedia is a cross-product explosion.** One DBpedia article appears as many rows (for example, 496 rows for
  *Oddworld: Stranger's Wrath*), one per combination of platform, developer, year, genre and series. So many
  dbpedia rows describe the same (game, platform) entity. The per-row developer and year fields are
  combinations, not attributes of that platform version.
* metacritic and sales contain injected near-duplicates (mostly high ids with shuffled titles) and fabricated
  distractor games (no scores, invented titles).

## 1. Schema matching (`sm_mapping.csv`)
Columns were mapped by value inspection, not by name:

| Source | Mapping |
|---|---|
| dbpedia | `nm`→name, `ly`→releaseYear, `stu`→developer, `sys`→platform, `g`→genres, `fran`→series |
| metacritic | `game_title`, `year_published`, `made_by`→developer (0.85: sometimes the publisher), `console`, `gnrs`, `press_rating`→criticScore, `player_rating`→userScore, `age_rating`→ESRB |
| sales | `Attribute_2` name, `_3` year, `_4` developer, `_5` publisher, `_6` platform, `_7` genres, `_8` critic, `_9` user, `_10` ESRB |

Unmapped: sales `Attribute_11` (unit sales in mixed units, not in the target). Targets without contributors:
publisher exists only in sales, series only in dbpedia, and scores/ESRB only in metacritic and sales.

## 2. Normalization (`common.py`, `s2_normalize.py`, `work/taxonomy_plan.json`, `work/taxonomy_coverage.csv`)
* **Platform.** Pipeline, in order:
  1. Map to the taxonomy `Platform Name` through an alias table.
  2. Then try sorted-token, no-space and prefix (truncation) lookups.
  3. Then a keyboard/OCR-weighted Levenshtein distance. Neighbour-key substitutions cost 0.35.
  4. Ties are broken by a per-source frequency prior only when one side is at least 5× more frequent.

  Exotic real platforms (MSX, FM Towns…) keep their cleaned source label. Uninformative strings (`P`,
  `Nintendo`, `Microsoft`) become unknown. Store fronts XBLA/PSN/WiiWare/DSiWare map to their console.

  Canonical rate among non-empty values: dbpedia 0.867, metacritic 0.993, sales 0.992. The dbpedia remainder is
  mostly genuine exotic platforms.
* **ESRB.** A closed set, with aliases (`K-A`→E, `Teen`, `E10 Plus`…) and resolution of noisy E10+ patterns.
  Single-character keyboard noise is resolved only when unambiguous (for example, `J`/`K`/`N`→M; `R`/`F` stay
  null). Canonical rate: 0.996 in both sources.
* **Other fields.**
  * Year: the first 1960–2024 year in the string; parse rate 0.985–0.987.
  * Scores: spaces removed; kept only when in range.
  * Titles: accents folded, `(1993 video game)` dbpedia disambiguators stripped (the year is kept as
    `dis_year`), punctuation removed, roman numerals compared as numbers.

## 3. Blocking (`s3_blocking.py`, `submission/blocking/candidates.csv`)
**Title-level candidates** are the union of:
* char-3gram TF-IDF top-15 neighbours with cosine ≥ 0.35
* sorted-token keys
* a generic-word-stripped key (`FIFA Soccer 13` ~ `FIFA 13`)
* an abbreviation key (initials of the full title = letters of the acronym)

**Record-level candidates** are cross-source pairs whose titles are linked and whose platforms are compatible
(equal, or one side unknown).

**Extra candidate sources:**
* Untitled metacritic/sales records are blocked on (platform, critic score).
* 18 "cluster-closure" pairs are appended. These are pairs implied transitively by the clustering that no pass
  proposed directly.

**Totals:** 298,849 pairs:

| Source pair | Pairs |
|---|---|
| dbpedia–metacritic | 175k |
| dbpedia–sales | 78k |
| metacritic–sales | 45k |

The reduction ratio against the 1.30 billion full cross-source product is 0.99977. Records with at least one
candidate:

| Source | Records with a candidate | Share |
|---|---|---|
| dbpedia | 35,180 / 45,358 | 78% |
| metacritic | 17,223 / 18,735 | 92% |
| sales | 6,873 / 6,985 | 98% |

Records without a candidate are mostly exotic-platform dbpedia rows and untitled records. This is candidate
coverage, not pair completeness.

## 4. Entity matching (`s4_match.py`, `work/state/pair_scores.csv`)
**Nodes:**
* metacritic records and sales records
* dbpedia *units* = (article, platform)

An article is the rows sharing a normalized title and disambiguation year. Noisy title variants are folded into
the dominant spelling: word-order/roman-identical variants always, and others when title similarity is ≥ 0.90
and the dominant spelling has ≥ 3× the rows.

**Title similarity** is the maximum of the weighted similarity on the raw and on the token-sorted strings. Two
hard constraints apply:
* Numeric tokens must agree (`NBA 2K2` ≠ `NBA 2K3`, `FIFA 06` ≠ `FIFA 07`).
* A *word-substitution guard* applies when each title contains a ≥3-letter word the other lacks, as in
  autobots/decepticons or NFL/NHL. Such pairs are accepted only with ≥ 3 points of evidence (scores plus year).

**Evidence points:**

| Evidence | Rule |
|---|---|
| Year, metacritic↔sales | same +1, ±1 +0.5, ±2 −0.5, more −1.5 |
| Year, vs a dbpedia unit | year in the unit's year set or matching `dis_year` +1; earlier than `dis_year` −1.5 |
| Developer similarity | ≥ 0.85 +1 |
| Critic score | ±1 +1, > 5 −1 |
| User score | ±0.1 +0.7, ≥ 1.0 −0.5 |

**Acceptance with a known, equal platform** (title similarity, then required evidence):

| Title similarity | Evidence required |
|---|---|
| ≥ 0.97 | ≥ −0.5 |
| ≥ 0.88 | ≥ 1 |
| ≥ 0.80 | ≥ 2 |
| ≥ 0.70 | ≥ 2.5 |

Abbreviation pairs get similarity 0.85, so they need 2 points of corroboration.

**Rationale for these settings:** from inspected samples:
* Pairs above 0.97 were near-uniformly the same game, so only explicit contradictions reject them. This
  separates same-name remakes such as *NFS Most Wanted* 2005/2012 and *BioShock* 2007/2010 by year and score.
* The 0.80–0.97 band mixed typos with different titles, so corroboration is required there.

**Same-source duplicates** (metacritic/sales): same platform, no contradiction ≤ −0.5, and one of: similarity
1.0 with evidence ≥ 1, ≥ 0.97 with ≥ 1.5, or ≥ 0.80 with ≥ 2.5.

**Accepted cross-source node pairs:**

| Node pair | Accepted |
|---|---|
| dbpedia–metacritic | 3,848 |
| dbpedia–sales | 2,442 |
| metacritic–sales | 4,884 |

These counts include unknown-platform proposals that clustering phase 2 decides on. Same-source duplicate edges:
158 metacritic and 33 sales.

## 5. Clustering / refinement (`s5_cluster.py`)
Clustering uses a greedy, score-ordered constrained union:
* **Phase 0** merges same-source duplicates into one source group: 191 merges.
* **Phase 1** adds known-platform edges only if the component keeps at most one metacritic group, one sales
  group and one dbpedia unit. 62 edges were rejected by this constraint and are logged in
  `state/rejected_edges.csv`. Most are a second, different title variant, for example *Star Wars: Starfighter*
  vs *Jedi Starfighter*.
* **Phase 2** handles unknown-platform metacritic/sales records and dbpedia articles with no known platform. They
  attach to a component only if it is clearly singled out: evidence ≥ 2 with a margin ≥ 1 over the next
  component, or evidence ≥ 1 with the same margin when the target already holds a metacritic/sales record.
  Result: 1,062 attached, 653 left as singletons (`state/unknown_platform_decisions.csv`).
* **dbpedia rows with unknown platform:**
  * If the article has exactly one platform, the rows join that unit (524 rows).
  * If the article has several platforms, the rows stay singletons (3,906 rows); guessing the platform would
    invent assignments.
  * Rows with an empty title also stay singletons.

**Structure** (counts are clusters, broken down by the sources present):

| Metric | Value |
|---|---|
| Clusters | 45,962 |
| Largest cluster | 62 (a dbpedia unit plus its match) |
| Singleton share | 0.707 |
| dbpedia only | 24,916 |
| metacritic only | 12,053 |
| sales only | 1,556 |
| metacritic + sales | 3,212 |
| dbpedia + metacritic | 2,041 |
| dbpedia + sales | 913 |
| all three sources | 1,271 |

13,302 cross-source correspondences are emitted: every cross-source record pair inside a cluster. All of them are
in the candidate file.

One check on the matching: every exact (normalized title, platform) key that occurs in two sources ends up
co-clustered except 10, and those 10 are same-name remakes separated by year. Large dbpedia-only mass is genuine
non-overlap: mostly different platforms (Mac/Linux/iOS ports, retro systems). That was verified on title-overlap
samples.

## 6. Fusion (`s6_fusion.py`)
Each row is fused only from its own cluster's members. A dbpedia unit's exploded rows together count as one
vote.

| Attribute | Rule |
|---|---|
| name | One candidate per metacritic/sales record plus the modal dbpedia spelling. The best "quality" wins: tokens seen ≥ 3 times in the corpus count +1, rare (typo) tokens −1, trailing `:` (truncation) −1.5, abbreviation tokens −1.5. Ties go to metacritic, then sales, then dbpedia. |
| platform | Canonical taxonomy name (vote). 89% of non-empty values are taxonomy names. |
| releaseYear | Vote of metacritic/sales version years. Otherwise the dbpedia disambiguation year, otherwise the earliest dbpedia year (the per-row dbpedia years are a cross-product). Output as `YYYY-01-01`. |
| developer / publisher / series | A global repair maps a spelling to a ≥ 5× more frequent, similar spelling (`Copeom`→Capcom, `Games Midway`→Midway Games, `Arc 5ystcm Works`→Arc System Works). Then a vote; ties go to the globally most frequent spelling, then metacritic > sales. Developer comes from metacritic/sales first (the dbpedia studio field is cross-product noise). |
| genres | JSON list: taxonomy top-level genres mapped from source genres, plus repaired source genre strings, ordered by support, at most 10. The comparison is a subset test and the vocabulary is non-exhaustive, so both representations are kept. |
| criticScore / userScore / ESRB | Vote across metacritic and sales; ties go to metacritic, the originating source of Metacritic scores. |

Measured disagreement inside metacritic+sales clusters:

| Attribute | Disagreement |
|---|---|
| critic | 1.8% |
| ESRB | 0.2% |
| year | 12% (mostly metacritic = sales + 1) |
| developer key | 36% (spelling and multi-studio variants) |
| user score | 75% (median absolute difference 0.2; the two sources look like different snapshots) |

The metacritic priority for user score and year is therefore a documented assumption, not verified correctness.

**Output checks:**
* 45,962 rows, and the `_id` set equals the membership cluster set.
* 0 schema violations (date pattern and range, ESRB set, score ranges, genre list size).
* 0 fused names that aren't traceable to a member's raw name.

**Density:**

| Attribute | Density |
|---|---|
| name | 0.95 |
| year | 0.89 |
| developer | 0.90 |
| genres | 0.93 |
| platform | 0.86 |
| critic | 0.40 |
| user | 0.34 |
| ESRB | 0.35 |
| series | 0.20 |
| publisher | 0.13 |

Missing values were never filled by guessing.

## Iterations (see `work/step_log.jsonl`, `work/revisions/` (snapshots r1, r4–r8, final; r2/r3 were intermediate and not snapshotted))
1. **r1: baseline.** Assumed one record per source per entity.
2. **r2: same-source duplicate detection.** Found injected near-duplicates, which the one-per-source constraint
   had been blocking.
3. **r3: word-substitution guard, `UNCHANGED` treated as missing.** Removed accepted *NFL/NHL* and
   *autobots/decepticons* pairs.
4. **r4: dbpedia articles with no known platform become attachable '?' units.** A 106-row unknown-platform unit
   was avoided by keeping multi-platform unknown rows as singletons. The weak-evidence attach rule was restricted
   after an RE5 record was misattached by exclusion.
5. **r5: roman numerals, `disney` prefix, store-front platforms.** Added 54 correspondences.
6. **r6: attribute matching for untitled records.** 42 matched uniquely on platform + critic + user (+ year).
7. **r7: abbreviation blocking and matching.** 22 accepted, with a number-consistency fix after an *MLB 2K11 /
   2K12* false merge.
8. **r8: word-order-identical dbpedia title variants always folded.** Fixed shuffled-title article fragments
   that were being blocked by the one-unit constraint.

Fusion: the Repair function and frequency tie-breaks were added, and a genre regex bug (`rts` matched *sports*)
was fixed.

## Rejected approaches
* Assigning dbpedia unknown-platform rows to the article's most frequent platform: this would guess.
* Inferring the platform from missing cross-product combinations: the product is sampled and noisy, so it's
  unreliable.
* A strict one-record-per-source constraint.
* Supervised or pseudo-labelled matchers: not permitted and not used.

## Known weaknesses and uncertainty
* The format of the gold platform and developer values is unknown. Platforms use taxonomy names (`Windows PC`,
  `PlayStation Portable (PSP)`); if the reference uses source spellings (`PC`, `PSP`), exact platform
  comparison will fail.
* Metacritic priority for user score and year is an assumption.
* dbpedia rows whose platform is missing in a multi-platform article, and records with an empty title (except 42
  metacritic/sales ones), can't be linked.
* dbpedia-only rows use the earliest year, which is wrong for later ports.
* Title thresholds are hand-set from inspected cases. Some sequel/subtitle confusions surely remain (for example,
  HD remasters matched to originals on the same platform), and some noisy true matches below 0.8 similarity are
  missed.
* The abbreviation expansion can match a wrong title that shares its initials when year and developer happen to
  agree.
