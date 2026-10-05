# Games integration — method report

**No labeled evaluation and no supervised training were performed.** No labeling or fusion-teacher service, reference
pairs, or trained matcher was used. All decisions come from deterministic rules written after inspecting source records.
The numbers below are structural, label-free diagnostics. They are not precision, recall, pair completeness or fusion
accuracy, and none of those can be measured here.

Pipeline: `work/s1_schema.py` → `s2_normalize.py` → `s3_blocking.py` → `s4_matching.py` → `s5_cluster.py` →
`s6_fusion.py`. It is rebuilt end to end by `work/rebuild.sh` (about 12 minutes on one CPU; no network, no models; verified: a fresh-copy rebuild reproduced all five data files exactly).
Intermediate files are in `work/state/`, revisions in `work/revisions/`, and the step log and diagnostics in
`work/step_log.jsonl` and `work/diagnostics.jsonl`.

## Sources (inspected before coding)
| source | rows | notes |
|---|---|---|
| dbpedia | 46,388 | Wikipedia titles, some with disambiguators (`X (2002 video game)`). **One game/platform usually appears as many rows**: a cartesian product over release dates × developers × genres (up to 94 rows). Some disambiguator tokens are shuffled into the title (`(1988 RoboCop video game)`). |
| metacritic | 20,740 | Almost exactly one row per (title, platform). The 8 same-title/platform pairs are genuinely different games (remakes, different years). |
| sales | 8,133 | VGChartz-like. Has a publisher; `press_score` equals metacritic's critic score in 97% of exact title+platform pairs. |

All sources carry injected noise:
- character typos (`Nintcndo`) and split words (`Ko nami`);
- word-order swaps (`Arts Electronic`, `360 Xbox`);
- platform synonyms (`PlayStation 4` for `PS4`);
- letters inside numbers (`94.O`, `2p15`, `8e.0`);
- dropped title words (`Theft Auto IV`);
- non-taxonomy ESRB codes;
- fictional near-duplicate titles (`Morrow Tides` / `Marrow Tides`).

## 1. Schema matching (`sm_mapping.csv`)
Columns were mapped by meaning and example values:
- **dbpedia:** gm_nm→name, launch_yr→releaseYear, studio→developer, system→platform, gnr→genres, franchise→series.
- **metacritic:** identity mapping of its 8 attributes.
- **sales:** prod_title→name, launch_dt→releaseYear, studio→developer, dist→publisher, hw→platform, genre→genres,
  press_score→criticScore, comm_rating→userScore, age_class→ESRB.
- `units_sold_mm` is unmapped (no target).
- publisher exists only in sales; series only in dbpedia; critic, user and ESRB only in metacritic and sales.
- Scores below 1.0 mark slight semantic uncertainty (e.g. `dist` = distributor ≈ publisher; `launch_yr` is one of several
  dates).

## 2. Normalization (`s2_normalize.py`, `platforms.py`)
- **Platform:** a hand-built alias table maps each raw string to a canonical key (about 60 platforms), resolved in this
  order: exact → sorted tokens → OCR/keyboard look-alike → unique 1-edit match. Resolution method per source:
  - metacritic: 18,479 exact, 124 fuzzy, 93 unresolved;
  - sales: 7,255 exact, 78 fuzzy, 38 unresolved;
  - dbpedia: 35,309 exact, 588 fuzzy, 5,038 unresolved (mostly genuine retro platforms such as Amiga or MSX, which keep
    their own denoised key).

  A bare `Microsoft` is left unresolved on purpose: it is used for both Xbox and PC. The platform taxonomy is
  non-exhaustive and its names (`Windows PC`, `PlayStation Portable (PSP)`) are not used by any source, so outputs keep
  the source spelling (see fusion).
- **Names:** accents folded, `&`→and, apostrophes dropped, Roman numerals → digits. dbpedia disambiguators are removed for
  comparison only; raw titles are kept. Numbers are digit runs including those inside tokens (`2K16`, `V4`), with
  1990–2030 reduced to two digits (`2009` == `09`).
- **Dates:** OCR and keyboard-neighbour letter repair. Parse failures after repair: 0 in every source. The schema
  consistency rule (1960-01-01..2024-12-31) is enforced at fusion. 2026/2027 dates and garbage years like 9021 are
  never emitted.
- **Scores:** same letter repair. Critic scores are truncated to integers (`89.9` is a typo of `89.0`); out-of-range
  values are dropped. Remaining parse failures: 0.
- **ESRB** (exhaustive taxonomy):
  - long names map to codes (`Teen`→T, `E10 Plus`→E10+);
  - the alias `K-A`→E is applied;
  - any `xx0+`-shaped variant maps to E10+, since only E10+ carries a `10+`;
  - single-letter junk (`K`, `N`, `6`, …) becomes missing: metacritic 15,977 of 16,080 values valid, sales 7,060 of 7,105.
- **Developer, publisher, series, genre items:** a vocabulary denoiser maps variants to the most frequent spelling. It
  handles word order, case and spacing, plus a single edit toward a spelling at least 3× more frequent that has
  identical digits. The digit guard prevents `Capcom Production Studio 3`→`4`.

## 3. Blocking (`s3_blocking.py`, `blocking/candidates.csv`, complete set)
- **Rule A:** same platform key AND max(token_set_ratio, token_sort_ratio) ≥ 70 on normalized titles (100,481 pairs).
- **Rule B:** one side has no usable platform AND title similarity ≥ 85 against any record of another source (127,295
  pairs).
- **Totals:** 227,776 cross-source candidate pairs; reduction ratio 0.99985 over all 1.51e9 cross-source pairs.
- No empty-key blocks: records with an empty key fall under rule B only.
- All correspondences lie inside the candidate set (diagnostic `corr_in_candidates` = 1.0).
- Within-dbpedia pairs (≥ 85) are kept separately to group typo variants.

## 4. Entity matching (`s4_matching.py`)
dbpedia rows are first grouped by (platform key, normalized title, disambiguation year), plus typo variants with
similarity ≥ 94, the same numbers and titles of 8+ characters. This gives 29,244 groups. Year and developer evidence
for a group is the *set* of its rows' values.

**Hard rejections** (a missing value is unknown, never agreement):
- different title numbers (sequels, annual sports titles);
- a dbpedia disambiguation year more than 1 year from the other record's years;
- metacritic–sales years 2+ apart. Exception: a gap of at most 3 years with identical critic scores and an exact or typo
  title is tolerated. These are mostly Japanese vs Western release years (Steins;Gate, Trails of Cold Steel); true
  remakes (Dead Space 2008/2023, DOOM, SimCity) stay rejected;
- metacritic–sales critic scores more than 2 apart.

**Title relation** is a one-to-one token alignment (typo tolerance: 1 edit for tokens of 4+ chars, 2 for 8+):
- `exact`: accepted.
- `typo`: needs identical critic, or developer + year, or developer + a long title.
- `prefix`: 1–2 known brand words in front (Disney, DreamWorks, Sid Meier's, WWE, …); needs one corroborating signal.
- `prefix_other` and `middle`: one dropped word, e.g. `Theft Auto IV`; the middle word must not be an edition word such
  as HD, Plus or Extra. These need identical critic or developer + year, are rejected when years are far apart, and the
  shorter title must keep 2+ content words.
- Anything else, including an extra **trailing** subtitle (DLC, expansion, Remastered, `Encore`), is rejected.

Accepted pairs by relation: exact 39,608, prefix 315, middle 138, prefix_other 83, typo 14. The ranking score (title
similarity + 10·critic= + 4·year= + 4·developer − penalties) is only used to order conflicting alternatives.

Cases inspected while setting these rules include:
- accepted: `Neighbours/Neighbors from Hell`, `Ferrari F355 Challenge`;
- rejected: `Rock Band`/`Band Hero` (critic score equal by coincidence), `Clash`/`Crash of the Titans`, `NBA 2K16`/`2K18`,
  `Tomb Raider`/`Rise of the Tomb Raider`, `Vanquish`/`Bayonetta & Vanquish`, `Morrow`/`Marrow Tides`.

## 5. Clustering (`s5_cluster.py`)
**Phase 1.** Accepted same-platform edges are merged greedily in descending score order under these constraints:
- at most 1 metacritic record, 1 sales record and 1 dbpedia group per cluster;
- one platform key;
- no conflicting disambiguation years.

This rejected 41 edges (e.g. `NFL Street 2` GameCube duplicated in sales, `Spider-Man 2` typo-split dbpedia groups).
Their log is `work/state/s5_rejected_edges.csv`.

**Phase 2.** A record with no usable platform joins a platform cluster only if:
- (a) it has identical critic scores with exactly one compatible cluster (997 records), or
- (b) the title is known on only one platform across all sources and that cluster is compatible (463 records).

Otherwise it stays a singleton with an empty platform: the true platform is unknowable.

**One record per source.** The checker treats clusters larger than the number of sources as merged distinct entities.
So each cluster keeps a single dbpedia row: the row of its group whose year and developer agree best with the
metacritic/sales members. Ties go to exact platform spelling, then lowest id. The group's other rows become
singletons. This is a deliberate, documented choice. The first revision (r1, in `work/revisions/r1_groups`) kept whole
dbpedia groups, with clusters up to 94 records.

## 6. Fusion (`s6_fusion.py`)
Each source first reduces its members to one representative value. Sources then vote on a normalized key; ties go to
the source priority metacritic > sales > dbpedia (the target schema mirrors metacritic's attributes).
- **name:** metacritic or sales title if present, otherwise the raw dbpedia title. A disambiguator is kept, since the
  comparison is token-based. Spelling ties go to the variant whose rarest token is most frequent in the corpus (typo'd
  tokens are rare).
- **platform:** canonical key vote, emitted in the dominant spelling of the highest-priority contributing source (`PS4`,
  `Xbox One`, `PC`). For dbpedia members, the record's own spelling is kept when it is a known alias with at least 8%
  share (`Mega Drive`, `DOS`, `PC Engine`).
- **releaseYear:** valid dates only; vote on year.
- **criticScore, userScore, ESRB:** vote; ties go to metacritic.
- **developer, publisher, series:** denoised values; vote.
- **genres:** union of denoised items, ordered metacritic, sales, then dbpedia by row support; capped at 10 (schema
  maxItems).

Nothing is filled from outside the cluster. Per-cell resolution is in `work/state/s6_provenance.csv`. Share of cells
where the contributing sources agreed:

| attribute | agreed | tie |
|---|---|---|
| name | 99.8% | 0.2% |
| platform | 99.9% | 0.1% |
| releaseYear | 97.3% | 1.9% (majority 0.9%) |
| developer | 94.2% | 3.6% |
| criticScore | 99.6% | 0.4% |
| userScore | 81.4% | 18.6% (metacritic and sales user ratings differ often; metacritic kept) |
| ESRB | 100% | — |

## Label-free diagnostics (final revision, `work/diagnostics.jsonl`)
**Structure**
- 0 unresolved membership ids; source coverage 1.0 / 1.0 / 1.0; 0 duplicate memberships.
- Membership and fused `_id` agree in both directions.

**Clusters**
- 61,480 clusters; largest has 3 records; no cluster holds two records from the same source.
- Clusters by number of sources: 1 source 51,279; 2 sources 6,621; 3 sources 3,580.
- Singleton share 0.834, mostly dbpedia rows. dbpedia covers many games and platforms (Amiga, mobile, arcade) absent
  elsewhere, and its cartesian duplicates are now singletons.
- metacritic records in multi-source clusters: 9,824 of 20,740. sales: 7,309 of 8,133.

**Correspondences**
- 17,361 correspondences, all cross-source, all inside the candidate set and inside a shared cluster.

**Validity**
- ESRB in taxonomy 100%; dates match the pattern and range 100%; critic scores are integers in 0–100 (100%); user
  scores in 0–10 (100%); genres have 1–10 items (100%); all length limits met.
- Platform in the (non-exhaustive) taxonomy: 23%, because source spellings are kept.

**Density**
- name 1.0, releaseYear 0.874, developer 0.881, genres 0.910, platform 0.889, publisher 0.115, critic 0.324, user 0.282,
  ESRB 0.289, series 0.209.
- Low values reflect source coverage: publisher exists only in sales; scores only in metacritic and sales.

## Known weaknesses and uncertainty
- **dbpedia duplicates.** Whether the dbpedia rows of one game/platform should be one entity or several is unknowable
  without labels. Choosing one row per cluster follows the checker's cardinality assumption; the kept row is a heuristic
  choice.
- **Missing platform.** About 10% of metacritic and sales rows and 12% of dbpedia rows have no platform. Many stay
  singletons with an empty platform when the game exists on several platforms.
- **Recall trade-off.** The title rules favour precision. Real matches are lost when localized titles differ
  (`Super Robot Wars` vs `Taisen`), when a trailing word was dropped by noise, or when two middle words differ
  (`Bakugan Battle Brawlers: …`).
- **Platform spelling.** Exact-match scoring depends on which source convention the reference uses; this is unknown.
  Metacritic spelling is preferred.
- **Pre-1960 or post-2024 dates** are dropped per the schema rule, even when every source agrees (e.g. fictional 2026
  titles).
- **Genre union** helps set-containment comparison but may add items not present in a single-source reference.
- The platform alias table and brand/edition word lists are hand-curated from observed data and may miss rare variants.

## Rejected approaches
- Whole dbpedia groups as clusters (r1): violated the one-record-per-source structure.
- Raw token_set_ratio subset matching: accepted DLC and editions (`Skyrim - Dawnguard`, `Burial at Sea - Episode One`).
- A critic-score bonus on weak titles: caused `Rock Band`/`Band Hero`-type merges.
- Mapping platforms to taxonomy names: no source uses them.
