# Games integration: method report (final revision r5-filler-remaster)

**No labeled evaluation and no supervised training were done.** No matcher was trained or fine-tuned, no pseudo-labels were created, and no embeddings or LLM calls were used. Every decision comes from deterministic rules that I reasoned out by looking at the source data. The numbers below are structural diagnostics only. They are not precision, recall, pair completeness or fusion accuracy.

Rebuild: `work/rebuild.sh` runs `s1_schema → s2_normalize → s3_blocking → s4_matching → s5_cluster → s6_fusion → s3b_export_candidates → diagnostics`. It takes about 2.5 minutes on one CPU and uses no network. I ran it in a fresh copy of the workspace, and it reproduced all five data files exactly. Earlier coherent revisions are kept in `work/revisions/r1..r4`. Actions are logged in `work/step_log.jsonl` and diagnostic results in `work/diagnostics.jsonl`.

## Source observations
- **dbpedia** (46,156 rows, 10.8k distinct names). The rows are cross-products of multi-valued article properties: one article gives one row per platform × date × developer × genre combination. Injected noise includes random casing ("OddwoRld"), removed or added whitespace ("IdSoftware", "shoote r"), and platform names leaking into the developer column. 1,804 rows come from collection articles ("List of Monster Jam video games", "... video game series"), which describe several different games under one title.
- **metacritic** (20,862 rows). (name, platform) is unique. Dates and scores contain injected whitespace ("2 011-01-01", "9 3.0").
- **sales** (11,443 rows). About 3,050 rows have no name, only a developer and a genre with scrambled word order ("Studios Tarsier", "puzzle Action"). These cannot be identified, so they stay as singletons. This is the only source with a publisher.
- Synthetic records (for example "Asterquill") appear consistently in all three sources. They include deliberate near-duplicate names ("Aster Ledger" / "Star Ledger", "Orchard Lanterns" / "Orchid Lantern"), and these are **not** merged.

## 1. Schema matching (`sm_mapping.csv`)
The column names line up semantically. I checked each one against its values. `sales.globalSales` has no target attribute and is left unmapped. `publisher` comes only from sales, `series` only from dbpedia, and scores and ESRB only from metacritic and sales.

## 2. Normalization
- **Name key:** accent-folded, lowercased, `&`→`and`, all non-alphanumerics removed. This removes the case and whitespace noise. For output, each key uses its most frequent raw spelling across all sources.
- **Platform:** an alias map (`work/platforms.py`) converts values to taxonomy names (PS4 → PlayStation 4, Switch → Nintendo Switch, DC → Sega Dreamcast, XOne → Xbox One, and so on). The PC family (PC, Microsoft Windows, Windows 9x/XP, ...) becomes **"PC"**, following the target schema's own example value; the taxonomy is non-exhaustive. Platforms not in the map keep their most frequent raw spelling. Canonical-platform rates are in `work/taxonomy_coverage.csv`. Most unmapped values are the deliberate "PC" and long-tail legacy dbpedia platforms.
- **Dates:** "Month DD, YYYY" and whitespace-corrupted ISO dates are converted to YYYY-MM-DD. All non-null dates parse. Dates outside the schema range 1960-01-01..2024-12-31 (a few synthetic 2025/2026 dates) are dropped rather than emitted.
- **ESRB:** K-A→E (the schema's alias). The value set is closed.
- **Scores:** spaces removed, then range-checked.
- **Developer and series:** dbpedia disambiguation suffixes such as "(company)" or "(franchise)" are stripped for comparison. Developer values that are actually platform names are dropped.
- **Genres:** lists are split on commas. Token-scrambled genres are repaired by matching their token multiset to a clean genre string seen in the data ("shooter First-person" → "First-person shooter").

## 3. Blocking (`blocking/candidates.csv`, 94,724 record pairs)
A unit is (source, name key, platform). Collection-article rows and junk sales rows are single-record units. Units are compared across sources in three ways:
- (a) same platform and exact name key;
- (b) same platform and rapidfuzz token_set_ratio ≥ 75;
- (c) units without a platform against every unit with the same name key.

At unit level this produced 18,288 exact, 44,141 fuzzy and 3,660 no-platform candidates. They are expanded to record pairs. All correspondences are contained in the candidate file (checked: 100%).

## 4. Matching (rules, `work/s4_matching.py`)
- **Exact name + platform:** accepted. The one exception is a remake contradiction, where both years are known, differ by ≥5, and the critic scores differ by >3. Exactly one pair was rejected this way (Resident Evil 4, PC: 2007 vs 2023). dbpedia years are not used as evidence because its rows are cross-products.
- **Fuzzy:** accepted only if the token sets are identical after removing filler/brand words (the, game, video, Disney, DreamWorks, Pixar, Sid Meier's, Tom Clancy's, Marvel('s), James Bond, soccer), the number/roman-numeral tokens are identical, and the year gap is ≤1 where known.
  - Why: the fuzzy candidates are dominated by sequels (NBA 2K16/2K17, Madden 07/08) and edition variants (Empires, Deluxe, Ultimate), and I inspected many of them.
  - 343 fuzzy edges were accepted, and 52 of these came from the next rule.
- **dbpedia article vs "HD"/"Remastered" release on the same platform:** accepted only when neither unit has an exact match. The dbpedia article lists that platform only because of the remaster. This is the least certain rule.
- **Platform-less units** attach to a same-name cluster if that cluster is unique. Otherwise, for metacritic and sales records, they attach if exactly one candidate cluster has an **identical critic score** in the other source, with user score within 0.5 and year within 1. On exact matches, metacritic and sales critic scores differ by more than 1 in only 0.1% of pairs. This attached 387 units, 132 of them through the critic-score evidence. 918 remain unattached as ambiguous.

## 5. Clustering
- Union-find over accepted edges, with the exact edges applied first.
- Each metacritic or sales unit may appear at most once per cluster. 11 fuzzy edges were rejected for violating this; all rejections are in `work/state/rejected_edges.csv`.
- The rows of one dbpedia unit share a cluster: they are fragments of one article on one platform.
- Collection-article rows are singletons. The first revision (r1) grouped them, which created clusters of up to 71 records; the self-check flagged these, and inspection showed they were different games sharing a list title.

## 6. Fusion
- Each source contributes one representative per attribute: the mode over its member rows, so exploded dbpedia rows cannot outvote other sources.
- Values are compared by normalized key. The value supported by the most sources wins. Ties go to metacritic, then sales, then dbpedia, because metacritic is the cleanest-formatted source. This priority is a heuristic; I have no evidence about which source is more accurate.
- Developer strings keep each source's own separator.
- Genres are the de-duplicated union (extra genres are allowed by the comparison), capped at 10.
- criticScore is an integer and userScore is rounded to 1 decimal.
- Per-cluster provenance is in `work/state/fusion_provenance.csv`.

## Diagnostics (final, from saved files)
**Coverage and joins**
- All 78,461 source records are in membership, each exactly once. 0 unresolved ids. The fused `_id` set equals the cluster set.

**Clusters**
- 44,709 clusters: 3 sources in 4,072; 2 sources in 6,892; 1 source in 33,745. Singleton share is 0.617.
- 207 clusters have >9 records (4.1% of records). These are dbpedia single-title units with many cross-product rows, for example Oddworld: Stranger's Wrath on PS3.
- Share of records in multi-source clusters: sales 93% (excluding junk rows), metacritic 50%, dbpedia 26%.

**Validity and density**
- Schema validity: 0 invalid ESRB, 0 invalid or out-of-range dates, 0 out-of-range scores, 0 genre lists over 10 items.
- Density: name .93, date .88, developer .95, genres .97, platform .90, critic .46, user .42, ESRB .42, publisher .17. Publisher exists only in sales.

**Source disagreement inside metacritic+sales clusters (7.4k)**

| Attribute | Rate |
|---|---|
| Year differs | 11% (mostly by 1 year, likely regional release dates) |
| userScore differs by >0.1 | 55% |
| Critic differs by >1 | 0.1% |
| ESRB differs | 0% |
| Developer differs | 2.5% |

The userScore conflict is resolved by the metacritic priority. I cannot verify which source is correct.

## Known weaknesses and uncertainty
- The platform representation (taxonomy names, "PC" kept as "PC") is a judgment call. If the reference keeps source abbreviations (PS4, DS), platform comparisons would fail systematically.
- Single-source dbpedia clusters keep the raw dbpedia developer string, including disambiguation suffixes. In multi-source clusters the metacritic or sales spelling wins.
- Every dbpedia row of one (title, platform) unit is grouped. For franchise articles ("Uno", "Scrabble", "SimCity") such a unit may still mix several distinct games on one platform. Only explicit list/series/collection pages were split.
- Exact name + platform matches can merge a remake with the original when the critic evidence is missing. The HD/remaster rule may also merge an original release with its remaster.
- 918 dbpedia/metacritic/sales records without a platform remain unattached because the name exists on several platforms. About 3k name-less sales rows are singletons with only developer and genre.
- Rejected approaches: fuzzy acceptance by similarity threshold alone (it merged sequels and synthetic look-alikes), and grouping collection-article rows by title.
