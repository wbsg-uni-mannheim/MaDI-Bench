# Products integration — method report

**Condition:** no labeled data, no labeling/fusion-teacher service, and no supervised training or fine-tuning.
I made all matching decisions myself, by rule or by directly inspecting source records. I have no reference
data, so this report contains **no accuracy, precision, recall, F1, pair-completeness or fusion-accuracy
numbers**. Every number below is a structural, label-free diagnostic computed from the saved artifacts.

Rebuild: `bash work/rebuild.sh` runs `work/s1_schema.py … s7_export.py` and then `work/diagnostics.py`.
It uses only the four source tables, the target schema and two retained review artifacts
(`work/manual/review_snapshot_v1.txt` and `work/manual/decisions.txt`). It makes no network calls and trains no model.

## Data observations
- There are 4 sources: products_1..4 with 660, 635, 574 and 498 rows, 2,367 in total. All share the same 25-column layout under different names (`products_3` uses `Attribute_N`).
- The data contains heavy injected noise:
  - OCR-like character corruption of brand, model and categorical fields (`Zqmxung`, `5ATA`, `US8 3.l`, `G.O`). The same corrupted string often repeats across records.
  - Token-shuffled values (`GeForce SUPER 2070 RTX`).
  - Truncated values (`Low Profi`).
  - Locale number formats (`3.500,0`, `2 000,0`).
  - Storage given in TB instead of GB in about 97 rows.
  - Weights and dimensions in mixed units.
- Titles are the least corrupted identity evidence.
- 8 synthetic out-of-domain products appear once in every source (kettle, two monitors, two RAM kits, speaker dock, flash drive, NVMe drive).
- The same product often appears twice within one source (59 fused clusters contain two records from the same source). So I did **not** impose a one-record-per-source constraint.

## 1. Schema matching (`s1_schema.py`, `submission/sm_mapping.csv`)
- Mapped by meaning and checked against sample values from every source. Column order matches across sources, but I verified that rather than assuming it.
- All 25 source columns map 1:1 to target attributes (`id`→id, `brnd/br/Attribute_3`→brand, …).
- No populated source column is left unmapped, and every target attribute has contributors in all sources.
- Scores: 1.0, or 0.8–0.9 for storage, dimensions, weight, storage-connection type and form factor, whose units or representations are inconsistent in the sources.
- The translated table keeps the raw strings. The input files are never modified.

## 2. Normalization (`normlib.py`, `s2_normalize.py`)
These are identity features used for blocking and matching:
- **Brand key**, in priority order:
  1. Brand names found in the title (the board partner beats NVIDIA/AMD).
  2. Exact alias match of the brand field (≈80 alias lists).
  3. Fuzzy match of the brand field.
  4. Product-line cues such as "My Passport"→WD or "KC2500"→Kingston.
  5. Description or URL.
- **Product type**: field aliases and corruptions (`55D`, `6PU`, `Solid State Drive` …), otherwise title keywords. The enum is closed (GPU/SSD/HDD/USB_STICK); out-of-domain items stay empty.
- **Capacity (GB)** from the title, with TB→GB conversion. It excludes interface rates (`6G SAS`, `6Gb/s`) and falls back to the storage field. The parser went through three iterations after in-cluster conflicts exposed misparses.
- **GPU chip** (`rtx 2070 super`, `quadro p1000`, `wx 7100` …) and **VRAM** come from the title, model or chipset fields.
- **Model-code tokens** are alphanumeric tokens of at least 6 characters (MPNs such as `WD40EFRX`, `GV-N207SWF3OC-8GD`).

## 3. Blocking (`s3_blocking.py`, `submission/blocking/candidates.csv`)
The automatic candidates are the union of three methods:
- **(A) Spec-key blocks** on (product type, capacity or GPU chip). Pairs must have the same brand, or at least one brand unknown.
- **(B) Shared model codes** that occur in 2–15 records.
- **(C) Character 3–5-gram TF-IDF title kNN**: k=8, cosine ≥0.35.

Automatic totals:
- 19,991 cross-source pairs out of 2,093,192 possible, a reduction ratio of 0.990.
- Unique contributions: spec 11,116, kNN 5,351, code 165.
- 1 record has no automatic cross-source candidate. The largest spec block holds 184 records.

The exported candidate set also adds:
- all 71 remaining cross-source pairs inside the spec blocks I reviewed by hand (every such pair was inspected);
- 9 pairs I linked explicitly across blocks.

That gives 20,071 pairs in total, and 100% of the correspondences lie inside it. This candidate coverage is not a recall figure.

## 4. Entity matching (`s4_matching.py` + direct review)
- `s4` is an interpretable automatic scorer:
  - **Hard contradictions:** product type, brand, capacity (>3%), GPU chip, VRAM.
  - **Positive evidence:** a shared rare code, or IDF-weighted descriptive-token Jaccard ≥0.55 with no difference in GPU variant tokens.
- Inspection showed it was both too conservative (it missed many listings of the same card) and wrong on variants (it merged ASUS Phoenix with TUF-3, and Aorus Waterforce WB with the AIO version). I therefore used it only as a proposal and diagnostic.
- **Final decisions were made by direct inspection of every spec block**: 388 blocks covering all 2,367 records, grouped by type, brand and capacity or chip, with the automatic proposals shown.
- The decisions are stored as 742 group lines in `work/manual/decisions.txt`. They are my own judgments, not reference labels, and I did not use them to score anything.

Rules I applied, following the entity definition (one manufacturer configuration):
- Model or part numbers win when present. Regional SKU suffixes of the same configuration were grouped (e.g. `MZ-76E1T0B/AM` vs `MZ-76E1T0BW`).
- **GPUs:** chip, VRAM, cooler or board line (Strix/TUF/Dual/Phoenix/Turbo/Windforce/Gaming/AMP/Twin Fan …) and the factory-OC level (`O8G` vs `A8G` vs `8G`, OC vs non-OC, EVO vs EVO V2, WB vs AIO) all define distinct entities.
- **Drives:**
  - Different entities: SAS vs SATA, 512e vs 512n, SFF vs LFF, 6G vs 12G, capacity, colour of portable drives, 2.5" vs M.2 versions, generations (860 vs 870 QVO, MP510 vs MP510B).
  - HP spare vs option numbers were grouped only when the specs agree.
  - OEM-rebadged drives with the same Seagate MPN (Dell `ST2000NM0023`, EqualLogic `ST3300657SS`) were grouped with the Seagate records.
- Ambiguous listings stay singletons rather than being forced into a group. Examples: a colour-less "T5 500GB", "Phoenix Boost" without OC information, and generic multi-capacity no-name USB sticks.
- Identical synthetic out-of-domain products were grouped (one record per source each).

Automatic vs final: 949 automatically accepted cross-source pairs, 2,255 final correspondences, 810 in common. This shows only how much the review changed the automatic proposals; it is not a quality measure.

## 5. Clustering (`s5_cluster.py`)
- Clusters are the connected components of the review group lines. Group lines are unions, so an id on two lines joins both groups. Reviewed records drop all their automatic edges.
- Result: 959 clusters.
  - Sizes: 1: 228, 2: 225, 3: 394, 4: 80, 5: 16, 6: 10, 7: 2, 8: 3, 9: 1.
  - Sources per cluster: 1: 232, 2: 230, 3: 411, 4: 86.
- The largest clusters are 860 EVO 250GB, 860 EVO 1TB/2TB and T5 1TB (8–9 offers). I inspected them: their capacities and form factors agree.
- The only remaining within-cluster "conflicts" in parsed capacity come from rounding or noisy listings (1.9 vs 1.92 TB, 3.8 vs 3.84 TB, `USB 3.1750 GB`), not from genuine entity conflicts.
- A final check of all 228 singletons against their TF-IDF nearest neighbours found no clear missed match. The high-similarity neighbours differ in chip, capacity or variant.

## 6. Fusion (`s6_fusion.py`, `work/state/s6_provenance.csv`)
Every value comes from the cluster's own members.
- **title, description, url, price, currency:** taken from the medoid member (the one with the highest title-token overlap with the others). Currency is kept only if it is a 3-letter code seen at least 3 times.
- **brand:**
  - Vote over trusted raw brand strings, i.e. an exact alias or all tokens present in member titles, grouped by canonical brand.
  - The winning group's most frequent spelling is used.
  - Fallback: the most common spelling of the brand derived from the title.
- **product_type:** majority of normalized types; closed enum.
- **model, model_number, chipset, bus, interface, storage connection, memory type, colour, form factor:**
  - **Trusted values:** a value counts only if (a) its tokens appear in member title/description/URL text; or (b) its collapsed form appears in a member text on token boundaries, which rejects truncations; or (c) it occurs in at least 2 records and all its tokens are words seen in titles/descriptions (part numbers are exempt from the word test). Rule (c) exists because OCR-like corruptions repeat across records, so frequency alone is not proof of cleanliness.
  - **Voting:** votes are grouped by token set, so shuffled copies support the same value. Ties go to consensus with the other values, then global frequency. The spelling chosen is the one appearing verbatim in a member title, if any.
  - **Reordering:** a single shuffled value that appears only once is put back into title word order.
  - **Typo repair:** categorical fields only (not model or model_number). A value is repaired only to a close trusted vocabulary value with identical digits.
  - Values that stay untrusted are left empty rather than guessed.
- **VRAM, storage, speeds, dimensions, weight:** numeric vote with 2–3% tolerance.
  - OCR repair for numerics (`G.O`→6.0, `B.O`→8.0).
  - Storage uses both the title and the field; TB field values become GB when the title states the TB figure or when an HDD field is ≤24.
  - HDD dimensions below 10 are treated as inches; HDD/GPU weights below 5 as kg.
- **Schema applicability:** chipset and VRAM only for GPUs; storage fields only for storage types; memory type only for GPU/SSD.

## Diagnostic panel (final revision, `work/diagnostics.jsonl`)
- **Coverage and ids:** 2,367 membership rows; 0 unresolved ids; 0 duplicate assignments; source coverage 1.0 for all four sources.
- **Consistency across files:** fused `_id` set equals membership cluster ids. The correspondences equal exactly the cross-source pairs implied by the clusters (2,255) and all lie inside the candidate set.
- **Singletons:** 23.8% of clusters. The share of records that are singletons per source is 0.09 / 0.11 / 0.09 / 0.09, so no source is isolated.
- **Schema validity:** 0 violations (enum, pattern, ranges, lengths).
- **Required fields:** `product_type` is empty for the 6 out-of-domain entities (monitors, kettle, RAM, dock); `brand` is empty for 25 generic no-name listings.
- **Fused density:** brand 0.97, model 0.88, model_number 0.42, product_type 0.99, chipset 0.23 (0.78 of GPU rows), vram 0.29, storage 0.69, read 0.14, write 0.12, bus 0.81, interface 0.47, storage_connection 0.46, memory 0.19, colour 0.12, form factor 0.48, dimensions/weight 0.03–0.05.
- **Fusion provenance counts** (cells, including n/a): vote 11.8k, medoid 4.8k, none 4.6k, rejected as corrupt 154, reordered 48, typo-repaired 13.

## Iterations (see `work/step_log.jsonl`)
1. **Capacity parser:** in-cluster capacity conflicts traced to `6G SAS`, `6Gb/s` and `146-GB` parsing. Fixed in `s2`, and all downstream stages were rerun.
2. **Brand fusion:** 31 empty brands and corrupted strings (`A5U5`, `M51`) trusted by frequency. Switched to alias- or title-supported trust; the Phoenix product-line cue now maps to ASUS.
3. **Categorical trust rule:** replaced the frequency rule with token, boundary and vocabulary checks after auditing the repaired and rejected values.

Earlier coherent revisions are kept under `work/revisions/`.

## Known weaknesses and uncertainty
- Matching quality depends on my direct judgments. The hardest cases are OC vs non-OC and "Boost" GPU listings, colour-less portable SSD listings, regional SKU variants, HP spare vs option part numbers, and generic Exos/IronWolf listings without a model number.
- Grouping regional SKUs and OEM-rebadged drives follows my reading of "one configuration". A stricter definition would split them.
- Fused text choices (title, model spelling) are heuristic.
- Vote-based categorical choices cannot tell which of several legitimate representations the evaluator expects (e.g. `SATA III` vs `SATA 6Gb/s`).
- Dimensions and weight are very sparse and their units are ambiguous; the conversions are heuristic.
- The storage field sometimes contradicts the title (e.g. `450` for a 4TB drive). The vote uses both sources but can fail on 1–2 member clusters.
- The review snapshot is tied to the `s2` features at review time. If normalization changes, `s5` still uses the snapshot's block membership, so the review decisions stay reproducible, but new records would need re-review.
