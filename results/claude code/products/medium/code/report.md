# Products integration: method report (revision r2)

**No labeled evaluation and no supervised training were done.** No labeling service, teacher, gold
file or pseudo-labels were used, and no matcher was trained or fine-tuned. Every number below is a
structural diagnostic that I measured on my own artifacts. None of them is an accuracy, precision,
recall, F1 or pair-completeness score.

Pipeline: `work/s1_schema.py` → `s2_normalize.py` (+ `s2b_taxonomy.py`) → `s3_blocking.py` →
`s4_match.py` (shared rules in `pairlib.py`) → `s5_cluster.py` → `s6_fusion.py` → `diagnostics.py`.
`work/rebuild.sh` runs the whole chain deterministically (about 35 s, no network, no randomness).
The configs are `work/match_config.json` and `work/overrides.json`.

## 0. Data observations that shaped the design
- There are 4 CSVs (811/811/780/698 rows). All share one 25-column layout under renamed headers
  (e.g. `brnd`, `prd_ttl`, `mno`, `stg`).
- Values carry heavy synthetic noise:
  - OCR character swaps: `329.O0`, `Kingzton`, `mon8tor`, `ST10000NEp004`.
  - Mixed locale number formats: `2.000,0` vs `2,000.0`.
  - Unit slips: storage `10.0` for a 10 TB drive.
  - Wrong or seller brand fields, and shuffled `model` strings.
- About 30% of the records are fictional catalogue items: monitors, RAM, recorders, lamps, plus some
  fictional SSDs, HDDs and USB sticks. Each appears as exactly one copy per source. Look-alike
  products deliberately share model numbers across fictional brands (e.g. `AF27Q-75` is used by both
  "Aurora Field" and "Aurelia Frame").
- The remaining records are real web offers (WDC-style) for GPUs, SSDs, HDDs and USB sticks. They
  include several offers of the same product inside one source.

## 1. Schema matching (`sm_mapping.csv`)
- The column order matches across sources, but I did not rely on that. I checked each mapping by
  meaning and example values: prices with currencies, GB-scale storage values, `GDDR6`, `SATA III`,
  `3.5"` and so on.
- The mapping is 1:1 for all 25 columns in each source. `product_url`/`productUrl`/`Link`/`link` map
  to `url`. Score is 1.0 because these are rule decisions.
- No target attribute is missing from any source, and no source column is left unmapped.

## 2. Normalization (`s2_normalize.py`; raw columns kept, derived columns prefixed `n_`/`f_`)
- **Numbers.** Locale-aware parsing: the last separator is the decimal mark, and repeated separators
  are thousands marks. OCR repair: O/o→0, l/I→1, S→5 next to digits. `N/A`-style tokens become null.
  Parse rates: price 99.9%; speeds and dimensions 100% (`state/norm_diag.json`).
- **Product type.**
  - OCR and descriptive aliases map to the exhaustive schema enum (`SS D`, `HXD`,
    `Solid State Drive`, `USB Flash Drive` …).
  - Missing types are inferred from title keywords.
  - Out-of-scope categories keep their own canonical upper-snake name for matching.
  - Coverage per source is in `work/taxonomy_coverage.csv`. About 336 records per source are
    genuinely out of scope, which is why only 50–58% of product_type values are canonical.
- **Capacity.**
  - The value is parsed from title and model (`TB/To/GB/Go`, and a bare `G` only when ≥16 and not
    glued to a letter). The source field is used as a cross-check.
  - Unit slips are repaired: a field ×1000 that equals the title value, and HDDs under 20 GB or SSDs
    under 4 GB are treated as TB.
  - Capacity is known for 97.7% of storage records.
- **GPU attributes.**
  - Chip is taken from the title first, then the chipset field. It is normalized (e.g. `RTX 2070 SUPER`
    or `RX 5700 XT`, with SUPER added even when written apart from the number).
  - VRAM comes from the field or the title (MB values ≥128 are converted to GB).
  - Chip is known for 95%, VRAM for 93%.
- **Brand for matching.** The earliest real brand mentioned in the title wins over the (often seller)
  brand field. Aliases are applied (WD→Western Digital, HPE→HP …). Typos are repaired against
  frequent spellings. Brands missing on GPUs are inferred from board sub-lines (TUF→ASUS,
  Windforce→Gigabyte …).
- **Identifier codes.** Alphanumeric tokens from title, model and model_number, cut at `/` so that
  regional suffixes drop. Spec-like strings that look like part numbers are filtered out, e.g. `RX5700XT`,
  `BASE1350MHZ`, `8GDDR6`, `PCIEXPRESSX16`.
- **Taxonomies.**
  - `taxonomy_plan.json` records the policy. The product-type enum is exhaustive.
  - The GPU-memory and storage-interface taxonomies are non-exhaustive. They drive matching
    conflicts only. Fused values keep the majority source spelling, and nothing is rewritten into a
    taxonomy form that the sources do not contain.

## 3. Blocking (`s3_blocking.py`; full set in `blocking/candidates.csv`)
The candidate set is the union of four generators:
1. A shared code token (codes occurring in more than 40 records are skipped).
2. An equal normalized model_number.
3. TF-IDF character 3–5-gram nearest neighbours on title+model (k=15, cosine distance < 0.6).
4. A spec key: (type, brand, capacity) for storage, (brand, chip) for GPUs, with blocks capped at 60.

Diagnostics (`state/blocking_diag.json`):

| Measure | Value |
|---|---|
| Candidate pairs | 25,701 of 4.80 M possible (reduction ratio 0.9947) |
| Cross-source share | 78.7% |
| Records with no candidate | 1 (`products_3_2111408`, a unique Dell 1.8 TB drive) |
| Median / max candidates per record | 15 / 76 |
| Pairs only from TF-IDF | 13,663 |
| Pairs only from spec key | 6,410 |
| Pairs only from codes | 71 |
| Pairs only from model_number | 4 |

The exported file adds 27 within-cluster pairs that constrained clustering joined transitively. It
checked every such pair explicitly for conflicts, so every correspondence lies inside the candidate
file (verified: 100%). This candidate coverage is not pair completeness; without reference matches
that cannot be measured.

## 4. Entity matching (`s4_match.py`, `pairlib.py`)
The entity definition says a different specification means a different entity, so the matcher
combines **contradiction rules** with a **similarity score**.

**Hard conflicts** (never overridden):
- In-scope product type differs; for out-of-scope types, only unrelated categories count.
- Capacity differs by more than 2%.
- GPU VRAM or chip differs.
- GPU memory generation differs (GDDR5 vs GDDR6, where `D5`/`D6`/`DDR5` mean GDDR on GPUs).
- Both brands are fictional and the brand fields are clearly different.

**Soft conflicts** (overridden only by a shared part number of 8+ characters):
- **Brand:** different brands when at least one is real (OEM relabels such as Dell/EMC/EQL drives
  share the Seagate part number and are allowed through the code).
- **Model tokens:** disjoint digit-bearing model tokens (T5/T7, SX6000/SX8200, N300/X300, S5/S11).
- **Variant words:** EVO/PRO/QVO mismatch; Plus, Touch and Pro (and Heatsink for WD) present on one
  side of a shared line only.
- **Product lines:** Seagate/WD lines; WD colours count as lines only for WD. For USB sticks, lines
  and variants such as Blade/Glide/Ultra/Flair/Go/Duo/Vault. Also generation markers (G2, G3).
- **Storage specs:** form factor (M.2 vs 2.5" vs 3.5") and interface family (SATA/SAS/NVMe/USB);
  for HDDs also rpm, SAS link speed (6G/12G) and 512n/512e.
- **GPU board:** sub-line (TUF/Phoenix/Strix/Dual/Windforce/Aorus/Gaming X/Nitro+/Pulse …), edition
  (OC `O6G` vs plain `6G` vs Advanced `A11G`), quad-HDMI boards, and Gigabyte "GAMING OC" vs "OC".
- **Part numbers:** same-family numbers (same 3-character prefix or the same letter/digit shape)
  that are not equal up to OCR noise. This is ignored when titles are near-identical (cos ≥ 0.95),
  because a single corrupted model number is then the likelier explanation.
- **Fictional-brand pairs:** model numbers must be equal up to at most 2 substitutions or a short
  suffix, and title-only cosine must be at least 0.75. Items with no brand, no capacity and no code
  need cosine ≥ 0.8.

**Score:**
max(word TF-IDF cosine on title+model, cosine on title only)
+ 0.5 if a code is shared
+ 0.3 if model_number agrees
− 0.3 if model_number differs and no code is shared
+ 0.25 for key agreement (same real brand, same capacity and a shared distinctive model token; for
GPUs, same chip, VRAM and a shared sub-line).

A pair is accepted at score ≥ 0.45 across sources and ≥ 0.8 within one source.

**Threshold rationale.** I first used 0.6. Inspecting the 0.45–0.6 band showed mostly true offer
pairs whose titles are cluttered with seller text; the errors in that band were the variant cases
that the conflict rules now catch. Below 0.45, pairs I inspected were mostly different products or
generic listings. Of the 25.7k candidates, 4,507 are accepted, 12.9k are rejected by a hard conflict
and 9.2k are conflict-free.

## 5. Refinement / clustering (`s5_cluster.py`)
- **Algorithm.** Accepted edges are processed in descending score order (deterministic ties). Two
  clusters merge only if **no cross pair**, candidate or not, has a hard or soft conflict. This
  replaced connected components, which produced 47- and 37-record chains mixing T5/T7/T7 Touch and
  860 EVO/PRO/QVO. Merges blocked by conflicts: 316 (listed in `state/rejected_merges.csv`).
- **Manual identity decisions** made during cluster review (`work/overrides.json`; direct decisions,
  not labels):
  - Merge the three ASUS GT710-4H quad-HDMI offers.
  - Keep HPE MSA 1.2 TB (787648-001) apart from HPE 872479-B21.
  - Keep Gigabyte GTX 1050 Ti OC (4GD) apart from OC Low Profile (4GL).
- **Same-source duplicates are allowed.** The real offers do contain repeated offers per source.

## 6. Fusion (`s6_fusion.py`; per-cell provenance in `work/state/fusion_provenance.csv`)
- **Categorical and text attributes** (brand, model, model_number, bus/interface/connection type,
  memory_type, colour, form_factor, chipset): majority vote over case- and punctuation-normalized
  keys. The representative is the most frequent raw spelling in the winning group. Ties go to the
  globally more frequent form, because rare forms are usually OCR-corrupted, then to lexical order.
- **product_type:** canonical enum by vote. For out-of-scope clusters, the most common source spelling
  (e.g. `MONITOR`).
- **Numeric attributes:** mode with 2% tolerance, ignoring values below the schema minimum (zeros =
  unknown).
  - storage_gb uses the per-record field value unless it contradicts the title (unit slip → title
    value).
  - vram uses the field, else the title value.
  - weights under 0.5 are treated as kg and converted (e.g. 0.008 → 8 g).
- **Type-specific attributes** are left empty where the schema's field applicability excludes them.
- **Title, description, price+currency and url** come from the cluster's medoid record, so price and
  currency stay paired.
- **brand** falls back to the title-detected brand for 7 clusters that have no brand field at all.
- **`id`** equals the cluster id (`_id`), which is the smallest member id prefixed `c_`.

## 7. Diagnostics (latest rows of `work/diagnostics.jsonl`, revision r2)
**Coverage and joins**
- All 3,100 source records are in `membership.csv`, with 0 unresolved ids and 0 source-label
  mismatches.
- 914 fused rows. Membership ↔ fused `_id` sets agree in both directions.
- 4,333 cross-source correspondences: 0 cross-cluster, 0 same-source, 100% inside the candidate file.

**Cluster structure**

| Cluster size | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| Count | 132 | 64 | 110 | 572 | 14 | 12 | 5 | 3 | 1 | 1 |

- Singleton share: 14.4%. Sources per cluster: 1→132, 2→67, 3→120, 4→595.
- 46 clusters contain same-source duplicates (180 records), all real multi-offer products.
- All 370 fictional-catalogue clusters (out-of-scope items plus fictional-brand SSD/HDD/USB) have
  exactly 4 members, one per source.

**Schema validity.** The only violations are 336 `product_type` values outside the enum. These are the
out-of-scope clusters, a deliberate choice: the self-check requires every record in membership, and
these items clearly form cross-source entities. Numeric ranges and the model_number pattern are all
valid.

**Density** (fused): brand 0.97, model 0.95, model_number 0.71, product_type 1.0, storage_gb 0.55,
chipset 0.37, vram 0.15, bus_type 0.52, interface_type 0.30, memory_type 0.11, colour 0.11,
form_factor 0.29, dimensions and weight 0.03–0.05. These track source sparsity; no values were
imputed.

**Provenance trace.** Every fused text value was checked for presence among its own members' raw
values. brand 99.2% (the 7 title-derived brands); all other checked attributes 100%.

## 8. Rejected approaches
- **Connected components over accepted pairs:** giant transitive chains (above).
- **Near-equality of codes by edit ratio:** it treated WD6003FFBX≈FZBX and ST2000NM0023≈0033 as the
  same part. It was replaced by OCR-only letter↔digit substitutions.
- **Colour words as product lines for every brand:** this split Samsung T5 colour offers. They are now
  WD-only.
- **Strict brand-field equality for fictional brands:** the brand field is noisy within one product's
  copies (e.g. "Nebula Forge" / "Nebula Foundry" / "Forge Nebula").
- **Title-derived filling of chipset or memory_type when no member states one:** skipped to avoid
  inventing values.

## 9. Known weaknesses and unresolved uncertainty
- **Colour variants.** Samsung T5/T7 and WD My Passport colours are merged into one entity when no
  conflicting part number separates them. Whether the benchmark treats colour as a separate
  configuration is unknown.
- **Generic offers.** Titles without a variant marker (e.g. "Asus GeForce GT 710 2GB") attach to
  whichever compatible cluster they match best. They may belong to a sibling variant.
- **Marginal decisions.** Some sub-line and edition decisions are uncertain, e.g. Gigabyte
  "OC" vs "GAMING OC", Low Profile vs standard, and Quadro P400 DVI vs miniDP bundles. A few noisy
  records (e.g. "Upgrade MacBooku … 860 EVO" whose model field says 850) may sit in the wrong cluster.
- **Out-of-scope rows.** These keep a non-enum `product_type`. If the evaluation expects them removed,
  their correspondences are extra pairs.
- **Fused representations.** bus_type, interface_type and similar keep the majority source spelling
  (e.g. `SATA 6Gb/s` vs `SATA III`). If the reference uses the other spelling, numeric-token
  comparison may disagree.
- **Unmeasurable accuracy.** Without reference data I cannot quantify precision or recall. The
  thresholds (0.45 / 0.8 / 0.75 / 0.95) and the rule set come from inspecting cases, not from an
  optimized metric.
