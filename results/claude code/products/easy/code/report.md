# Products integration — method report (revision r4)

**No labeled evaluation and no supervised training were performed.** There were no reference matches, labels, judges or
teacher services; no model was trained or fine-tuned. All numbers below are label-free structural diagnostics computed
from the saved artifacts (`work/diagnostics.jsonl`, `work/state/*_stats.json`). They are not precision, recall, F1,
pair completeness or fusion accuracy.

Pipeline: `work/s1_schema.py` → `s2_normalize.py` → `s3_blocking.py` → `s4_matching.py` (evidence in `pairfeat.py`) →
`s5_cluster.py` → `s6_fusion.py` → `s7_export.py` → `s8_diagnostics.py`. `work/rebuild.sh` runs all of them
deterministically (no randomness, no network). Running it in a fresh copy of the workspace with no `submission/`
directory reproduced all five data files byte-for-byte.

## Data observations
- There are 4 sources (806/806/784/695 rows) with the same 25 columns; only the URL column name differs
  (`product_url`, `productUrl`, `Link`, `link`).
- About 40% of the rows are outside the four target product types (MONITOR, RAM, lamps, keyboards…). Most of these
  look synthetic and appear once in each source with identical text. The real WDC-style offers (HDD, SSD, GPU,
  USB sticks) are noisy:
  - spaces injected inside tokens ("Bl ack", "US D", "530. 0");
  - `storage_gb` given in TB (e.g. 4.0 for a 4 TB drive), and VRAM in MB;
  - locale-formatted prices;
  - brand-field variants ("WD", "WD_Black", "Western Digital");
  - the same model number shared by several capacity variants, and a few records whose model number or brand
    belongs to a different product.

## 1. Schema matching
- Every source column maps by name to the target attribute of the same name (score 1.0). The URL columns map to
  `url` (score 0.95), because their values are offer links.
- Every target attribute has contributors in all sources, and every populated source column is mapped.
- Output: `sm_mapping.csv`.

## 2. Normalization (raw values kept as `raw_*` in `work/state/norm.pkl`)
- **Numbers:** thousands separators and injected spaces are removed, and locale prices are parsed. Numeric parse
  failures went from 21 prices to 0; 1 `vram_gb` value is "n/A". Zeros are treated as missing.
- **Storage units:** `storage_gb` is converted TB→GB only when the record's own title states the TB capacity
  (63 conversions). VRAM ≥ 512 is treated as MB and divided by 1024.
- **Product type:** upper-cased; "USB flash drive"/"Flash Drive" → USB_STICK and EXTERNAL_SSD → SSD.
- **Brand key:** lower-cased alphanumerics with aliases (WD*/Western Digital, A-DATA, HPE/HP, Kingston Technology,
  TeamGroup, ThinkPad→Lenovo). Brands are also detected in the title using a vocabulary of brand keys seen at least
  3 times.
- **Title-derived matching keys:**
  - capacity (a single unambiguous TB/GB mention, excluding Gb/s);
  - GPU VRAM (GB, G or MB);
  - GPU chip (e.g. RTX2060SUPER, including "2060S" and a detached "SUPER");
  - HDD RPM;
  - identifier codes: alphanumeric tokens of at least 5 characters from model number, model and title, excluding
    unit and interface-generation tokens;
  - short model tokens (t5, sn550, 970, 27…);
  - 2.5″/3.5″ form-factor mentions.
- Taxonomy plan and per-source coverage: `work/taxonomy_plan.json`, `work/taxonomy_coverage.csv`. Only the
  `product_type` enum is exhaustive. The memory and interface vocabularies are non-exhaustive, so unmapped values
  are kept.

## 3. Blocking
- The candidate set is the union of three methods:
  - (a) equal normalized model number;
  - (b) shared identifier code with document frequency ≤ 16;
  - (c) character 2–4-gram TF-IDF (title + brand): top-5 neighbours of each record in every source, cosine ≥ 0.35.
- Result: 32,627 pairs out of 4,775,595 possible, a reduction ratio of 0.9932.
- Only one record has no candidates. The most candidates for one record is 92. 21% of the pairs are within one
  source; those are only used for exact model-number duplicates.
- The exported `blocking/candidates.csv` also contains 8 within-cluster pairs that were joined only transitively,
  so every correspondence is a member of the candidate set.
- This coverage is not pair completeness.

## 4. Entity matching (rules, `s4_matching.py`)
**Hard conflicts** reject a pair outright:
- different model numbers from the same numbering scheme. Containment (e.g. `…-WESN` suffixes) or one record
  mentioning the other's number is not a conflict. For the four hardware types only, a mostly-digit vendor part
  number vs a model-name code is treated as two different schemes.
- title/field capacity differs by more than 3%;
- GPU VRAM or GPU chip differs;
- HDD RPM differs by more than 15% (sellers disagree on 5400 vs 5900 "class" drives);
- brands are disjoint, unless both titles share a part number (OEM-rebadged drives).

**Accept rules:**
- `id`: equal model number, or a shared rare code with title cosine ≥ 0.25 and no contradicting title codes.
- `text` (for records without identifier evidence): mean word/char TF-IDF cosine ≥ 0.65, with none of these:
  - type conflict or code contradiction;
  - a *modifier* word on one side only (Pro, Plus, Super, Ti, XT, Extreme, Mini, Slim, Hub, Go, Touch…);
  - different product-line or colour words on the two sides (Strix vs TUF, Dual vs Turbo);
  - disjoint short model tokens (T5 vs T7, 27 vs 34);
  - SAS vs SATA, or 2.5″ vs 3.5″.

  A lower band, cosine 0.50–0.65, is accepted only with positive capacity (or VRAM + chip) agreement, compatible
  brands, and no two-sided brand-specific word or model-token difference.
- Records from the same source are linked only on an exact model-number match.

**Why these thresholds:**
- Cosine 0.65 and 0.50 were chosen by reading pairs on both sides of each value. Below about 0.5 almost all pairs
  are different variants of one series. Between 0.5 and 0.65, true pairs mostly carry explicit capacity agreement.
- Each conflict rule was added after inspecting a concrete failure pattern, such as series codes (SX8200, SA500)
  bridging capacity variants, or synthetic monitor variants whose titles differ only in size.

**Decision counts:** 3,884 `id` and 406 `text` pairs accepted; 26,816 rejected by a conflict (most often
model-number 19,967 and capacity 9,299); 46 same-source pairs rejected.

## 5. Clustering (`s5_cluster.py`)
- Accepted edges are processed strongest first: model-number equality, then shared code, then text, then by cosine.
- Two clusters merge only if no pair of their members has a hard conflict (computed on the fly, including pairs
  that were not candidates). Two records from the same source must also be directly linked.
- 38 merges were refused for a conflict and 108 for the same-source rule. This prevents transitive bridges.
- Unmatched records are kept as singletons.
- Result: **905 entities**. Cluster sizes: 110 of size 1, 44 of size 2, 115 of size 3, 633 of size 4, 2 of size 5,
  1 of size 6.
- 635 clusters contain all 4 sources; the singleton share is 12.2%; 4 clusters contain two records from one source
  (exact model-number duplicates).
- `correspondences.csv` lists every cross-source pair inside a final cluster (4,217 pairs; none within one source).

## 6. Fusion (`s6_fusion.py`)
Each attribute is resolved only from the cluster's own records:
- **Categorical:** vote over normalized equivalence keys, and output the most frequent member spelling within the
  winning group. Ties go to the spelling that is more frequent across the whole dataset, then alphabetical.
  Injected spaces are undone only when the de-spaced spelling is attested in the column.
- **Brand:** case-insensitive vote.
- **`bus_type`:** voting classes that merge USB-IF renames, SATA III = 6Gb/s, SAS 6/12, and PCIe generation + lanes.
- **`interface_type`:** voted by family.
- **`memory_type`:** cased to the GPU memory taxonomy.
- **`form_factor`:** 2.5″ variants → "2.5-inch", 3.5″ → "3.5-inch" (the schema's families), low-profile →
  "Low Profile". **`storage_connection_type`:** "<size>-inch SATA/SAS".
- **Numbers:** the value with the most members within 2%. For `storage_gb`, title capacities add half-weight
  support. Ties go to the value closest to the median. Out-of-range values become empty.
- **Title, description, price, currency, url:** taken from the medoid record (most central title), so they
  describe one real offer.
- `product_type` is voted on the canonical labels. Out-of-scope entities (324 rows: MONITOR, RAM…) keep their
  source category rather than being forced into the 4-value enum. This is a deliberate schema deviation, reported
  as a violation below.
- Provenance per cell is in `work/state/fusion_provenance.csv`.

## Label-free diagnostic panel (r4, re-read from `submission/`)
| check | value |
|---|---|
| source records / membership rows / missing / unresolved ids | 3091 / 3091 / 0 / 0 |
| correspondences in candidate set | 4217 / 4217 |
| correspondences inside one cluster | 4217 / 4217 |
| fused `_id` set equals membership `cluster_id` set | yes |
| fused header matches target attributes | yes |
| schema violations | product_type 324 (out-of-scope categories); no others |
| fused value found verbatim among its own members' raw values (whitespace/case-insensitive) | brand, model, model_number, chipset_name, bus_type, color: 1.00; form_factor 0.51 (canonicalized by design) |
| memory_type in GPU memory taxonomy | 0.97 |
| output density | brand .99, model .99, model_number .83, storage_gb .50, chipset .42, bus_type .58, form_factor .32, vram .14 |

## Rejected approaches and known weaknesses
- **Rejected approaches:**
  - A pure title-similarity threshold of 0.8 still merged edition variants such as "Gaming" vs "Mini", and
    "12G 15K" vs "6G 10K" drives.
  - Disabling model-number conflicts for any pair of dissimilar numbers (revision r2) merged synthetic monitor
    variants into clusters of 8.
  - Requiring every rare word of the shorter title to appear in the longer one rejected many true pairs over
    generic words ("series", "server", "placa").
- **Unresolved uncertainty:**
  - Some suffix-variant SKUs are treated as the same product because one model number contains the other
    (e.g. GT710-SL-2GD5 vs -CSM).
  - A few records' model numbers may belong to a different product; when the evidence is only that number,
    those records follow it.
  - Low-information titles without model numbers ("Seagate Barracuda 500GB") may stay singletons or join the
    wrong edition.
  - Colour variants are only separated when both titles name a colour.
  - Weight and dimension fields mix units in a few clusters (e.g. 0.008 vs 8); the vote does not convert them.
  - Output formats for `bus_type`, `color` (several languages) and `interface_type` follow the member majority,
    which may not match the reference's canonical spelling.
  - How the out-of-scope categories should be typed is uncertain.
