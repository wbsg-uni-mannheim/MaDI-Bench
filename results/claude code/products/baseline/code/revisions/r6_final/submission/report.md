# Products integration report (label-free baseline)

**No labelled evaluation was done and no supervised training was used.** No matcher was trained or fine-tuned, no pseudo-labels were created, and no labeling, judging or fusion-teacher service was called. Every rule and threshold below was set by hand after inspecting source records. All numbers in this report are structural diagnostics taken from the saved artifacts. They are not precision, recall, F1, pair completeness or fusion-accuracy scores.

Rebuild: `work/rebuild.sh` (s1 → s2 → s3 → s4 → s6 → s7). It is deterministic, makes no network calls and uses no model files. I checked it by running it in a fresh copy of the workspace with `submission/` and `work/state/` removed. It produced the same membership, correspondences, fused table, mapping and candidates (the comparison was order-insensitive and all five matched).

## Inputs
There are four JSON sources: dataset_1 (812 records), dataset_2 (812), dataset_3 (762) and dataset_4 (626), 3,012 records in total. All four have the same 25 fields under different column names. Record ids are the source `id` values, used unchanged. The `source` value is the file stem.

A key structural observation: dataset_1 and dataset_2 have **identical** per-type counts (GPU 231, HDD 253, SSD 246, USB_STICK 82). dataset_3 and dataset_4 have smaller counts for every type. This suggests that each entity has roughly one offer per source, always present in ds1 and ds2 and sometimes in ds3 and ds4. I used it as a *soft* structural prior: at most one record per source per cluster, built with one-to-one assignment. It is not a hard rule. A few same-MPN duplicates exist inside single sources (e.g. WD80PURZ twice in ds1), and the Exos X10/X16 case showed that the ds1 and ds2 records do not always belong to the same entity. Records are left unmatched whenever the evidence contradicts a match.

## 1. Schema matching (`work/s1_schema.py`)
The mapping was decided from the metadata descriptions plus value inspection, not from column position alone. All 25 source fields map one-to-one onto target attributes (id, brand, title, description, price, priceCurrency, url, model, model_number, product_type, chipset_name, vram_gb, storage_gb, read/write speed, bus_type, interface_type, width/length/height, weight, storage_connection_type, memory_type, color, form_factor). ds2's `depthMm` → `length_mm` has score 0.9 because the metadata says "length or depth". Every other mapping has score 1.0. No target attribute is left without a source column. The metadata warns that unit suffixes are not guaranteed, so units are handled in normalization.

## 2. Normalization (`work/s2_normalize.py`; coverage in `work/taxonomy_coverage.csv`)
Raw values are kept; canonical values go into `n_*` columns.
- **Capacity**: the field value is checked against capacities parsed from the title (TB/GB/To/Go, with bare G/T only for values ≥16). Where the field held TB as a bare number (e.g. 6.0 for a 6TB drive), it is multiplied by 1000 if the title confirms it. Binary 1024/2048/4096 GB are mapped to 1000/2000/4000. Resolution sources were: field=title 1899, field×1000=title 139, title override 11, description-backed 30, heuristic ×1000 1 (small value on an HDD/SSD, no text evidence), no value 42.
- **VRAM**: MB values of 256 or more are divided by 1024.
- **GPU chip**: a regex key is taken from the chipset field, falling back to the title (e.g. `nv2070super`, `rx5700xt`, `quadrop400`). The title key wins when it names a more specific variant of the field's chip (field "GTX 1650", title "GTX 1650 SUPER"). Fused output uses a canonical display name built from the key (e.g. "GeForce RTX 2070 SUPER").
- **Brand**: an alias table (WD → Western Digital, Hewlett Packard Enterprise → HPE, A-DATA → ADATA, Kingston Technology → Kingston, Exos/BarraCuda/IronWolf-as-brand → Seagate, …). When the brand field is empty, the brand is taken from the title if a known brand appears there. For match compatibility, HP/HPE and ADATA/XPG count as one family, and NVIDIA/AMD count as unknown (they are the chip vendor, not the board maker).
- **Taxonomies**: memory types use the GPU_Memory_Taxonomy variants (DDR5/DDR6 on GPUs → GDDR5/GDDR6). Bus types use Storage_Interface_Taxonomy spellings (SATA 6Gb/s, SAS 12Gb/s, PCIe 3.0 x4, USB 3.0 for 3.1/3.2 Gen 1, USB 3.2 Gen 2). Interface is reduced to its family (SATA/NVMe/SAS/USB/USB-C/…). Form factor maps to the schema's expected families (2.5-inch, 3.5-inch, M.2, mSATA, Low Profile, ATX, External). All of these vocabularies are non-exhaustive, so values outside them are kept. Color names in other languages are translated (Negru/Zwart/sort → Black, and so on). Currency symbols are mapped to ISO codes.
- **Product type**: 3 records had a null type. The type was inferred from title keywords.

## 3. Blocking (`work/s3_block.py`; full set in `submission/blocking/candidates.csv`)
Product type is a hard block. Within each type and source pair, the candidate set is the union of:
- the text top-10 in both directions (0.5 × char 2–5-gram TF-IDF + 0.5 × word TF-IDF over title + model + MPN);
- shared product-code tokens;
- brand-family + GPU-chip or brand-family + capacity blocks with at most 40 members.

The exported file also includes the 10 transitive pairs inside final clusters.

Diagnostics: 73,855 candidate pairs out of 955,554 same-type cross-source pairs, a 97.8% reduction against all cross-source pairs. Every record has at least one candidate, and every correspondence is in the candidate set. Adding the spec-key blocks raised ds1–ds2 pairing from 741 to 770. The added blocks surfaced true-looking partners in crowded families (for example, many GTX 1660 SUPER cards) that the text top-K had missed.

## 4. Matching (`work/s3_block.py` features, `work/s4_match.py` score)
**Score:** 0.5·char-cos + 0.5·word-cos, plus bonuses for identifier and spec agreement and minus penalties for contradictions:
- **Bonuses**: exact shared code +0.35, same MPN +0.25 (regional letter suffixes allowed, e.g. MZ-76E250B vs …BW), equal capacity, chip, model-number token or product line.
- **Hard penalties**:
  - capacity ≠ (−1.0; 2% tolerance)
  - chip ≠ (−1.0)
  - VRAM ≠ (−1.0)
  - brand-family ≠ (−1.0)
  - MPN from the same naming scheme ≠ (−1.0)
  - title product code from the same scheme ≠ (−0.8)
- **Soft penalties**: model-number token (860 vs 970, SN500 vs SN550; digit-core compatible), RPM, SAS generation, interface kind, title form factor (−1.0), exclusive product lines (WD Blue/Red/Purple…, Exos/IronWolf/Archive…, EVO/QVO/PRO, GPU board series Strix/Dual/TUF/Windforce/…), tier modifiers (Pro/Plus) and GPU memory type. Soft penalties are multiplied by 0.3 when an identifier matches exactly, because OEM rebadges such as Dell/EMC-sold Seagate drives and noisy RPM fields would otherwise block same-MPN matches.

Weights and thresholds were set by inspecting sorted pair lists near the threshold, not by optimizing any score.

## 5. Clustering (`work/s4_match.py`)
Clustering proceeds one source at a time, per type:
1. Seed clusters from ds1.
2. Assign ds2 to clusters by Hungarian assignment. Each record has a "stay unassigned" dummy valued at the threshold, so nobody is forced into a bad match; an earlier version without the dummy wrongly split good pairs.
3. Assign ds3, then ds4, the same way.

A record's affinity to a cluster is 0.5·mean + 0.5·max over its scored member pairs. It is vetoed to the minimum if any member pair is below 0, so one weak bridge cannot pull a record into a cluster. Assignment requires affinity ≥ 0.2; I chose this value after inspecting the 0.2–0.45 band, which was mostly correct matches with short or sparse titles.

A final leftover pass attaches a singleton to a cluster that lacks its source only if every member pair is a scored candidate, has no contradiction flag of any kind, and scores at least 0.1. At 0.0, 2 of the 5 merges I inspected were wrong, so I raised it to 0.1.

Correspondences are all cross-source pairs within each final cluster. Scores are a logistic transform of the rule score, and the 10 transitive-only pairs get 0.5.

## 6. Fusion (`work/s6_fusion.py`)
Values are resolved attribute by attribute, using only the cluster's own members:
- **Categorical values**: majority vote over canonical values, with a deterministic tie-break on source order (ds1 < ds2 < ds3 < ds4). For bus_type, ties go to the more specific value.
- **Numbers**: groups of values within 2% of each other; the largest group wins.
- **Brand on GPUs**: ignores NVIDIA/AMD when a board-partner brand is present.
- **chipset_name**: the canonical display name of the winning chip key.
- **title, description, price, currency, url**: taken together from the medoid member (the one with the highest average title-token overlap), so they come from one coherent offer.

Field-applicability rules follow the schema (chip/VRAM only for GPUs, storage fields only for storage types). Values outside a schema range, and model numbers that fail the schema pattern, are left empty. Missing values are never guessed, and values are never enriched from outside the cluster; the only title-derived value is the capacity check described in section 2.

## Diagnostics (final revision; `work/diagnostics.jsonl`)
- **Records**: 3,012 of 3,012 in membership. There are 0 unresolved ids, 0 clusters without a fused row and 0 fused rows without members.
- **Entities**: 881. Cluster sizes: 1 → 50, 2 → 84, 3 → 194, 4 → 553. Singleton share is 5.7%, and no cluster has more than one record from the same source.
- **ds1–ds2 pairing**: 779 clusters contain both ds1 and ds2 (the structural prior suggests close to 812).
- **Contradiction flags on accepted direct pairs** (3,984 correspondences):
  - capacity 1 (source typo: title says 32GB but the part number says 016G)
  - chip 0, VRAM 0, MPN 0
  - brand 23 (all OEM rebadges sharing the drive maker's MPN)
  - model token 13, RPM 9, SAS generation 11, memory type 5, interface 5
  - exclusive product line 3, code scheme 2
- **Within-cluster disagreement** after normalization: capacity 1.0%, chip 0%, brand family 0.5%, bus 33% (mostly granularity such as "PCIe 3.0" vs "PCIe 3.0 x4").
- **Fused density**: brand 0.99, product_type 1.0, model 0.97, model_number 0.73, storage_gb 0.71 (about the share of storage entities), chipset/VRAM 0.28 (about the share of GPU entities), bus 0.95, form factor 0.58, color 0.22.
- **Validity**: 0 schema range or enum violations and 0 applicability violations. Every fused brand, capacity, VRAM, model_number and bus_type value is found among its own members' values.

## Revisions (`work/revisions/r1…r5`, step log in `work/step_log.jsonl`)
- **r1**: first end-to-end run, 931 entities.
- **r2**: MPN containment compatibility, RPM parsing fix, cluster veto, assignment dummy-column fix, chip canonicalization and spec-key blocking.
- **r4**: OEM softening, title-only form factor, DDR3 = GDDR3, leftover pass.
- **r5**: hard MPN and code-scheme conflicts.
- **Final**: type inference for M.2/NVMe/mSATA titles, which removed the one null-type entity.

## Known weaknesses and unresolved uncertainty
- **Generic no-brand USB sticks** ("OTG pen drive 4GB…128GB") are essentially unidentifiable. Their groupings are low-confidence.
- **Listings that only say a family** (e.g. "HP 600GB SAS 10K", "WD 1TB SATA3 64MB") may be attached to the wrong specific SKU when several variants of that family exist.
- **Regional or revision MPN variants** (e.g. ST3000VX009/010, WD80PURZ/WD82PURZ) are treated as different entities. The reference may group them differently.
- **The one-per-source assumption** is a structural hypothesis inferred from the counts. Where the reference puts two same-source offers into one entity, my output necessarily splits them.
- **Fusion targets**: the exact target spellings of the reference (e.g. "M.2" vs "M.2 2280", "USB 3.1" vs "USB 3.0") are unknown. I chose taxonomy-aligned canonical forms. Dimension and weight fields are rare, and their units are only range-checked.

**Rejected approaches**: I did not fall back to connected components over pairwise thresholds, to avoid transitive over-merging. I did not use embeddings, because none were needed and they could not be rebuilt offline without a cache. I did not fill values from titles beyond the capacity check.
