# MaDI-Bench Use Cases

This directory holds the 20 integration tasks of MaDI-Bench: five domains, each with one **base** task and an **easy**, **medium**, and **hard** variant.

```
use cases/<domain>/<base|easy|medium|hard>/
```

The base task is the real-world integration problem. Each variant keeps the same problem and changes how hard its steps are, using the difficulty knobs described in [`../knobs/`](../knobs/) and in Section 4 of the paper. How to evaluate a system on a task, and which file is the gold standard for which step, is described in the [top-level README](../README.md#evaluating-a-system).

All numbers below were recomputed from the files in this directory; where the paper reports the same quantity (Tables 1, 2, and 12), they agree with it.

**Contents:** [Overview](#overview) · [Labeled resources](#labeled-resources) · [Task folder layout](#task-folder-layout) · [Entity matching files](#entity-matching-files) · [Fusion files](#fusion-files) · [Games](#games) · [Companies](#companies) · [Music](#music) · [Products](#products) · [Scientific Papers](#scientific-papers)

## Overview

| Domain | Sources | Source format (base / variants) | Target attributes | Rows: base | easy | medium | hard |
|---|---:|---|---:|---:|---:|---:|---:|
| [Games](#games) | 3 | CSV / CSV | 10 | 74,951 | 78,461 | 75,261 | 71,078 |
| [Companies](#companies) | 3 | CSV / CSV | 8 | 14,016 | 14,022 | 14,332 | 14,180 |
| [Music](#music) | 3 | CSV / CSV | 8 | 37,255 | 39,698 | 37,335 | 35,169 |
| [Products](#products) | 4 | JSON / CSV | 24 | 3,012 | 3,091 | 3,100 | 2,367 |
| [Scientific Papers](#scientific-papers) | 3 | JSON Lines / CSV | 11 | 182,059 | 182,213 | 182,067 | 136,876 |

Target attributes are counted without the identifier `id`. Rows are the records of all sources of a task; Table 12 of the paper gives them in thousands. In the per-source tables below, *density* is the share of filled cells over a source's attributes without its identifier, as in Table 1 of the paper: empty values, empty lists (`[]`), and placeholder strings such as `None` and `null` count as missing (the full rule, which also counts zero durations as missing, is implemented in [`reproduction/tables/table12/source_density.py`](../reproduction/tables/table12/source_density.py)).

## Labeled resources

### Base tasks

These are the counts of Table 2 of the paper:

| Domain | Normalization val | Normalization test | EM source pairs | EM train | EM val | EM test | EM test matches | Fusion val | Fusion test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Games | 202 | 182 | 2 | 934 | 234 | 739 | 30.0 % | 775 | 793 |
| Companies | 91 | 92 | 2 | 1,971 | 948 | 599 | 29.4 % | 614 | 610 |
| Music | 84 | 80 | 2 | 36,658 | 17,010 | 2,000 | 33.3 % | 727 | 726 |
| Products | 235 | 250 | 3 | 4,484 | 599 | 599 | 25.0 % | 819 | 836 |
| Scientific Papers | 281 | 248 | 2 | 10,666 | 2,666 | 13,332 | 25.0 % | 902 | 922 |
| **Total** | **893** | **852** | | **54,713** | **21,457** | **17,269** | | **3,837** | **3,887** |

- **Normalization:** labeled cells (source value and its correctly normalized form) in `input/normalization/validation.csv` and `test.csv`, 1,745 in total.
- **Entity matching (EM):** labeled record pairs per split, summed over the source pairs, 93,439 in total; "EM test matches" is the share of matching pairs in the test split. The splits are disjoint at the pair level. The test splits were verified by hand, except for Scientific Papers (matches derived from DOIs) and Products (matches taken from WDC Products). Products counts distinct pairs: its files contain three duplicate rows (training 4,486 rows, test 600 rows).
- **Fusion:** each split has 100 entities whose values human verifiers determined by web research in sources outside the task, 1,000 entities in total. The columns give the graded attribute values (non-empty gold cells over the target attributes), 7,724 in total; the paper rounds this to close to 8,000.

### Variants

| Task | Normalization val | Normalization test | EM train | EM val | EM test |
|---|---:|---:|---:|---:|---:|
| Games easy | 184 | 215 | 891 | – | 718 |
| Games medium | 235 | 270 | 1,056 | – | 736 |
| Games hard | 201 | 228 | 743 | – | 737 |
| Companies easy | 231 | 243 | 1,968 | 948 | 599 |
| Companies medium | 170 | 157 | 1,971 | 948 | 599 |
| Companies hard | 152 | 156 | 1,503 | 800 | 527 |
| Music easy | 71 | 92 | 36,612 | 17,009 | 1,998 |
| Music medium | 204 | 216 | 36,658 | 17,010 | 2,000 |
| Music hard | 152 | 162 | 24,566 | 11,393 | 1,587 |
| Products easy | 382 | 352 | 2,442 | 517 | 517 |
| Products medium | 442 | 442 | 2,558 | 524 | 518 |
| Products hard | 411 | 437 | 3,644 | 537 | 539 |
| Scientific Papers easy | 264 | 264 | 10,584 | 2,659 | 13,301 |
| Scientific Papers medium | 284 | 272 | 10,665 | 2,666 | 13,332 |
| Scientific Papers hard | 180 | 192 | 6,262 | 1,563 | 10,061 |

- **Normalization:** each variant has its own validation and test set, built on the variant's source values.
- **Entity matching:** the pairs of the variant's `*_corner_filled.csv` splits, summed over the source pairs (see [Entity matching files](#entity-matching-files)). The Games variants have no EM validation split.
- **Fusion:** the variants keep the 100 validation and 100 test entities of their base task and the same graded values (the counts of the base table above).
- **Schema matching:** each variant has its own gold mapping, `input/schemamatching/sm_mapping.csv`, because the variants rename the source columns.

## Task folder layout

The base tasks and the variants differ in a few places:

```
use cases/<domain>/base/
├── input/
│   ├── data/              source tables, and <source>_metadata.json per source (a schema.org
│   │                      Dataset description: provenance, columns, publication date)
│   ├── schemamatching/    target_schema.json, sm_mapping_gold.json (gold mapping), taxonomy CSVs
│   ├── entitymatching/    labeled record pairs per source pair: training, validation, test
│   ├── normalization/     validation.csv, test.csv
│   └── fusion/            validation and test gold, *_better_readability.csv, GOLD_VERSION, README.md
├── <domain>_workflow.ipynb    the P1 notebook (Products and Papers: <domain>_workflow_minimal.ipynb)
└── output/                P1's outputs, among them the fused table (silver reference) and the
                           normalized source tables; see output/README.md

use cases/<domain>/<easy|medium|hard>/
├── config/difficulty.yaml knob levels, knob order, and seed of the variant
├── input/
│   ├── data/              source tables (CSV; no metadata files)
│   ├── schemamatching/    target_schema.json, sm_mapping.csv (gold mapping of the renamed columns), taxonomy CSVs
│   ├── entitymatching/    *_corner_filled.csv (the variant's splits), *_baseline_pruned.csv, and copies of the base splits
│   ├── normalization/     as in the base task, built on the variant's source values
│   └── fusion/            as in the base task
└── output/                the generator's logs: provenance/ (the changes of each knob) and baselines/
                           (measurements of the base task and the realized knob levels)
```

- **Source tables:** the first column of every source table is its record id. Most ids carry a source prefix (`metacritic_10077`, `mbrainz_1`, `dblp-00000`); the Companies Forbes and DBpedia ids are URLs; the Products base sources use plain numeric ids, which the variants prefix as `products_<N>_<id>`.
- **Target schema:** `target_schema.json` is a JSON Schema that gives each attribute a type and a value constraint, for example a range, a pattern, a list of allowed values (`enum`), or a taxonomy. A taxonomy-bound attribute names the taxonomy file (`x-pydi-taxonomy`) and the taxonomy column whose values the output should use (`x-pydi-taxonomy-column`); `x-pydi-taxonomy-exhaustive: false` marks taxonomies that do not list every possible value.
- **Normalization sets:** each row of `validation.csv` and `test.csv` names a source cell (`source`, `source_id`, `source_column`, target `attribute`), its `raw_value`, the `expected_value`, a `category` (`identity`, `normalization`, or `knowledge`), and the `rule` that recovers it.
- **Readable fusion files:** next to each fusion split, a `*_better_readability.csv` file shows each entity's key (`id`; Products and Scientific Papers: `source_ids`) and its gold values in the target-schema columns, in schema order, without provenance. The original files remain the gold standard and, where present, keep the provenance of the values.
- **Products** has a folder `base/utils/` with the notebooks used to prepare the task from WDC Products.

## Entity matching files

| Domain | Source pairs | Base task: splits | Variants: splits |
|---|---|---|---|
| Games | DBpedia–Sales, Metacritic–DBpedia | `dbpedia_2_sales_{train,val,test}.csv`, `metacritic_2_dbpedia_{train,val}.csv`, `dbpedia_2_metacritic_test.csv` | `dbpedia_2_sales_{train,test}_corner_filled.csv`, `metacritic_2_dbpedia_{train,test}_corner_filled.csv` |
| Companies | Forbes–DBpedia, Forbes–FullContact | `forbes_2_dbpedia_{train,val,test}.csv`, `forbes_2_fullcontact_{train,val,test}.csv` | `forbes_2_dbpedia_{train,val,test}_corner_filled.csv`, `forbes_2_fullcontact_{train,val,test}_corner_filled.csv` |
| Music | MusicBrainz–Discogs, MusicBrainz–Last.fm | `musicbrainz_2_discogs_{train,val,test}.csv`, `musicbrainz_2_lastfm_{train,val,test}.csv` | `musicbrainz_2_discogs_{train,val,test}_corner_filled.csv`, `musicbrainz_2_lastfm_{train,val,test}_corner_filled.csv` |
| Products | Dataset 1–2, 1–3, 1–4 | `prod1_to_prod{2,3,4}_{train,val,test}.csv` | `products_1_2_products_{2,3,4}_{train,val,test}_corner_filled.csv` |
| Scientific Papers | DBLP–Crossref, DBLP–OpenAlex | `dblp_crossref_{train,val,test}.csv`, `dblp_openalex_{train,val,test}.csv` | `dblp_2_crossref_{train,val,test}_corner_filled.csv`, `dblp_2_open_alex_{train,val,test}_corner_filled.csv` |

**Evaluation splits.** The test files above are the evaluation sets of blocking and entity matching: the base task's `*_test.csv` and the variant's `*_test_corner_filled.csv`. The variant folders contain two more families of files, which are not evaluation sets:

- `*_baseline_pruned.csv`: the base task's pairs whose records survive in the variant;
- the plain-named files (for example `forbes_2_dbpedia_test.csv`): copies of the base task's splits. They may reference records that the variant dropped (up to 141 of the 200 pairs of a Products test file).

The `*_corner_filled.csv` splits are the pruned splits refilled with positive and negative corner cases (record pairs close to the decision boundary between matches and non-matches); every `*_test_corner_filled.csv` file references only records of the variant. The Games variants have no validation split; their plain-named training files combine the base task's training and validation pairs.

**Formats.**

| Files | Header | Columns | Labels |
|---|---|---|---|
| Games, Companies, Music: base splits and their copies in the variants | none | `id1, id2, label` | `TRUE`/`FALSE`, `True`/`False`, or `true`/`false` (Games: `TRUE`/`FALSE`; Companies and Music mix the spellings across files) |
| Products base | yes | `id1, id2, label` | `1`/`0` |
| Products variants: copies of the base splits | none | `id1, id2, label` (ids with the `products_<N>_` prefix) | `true`/`false` |
| Scientific Papers: base splits and their copies in the variants | yes | `id_dblp, id_crossref, label` or `id_dblp, id_openalex, label` | `1`/`0` |
| All variants: `*_baseline_pruned.csv`, `*_corner_filled.csv` | yes | `id1, id2, source_1, source_2, label` | `true`/`false` |

The first id of a pair belongs to the first source in the file name.

**Other files in `input/entitymatching/`** (not evaluation sets):

- `*_all.csv` (Companies, Products, and the Music variants): a larger set of labeled pairs of the source pair; the Products base file marks the split of each pair in a `split` column.
- Music: `musicbrainz_2_discogs.csv` and `musicbrainz_2_lastfm.csv` list 2,864 and 3,180 pairs, all labeled as matches; `*_train_small.csv` holds 999 labeled pairs per source pair.

**Known issues.**

- Games base: 259 of the 1,168 training and validation pairs reference DBpedia ids that are not in `dbpedia.csv`. The test splits reference only existing records.
- Companies: three matching pairs in `forbes_2_dbpedia_train.csv` (base task and its copies in the variants) use DBpedia URIs cut off at a comma (`http://dbpedia.org/resource/Workday` instead of `Workday_Inc.` in `dbpedia.csv`, likewise for Waste Management and Caesars Entertainment). The same three pairs have no label in `forbes_2_dbpedia_all.csv`.
- Music variants: `*_all.csv` labels 56 (Discogs) and 5 (Last.fm) pairs as non-matches that the base task's test split labels as matches (in the hard variant, 22 and 4 of them are in `*_test_corner_filled.csv`).

## Fusion files

| Tier | Games, Companies, Music | Products | Scientific Papers |
|---|---|---|---|
| base | `test_set.xml`, `validation_set.xml` | `fusion_test_set.csv`, `fusion_validation_set.csv` | `fusion_test.jsonl`, `fusion_val.jsonl` |
| variants | `test_set.xml`, `validation_set.xml` | `test_set.xml`, `validation_set.xml` | `test_set.xml`, `validation_set.xml` (JSON Lines despite the extension) |

- **XML:** one element per entity with an `<id>` naming a member record (Products variants: `<source_ids>`, the comma-separated list of member records) and one child element per attribute. In Companies, Music, and Products, a `provenance` attribute names the source records that support a value (several joined with `+`); the Games gold has none. List values nest one element per item. The scorer takes the `<id>` and the provenance records as the members of the entity.
- **Products base CSV and Scientific Papers JSON Lines:** one row per entity with a `source_ids` list of its member records and one column per attribute.
- **Version 2:** `GOLD_VERSION` contains `v2` in every task. Cells that no source record of the entity could yield, even after normalization, are blanked (XML: an empty element with a `v2_removed` attribute giving the reason; CSV: empty; JSON Lines: `null`) and are not graded.
- **Re-keyed records:** where a variant removed the anchor record of a gold entity, the gold record is keyed on a surviving member.

---

## Games

Three video-game sources: **Metacritic** (review scores and ESRB ratings), a dataset about game **sales** (commercial performance), and **DBpedia** (encyclopedic attributes such as release dates, developers, platforms, genres, and series). The main challenge is that the same game on a different platform is a separate entity, so a matcher must not rely too much on the title: identical titles recur across platforms, and sequels, special editions, and downloadable content differ only slightly in name. The task also requires mapping platform names and genres onto the platform and genre taxonomies.

| Source | File | Contributes | Attributes | Density (base) | Rows: base | easy | medium | hard |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DBpedia | `dbpedia.csv` | release dates, developers, platforms, genres, series | 6 | 90.9 % | 46,580 | 46,156 | 46,388 | 45,358 |
| Metacritic | `metacritic.csv` | review scores, ESRB ratings | 8 | 97.7 % | 20,494 | 20,862 | 20,740 | 18,735 |
| Sales | `sales.csv` | commercial performance | 10 | 98.7 % | 7,877 | 11,443 | 8,133 | 6,985 |
| **Total** | | | | | **74,951** | **78,461** | **75,261** | **71,078** |

Target schema: 10 attributes (`name`, `releaseYear`, `developer`, `genres`, `publisher`, `platform`, `criticScore`, `userScore`, `ESRB`, `series`). Taxonomies: `Gaming_Platforms_Taxonomy.csv`, `Video_Game_Genres_Taxonomy.csv`, `ESRB_Rating_Taxonomy.csv`.

## Companies

Records from the **Forbes** Global 2000 list of 2014, a **DBpedia** company extract, and a **FullContact** company-profile sample. Forbes provides financial figures, industries, and countries; DBpedia contributes founding dates, headquarters, industries, key people, and financial figures; FullContact adds headquarters locations, founding dates, and key people. The companies span the globe, which makes normalizing legal suffixes and resolving locations non-trivial. Further difficulties are company-name variants, spelling differences, corporate hierarchies, and the normalization of financial figures, countries, and industry terms from different classification schemes.

| Source | File | Contributes | Attributes | Density (base) | Rows: base | easy | medium | hard |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DBpedia | `dbpedia.csv` | founding dates, headquarters, industries, key people, financial figures | 8 | 62.2 % | 10,085 | 10,089 | 10,075 | 10,069 |
| Forbes | `forbes.csv` | financial figures, industries, countries | 6 | 99.1 % | 2,000 | 2,000 | 2,205 | 2,107 |
| FullContact | `fullcontact.csv` | headquarters locations, founding dates, key people | 5 | 61.9 % | 1,931 | 1,933 | 2,052 | 2,004 |
| **Total** | | | | | **14,016** | **14,022** | **14,332** | **14,180** |

Target schema: 8 attributes (`name`, `founded`, `country`, `city`, `industry`, `assets`, `revenue`, `keypeople`). Taxonomies: `CLDR_Country_Taxonomy.csv`, `GICS_Industry_Taxonomy.csv` (Global Industry Classification Standard). The FullContact source uses opaque column names (`Attribute_1` …).

## Music

Release-level records from **Discogs**, **Last.fm**, and **MusicBrainz**. Discogs provides release metadata with music labels, genres, countries, and track lists; Last.fm contributes album metadata; MusicBrainz provides release titles, artists, dates, countries, durations, and track lists. The records describe albums, EPs, and singles, with partial overlap across sources. The main challenge lies in heterogeneous value formats: for example, Discogs mostly records only the release year where MusicBrainz records the full date, and Discogs writes country abbreviations where MusicBrainz writes the full country name. Further difficulties are title and artist variants, sparse source records, and the normalization of dates, countries, and track lists.

| Source | File | Contributes | Attributes | Density (base) | Rows: base | easy | medium | hard |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Discogs | `discogs.csv` | release metadata, labels, genres, countries, track lists | 8 | 93.1 % | 22,627 | 23,887 | 22,660 | 22,255 |
| Last.fm | `lastfm.csv` | album metadata | 4 | 73.2 % | 9,865 | 11,022 | 9,857 | 9,339 |
| MusicBrainz | `musicbrainz.csv` | release titles, artists, dates, countries, durations, track lists | 6 | 94.6 % | 4,763 | 4,789 | 4,818 | 3,575 |
| **Total** | | | | | **37,255** | **39,698** | **37,335** | **35,169** |

Target schema: 8 attributes (`name`, `artist`, `release-date`, `release-country`, `label`, `genre`, `tracks`, `duration`). Taxonomy: `Music_Genres_Taxonomy.csv`. The MusicBrainz source uses opaque column names (`Attribute_1` …). The densities count a duration of 0 as missing, as the target schema prescribes. The target schemas of the Music variants carry fewer value constraints than the base schema (for example, no range for `duration`); in the other domains, the variants use the schema of their base task.

## Products

A sample of the entity matching benchmark **WDC Products**, covering GPUs, SSDs, HDDs, and USB sticks. The attribute values were extracted from free-text product offers with information extraction techniques, and the four sources describe the offers with different source schemas. The target schema has 24 attributes, the most in the benchmark. This makes matching delicate: a small difference in a single technical attribute can separate a match from a non-match, and the same attribute appears under different names and in different value formats across sources.

| Source | File (base / variants) | Attributes | Density (base) | Rows: base | easy | medium | hard |
|---|---|---:|---:|---:|---:|---:|---:|
| Dataset 1 | `dataset_1.json` / `products_1.csv` | 24 | 53.2 % | 812 | 806 | 811 | 660 |
| Dataset 2 | `dataset_2.json` / `products_2.csv` | 24 | 52.5 % | 812 | 806 | 811 | 635 |
| Dataset 3 | `dataset_3.json` / `products_3.csv` | 24 | 51.9 % | 762 | 784 | 780 | 574 |
| Dataset 4 | `dataset_4.json` / `products_4.csv` | 24 | 52.4 % | 626 | 695 | 698 | 498 |
| **Total** | | | | **3,012** | **3,091** | **3,100** | **2,367** |

Target schema: 24 attributes (`brand`, `title`, `description`, `price`, `priceCurrency`, `url`, `model`, `model_number`, `product_type`, `chipset_name`, `vram_gb`, `storage_gb`, `read_speed_mb_s`, `write_speed_mb_s`, `bus_type`, `interface_type`, `width_mm`, `length_mm`, `height_mm`, `weight_g`, `storage_connection_type`, `memory_type`, `color`, `form_factor`). Taxonomies: `Product_Type_Taxonomy.csv`, `Storage_Interface_Taxonomy.csv`, `GPU_Memory_Taxonomy.csv`; the target schema fixes the unit of each capacity, speed, dimension, and weight attribute (gigabytes, megabytes per second, millimetres, grams). The entity matching sets reuse the matches of WDC Products; the released sources do not contain its cluster ids. Of the 24 attributes, 19 are graded in fusion; `title`, `description`, `price`, `priceCurrency`, and `url` are not.

## Scientific Papers

Computer-science paper records from **DBLP**, **Crossref**, and **OpenAlex**, with bibliographic attributes such as title, authors, publication year, venue, page range, and citation-related counts. This is the largest task by row count, so blocking efficiency is a challenge: a blocker with a poor reduction ratio leads to hundreds of thousands of record pair comparisons, which makes the task suited for evaluating efficiency as well as effectiveness. Further challenges are title and author-list variants, incomplete identifiers, the normalization of publication types and venues, and sparse metadata for volume, issue, pages, and citation counts. The entity matching pairs were derived from DOIs; the released sources do not contain them, and the fusion gold identifies the member records of each paper by their source ids (`source_ids`, for example `dblp-…`, `crossref-…`, `open_alex-…`).

| Source | File (base / variants) | Contributes | Attributes | Density (base) | Rows: base | easy | medium | hard |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Crossref | `crossref.jsonl` / `crossref.csv` | type, title, authors, year, venue, publisher, abstract, volume, issue, pages, reference and citation counts | 13 | 85.9 % | 60,749 | 60,705 | 60,749 | 40,916 |
| DBLP | `dblp.jsonl` / `dblp.csv` | type, title, authors, year, venue, volume, issue, pages | 9 | 68.1 % | 60,591 | 60,690 | 60,595 | 55,972 |
| OpenAlex | `open_alex.jsonl` / `open_alex.csv` | type, title, authors, year, venue, topics, volume, issue, pages, reference and citation counts | 12 | 88.6 % | 60,719 | 60,818 | 60,723 | 39,988 |
| **Total** | | | | | **182,059** | **182,213** | **182,067** | **136,876** |

Target schema: 11 attributes (`type`, `title`, `authors`, `publication_year`, `journal`, `volume`, `issue`, `first_page`, `last_page`, `referenced_works_count`, `cited_by_count`); no taxonomies. The variant sources have more columns than the base sources (Crossref 14, DBLP 13, OpenAlex 13 attributes besides the id).
