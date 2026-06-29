# MaDI-Bench Use Cases

This directory holds the 20 integration tasks of MaDI-Bench: five domains, each with one **base** task plus an **easy**, **medium**, and **hard** variant.

```
use cases/<domain>/<base|easy|medium|hard>/
```

The base task is the real-world integration problem. Each variant keeps that same problem but dials the difficulty of the pipeline steps up or down, using the difficulty knobs in [`../knobs/`](../knobs/). For the layout inside each task (inputs, ground truth, reference outputs), see the [top-level README](../README.md#use-cases--the-benchmark-tasks).

## Overview

| Domain | Sources | Source format | Target attributes | Records (base) | Records (hard) |
|---|---:|---|---:|---:|---:|
| [Games](games/) | 3 | CSV | 10 | 74,951 | 71,078 |
| [Companies](companies/) | 3 | CSV | 8 | 14,016 | 14,180 |
| [Music](music/) | 3 | CSV | 8 | 37,255 | 35,169 |
| [Products](products/) | 4 | JSON / CSV | 25 | 3,012 | 2,365 |
| [Scientific Papers](papers/) | 3 | JSONL / CSV | 12 | 182,059 | 136,876 |

### Labeled ground-truth resources (base tasks)

Beyond the source tables, each task ships labeled splits for entity matching and hand-annotated records for data fusion (counts per the paper):

| Domain | Source pairs | EM train | EM val | EM test | Fusion val | Fusion test |
|---|---:|---:|---:|---:|---:|---:|
| Games | 2 | 934 | 234 | 739 | 100 | 100 |
| Companies | 2 | 1,971 | 948 | 599 | 100 | 100 |
| Music | 3 | 36,658 | 17,010 | 2,000 | 100 | 100 |
| Products | 3 | 4,488 | 600 | 600 | 100 | 100 |
| Scientific Papers | 2 | 10,666 | 2,666 | 13,332 | 100 | 100 |

Across the five base tasks this amounts to more than 93,000 labeled record pairs for entity matching and 1,000 hand-annotated fusion records carrying close to 10,000 verified attribute values.

Each fusion directory also ships a **Better Readability** CSV alongside every original validation/test split, named with the suffix `_better_readability.csv`. These files keep only the target-schema columns (from `input/schemamatching/target_schema.json`), in schema order, and drop the provenance and source-helper fields — which makes them easier to skim when first exploring a task. The original fusion files remain the authoritative artifacts: they retain the source attributes, provenance annotations, and raw left/right values needed for tracing and evaluation. When a target-schema field is absent from an original split, the Better Readability CSV leaves that column empty rather than inferring a value.

---

## Games

Three video-game sources: **Metacritic** (review scores and ESRB ratings), a **Sales** dataset (commercial performance), and **DBpedia** (release dates, developers, platforms, genres, series). The main challenge is that the same game on a different platform is a separate entity, so a matcher must not over-rely on the title — identical titles recur across platforms and sequels, and special editions differ only slightly in name. The task ships normalization taxonomies for platform, genre, date, and rating.

| Source | What it contributes | Cols | Density | base | easy | medium | hard |
|---|---|---:|---:|---:|---:|---:|---:|
| DBpedia | release dates, developers, platforms, genres, series | 7 | 92.2% | 46,580 | 46,244 | 46,388 | 45,358 |
| Metacritic | review scores, ESRB ratings | 9 | 98.0% | 20,494 | 21,214 | 20,740 | 18,735 |
| Sales | commercial performance | 11 | 98.8% | 7,877 | 11,467 | 8,133 | 6,985 |
| **Total** | | | | **74,951** | **78,925** | **75,261** | **71,078** |

Target schema: 10 attributes (plus an `id`). All sources are CSV.

## Companies

Records from the **Forbes** Global 2000 list, a **DBpedia** company extract, and a **FullContact** company-profile sample. Forbes brings financial and ranking information, DBpedia brings founding dates, headquarters, industries, and key people, and FullContact adds contact and personnel information. Companies span the globe, which makes normalizing legal suffixes and resolving countries non-trivial. Further difficulties include company-name variants, spelling differences, corporate hierarchies, and the normalization of financial figures, countries, and industry categorization taxonomies.

| Source | What it contributes | Cols | Density | base | easy | medium | hard |
|---|---|---:|---:|---:|---:|---:|---:|
| DBpedia | founding dates, headquarters, industries, key people | 9 | 66.4% | 10,085 | 10,583 | 10,075 | 10,069 |
| Forbes | Global 2000 financials and ranking | 7 | 99.2% | 2,000 | 2,025 | 2,205 | 2,107 |
| FullContact | contact and personnel profiles | 6 | 69.0% | 1,931 | 2,166 | 2,052 | 2,004 |
| **Total** | | | | **14,016** | **14,774** | **14,332** | **14,180** |

Target schema: 8 attributes (plus an `id`). All sources are CSV. The FullContact source uses opaque column names (`Attribute_1` …).

## Music

Release-level records from **Discogs**, **Last.fm**, and **MusicBrainz**. Discogs provides release metadata with labels, genres, countries, and track lists; Last.fm contributes album metadata; MusicBrainz provides release, artist, and label information. The records describe albums, EPs, and singles with partial overlap across sources. The main challenge lies in heterogeneous value formats — for example, album durations recorded in different formats — along with title and artist variants, sparse source records, and normalization of dates, countries, and track lists.

| Source | What it contributes | Cols | Density | base | easy | medium | hard |
|---|---|---:|---:|---:|---:|---:|---:|
| Discogs | release metadata, labels, genres, countries, tracks | 9 | 98.6% | 22,627 | 23,887 | 22,660 | 22,255 |
| Last.fm | album metadata | 5 | 89.4% | 9,865 | 11,022 | 9,857 | 9,339 |
| MusicBrainz | release, artist, label information | 7 | 96.8% | 4,763 | 4,789 | 4,818 | 3,575 |
| **Total** | | | | **37,255** | **39,698** | **37,335** | **35,169** |

Target schema: 8 attributes (plus an `id`). All sources are CSV. The MusicBrainz source uses opaque column names (`Attribute_1` …); variant sources expose a few additional fields.

## Products

A sample of the **WDC Products** entity benchmark, covering GPUs, SSDs, HDDs, and USB sticks. The four sources describe product offers using different cross-schema variants. The target schema has the most attributes in the benchmark (25), which makes matching delicate: a small difference in a single technical attribute can separate a match from a non-match, and the same attribute appears under different naming conventions across sources. The task includes normalization taxonomies for units, capacities, dimensions, and speeds.

| Source | What it contributes | Cols | Density | base | easy | medium | hard |
|---|---|---:|---:|---:|---:|---:|---:|
| Dataset 1 | product offers (schema variant 1) | 27 | 58.3% | 812 | 805 | 810 | 653 |
| Dataset 2 | product offers (schema variant 2) | 27 | 57.7% | 812 | 805 | 810 | 635 |
| Dataset 3 | product offers (schema variant 3) | 27 | 57.1% | 762 | 784 | 780 | 581 |
| Dataset 4 | product offers (schema variant 4) | 27 | 57.6% | 626 | 701 | 703 | 496 |
| **Total** | | | | **3,012** | **3,095** | **3,103** | **2,365** |

Target schema: 25 attributes (plus an `id`). Base sources are JSON; the variants are CSV.

## Scientific Papers

Computer-science paper records from **DBLP**, **Crossref**, and **OpenAlex**. DOIs were used only to derive the entity-matching pairs and fusion splits; released source files omit DOI and expose stable source-record ids instead. The fusion ground truth uses a `source_ids` helper list (for example `dblp-...`, `crossref-...`, `open_alex-...`) to identify the source records that describe each fused paper. Bibliographic attributes include title, authors, publication year, venue, page range, and citation counts. This is the largest task by row count, so blocking efficiency is a challenge: a blocker with a poor reduction ratio leads to hundreds of thousands of record-pair comparisons, which makes the task well suited for evaluating efficiency as well as effectiveness. Further challenges involve title and author-list variants, incomplete identifiers, publication-type and venue normalization, and sparse metadata for volume, issue, pages, and citation counts.

| Source | What it contributes | Cols | Density | base | easy | medium | hard |
|---|---|---:|---:|---:|---:|---:|---:|
| Crossref | type, title, authors, venue, pages, references, citations | 14 | 88.1% | 60,749 | 60,705 | 60,749 | 40,916 |
| DBLP | type, title, authors, venue, volume, issue, pages | 10 | 71.3% | 60,591 | 60,690 | 60,595 | 55,972 |
| OpenAlex | type, title, authors, source, topics, pages, citations | 13 | 89.5% | 60,719 | 60,818 | 60,723 | 39,988 |
| **Total** | | | | **182,059** | **182,213** | **182,067** | **136,876** |

Target schema: 12 attributes (plus an `id`). Base sources are JSONL; the variants are CSV, which expose a few additional fields.
