# MaDI-Bench: An End-to-End Data Integration Benchmark

**Aaron Steiner\*, Ralph Peeters\*, Christian Bizer**
Data and Web Science Group, University of Mannheim, Germany
*\*These authors contributed equally to this work.*

**Paper:** https://arxiv.org/abs/2606.30371 · **Website:** https://wbsg-uni-mannheim.github.io/MaDI-Bench/

---

## Abstract

> Data integration is the process of combining data from multiple, heterogeneous sources into a consistent, unified representation. Data integration involves a sequence of interdependent tasks including schema matching, value normalization, blocking, entity matching, and data fusion. Existing table-based benchmarks either evaluate these steps in isolation or cover only incomplete versions of the data integration pipeline, omitting specific steps. The lack of public end-to-end data integration benchmarks hinders research on data integration methods that address the integration process as a whole and account for the interdependencies among the different tasks. This paper fills this gap by introducing the Mannheim Data Integration Benchmark (MaDI-Bench), the first benchmark for the end-to-end integration of relational tables covering all steps of the integration process. MaDI-Bench contributes (i) a set of end-to-end data integration tasks spanning several application domains, each requiring the full schema matching, value normalization, entity matching, and data fusion pipeline, and (ii) a generic method for deriving task variants that mitigates rapid benchmark saturation as data integration systems advance. We validate the benchmark using human-engineered pipelines, a best-of-breed pipeline, an LLM workflow, and a pipeline written by a coding agent. The validation demonstrates the utility of the benchmark for measuring the step-wise as well as the end-to-end performance of data integration pipelines. All benchmark artifacts are available for public download.

The abstract and all counts in this README follow the paper.

**Keywords:** data integration, schema matching, entity matching, data fusion, end-to-end evaluation, large language models

**Contents:** [What is MaDI-Bench?](#what-is-madi-bench) · [Repository structure](#repository-structure) · [Installation](#installation) · [The tasks](#the-tasks) · [Evaluating a system](#evaluating-a-system) · [Reference pipelines and results](#reference-pipelines-and-results) · [Generating variants](#generating-variants) · [Citation](#citation)

---

## What is MaDI-Bench?

The Mannheim Data Integration Benchmark (MaDI-Bench) is a benchmark for the **end-to-end integration of relational tables**. Each task gives an integration system several heterogeneous source tables and a target schema. The system must return a single table that conforms to the target schema and contains one record per real-world entity. This requires all steps of the integration process:

**Schema matching → value normalization → blocking → entity matching → data fusion**

MaDI-Bench provides ground truth for every step, so a system can be evaluated step by step and end to end. Because errors propagate from one step to the next, the end-to-end evaluation shows their combined effect on the integrated table.

- **20 tasks in 5 domains:** Games, Companies, Music, Products, and Scientific Papers. Each domain has one base task built from real sources and three generated variants (*easy*, *medium*, *hard*).
- **Labeled resources for every step.** Across the five base tasks: a gold schema mapping per task; 93,439 labeled record pairs for entity matching, split into training, validation, and test sets (the test sets also evaluate blocking); 1,745 labeled normalization cells in validation and test sets; and 1,000 human-verified fusion records (100 validation and 100 test records per task) carrying close to 8,000 verified attribute values (7,724 non-empty gold cells over the target attributes, Table 2 of the paper). The variants provide the same kinds of resources for their own records.
- **Variant generation.** Eight difficulty knobs perturb a base task to derive easier and harder versions of the same integration problem. The easy variants suit simpler and cheaper methods; the hard variants leave room for future systems.
- **Four validation pipelines:** human-designed PyDI pipelines (P1), a best-of-breed pipeline (P2), an LLM workflow (P3), and a pipeline written by the coding agent Claude Code (P4).
- **Evaluation code.** The package `madi_bench.evaluation` in this repository scores value normalization and data fusion. Schema matching, blocking, entity matching, and the end-to-end metrics are scored with the evaluators of the [PyDI data integration framework](https://github.com/wbsg-uni-mannheim/PyDI) at the commit that this repository pins.

All files use common formats: CSV, JSON, JSON Lines, and XML.

---

## Repository structure

| Path | Contents |
|---|---|
| [`use cases/`](use%20cases/) | **The benchmark:** the 20 tasks with source tables, target schemas, and ground truth; for each base task also the P1 notebook and P1's reference outputs. Folder layout, file formats, and statistics: [`use cases/README.md`](use%20cases/README.md). |
| [`madi_bench/evaluation/`](madi_bench/evaluation/) | Evaluation package for normalization and fusion, with the command-line tools `madi-score-normalization` and `madi-score-fusion`. Its tests are in [`tests/`](tests/). |
| [`results/`](results/) | Outputs and scores of the validation runs: [`best of breeds/`](results/best%20of%20breeds/) (P2), [`llm pipeline/`](results/llm%20pipeline/) (P3), [`claude code/`](results/claude%20code/) (P4), `scores_v2.json` (the scores of the stored outputs of P1 to P3 under the rules of [Evaluating a system](#evaluating-a-system)), and `paper_tables/` (the measurements behind the other cells of the paper's tables). |
| [`reproduction/`](reproduction/) | The scripts behind the paper's numbers: scoring of the stored outputs (`scoring/`), the table cells (`tables/`), and the headless run of the P1 notebooks (`p1/`). |
| [`pipelines/`](pipelines/) | The best-of-breed pipeline (P2). |
| [`baselines/llm-pipeline/`](baselines/llm-pipeline/) | P3: code and prompts of the LLM workflow. |
| [`baselines/claude-code/`](baselines/claude-code/) | P4: the seven skills and the prompts given to Claude Code. |
| [`usecases_synthetic/`](usecases_synthetic/) | The variant generator and the committees that measure the difficulty of the variants (Tables 11 and 12 of the paper). |
| [`knobs/`](knobs/) | Specification cards of the difficulty knobs (numbering: see below). |
| [`difficulty_dimensions.md`](difficulty_dimensions.md) | The per-step difficulty dimensions that the knobs target. |
| [`cluster/slurm/`](cluster/slurm/) | SLURM job scripts used for the P2 runs, the committee runs, and variant generation. |
| [`website/`](website/) | Source of the benchmark website. |

**Knob numbering.** The paper numbers the eight knobs 1 to 8. The knob cards in `knobs/` and the generator configurations in `usecases_synthetic/config/` use an older numbering from 01 to 10: paper knobs 1 to 6 are knobs 01 to 06, paper knob 7 (schema naming divergence) is knob 08, and paper knob 8 (source reliability differentiation) is knob 10. Knobs 07 and 09 are specified but not used in the released variants.

---

## Installation

The tasks are plain files and need no installation. The code needs Python 3.12 (tested with 3.12.13) and git.

```bash
git clone https://github.com/wbsg-uni-mannheim/MaDI-Bench.git
cd MaDI-Bench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

This installs the repository in editable mode together with PyDI at commit [`cd51e25`](https://github.com/wbsg-uni-mannheim/PyDI/tree/cd51e25e5e7f0493c45f678d0d0ef6e123d9133a), which `pyproject.toml` pins. PyDI's default branch lacks the `synthetic` extra and the `PyDI.evaluation` package that the P2 pipeline, the variant generator, and the end-to-end metrics use. The install brings everything that the evaluation package, P2, the variant generator, and the test suites import (about 180 packages, including PyTorch). Optional extras:

- `neural` adds `faiss-cpu`, the nearest-neighbour index of the SC-Block blocker (without it, SC-Block falls back to scikit-learn).
- `llm` and `matchers` name packages that PyDI's `synthetic` extra already installs.

**pycountry.** The end-to-end values of Table 8 were computed with pycountry 26.2.16. PyDI at the pinned commit requires pycountry below 25, so pip installs 24.6.1; for the Table 8 values, run `python -m pip install pycountry==26.2.16` after the install (pip then reports the version conflict; the test suites pass with either version).

**P1 notebooks.** Opening them interactively needs Jupyter (`python -m pip install jupyter`); `reproduction/p1/run_p1.sh` runs them as scripts. The Papers notebook also needs XGBoost (`python -m pip install xgboost`).

**PyDI version of the reported runs.** P1 and the scoring of the reported numbers used code equivalent to `cd51e25`. P2 and the difficulty committees ran on its parent commit `bc05405`, which differs in two places: its entity matching comparators return 0.0 instead of raising an error when they are called with raw values instead of records, and its fusion engine finds connected components recursively (`cd51e25` does so iteratively, in the same order, which avoids Python's recursion limit on very large components).

**Scoring only.** The evaluation package (`madi-score-fusion`, `madi-score-normalization`) needs nothing but pandas:

```bash
python -m pip install pandas
python -m pip install -e . --no-deps
```

`reproduction/scoring/` needs the full install, because it calls PyDI's evaluators.

**Tests.** `pytest tests/` tests the evaluation package; `pytest pipelines/tests usecases_synthetic/tests` tests P2 and the variant generator. Some tests are skipped, for example those that need an API key.

**Hardware.** The tasks and the scorers need no GPU. P2 and the committee runs fine-tune the Ditto matcher and the SC-Block blocker. Their SLURM scripts in `cluster/slurm/` (`run_bob_em_train.sbatch` and `run_best_of_breed.sbatch` for P2; `run_em_train_committee.sbatch`, `run_measure_validate.sbatch`, and `run_validate_level.sbatch` for the committees) request one GPU with 48 GB memory, 12 CPU cores, and 64 GB RAM per job. Variant generation runs on CPUs (`run_generate_variant.sbatch` requests 16 cores and 96 GB RAM).

**API keys.** Components that call an LLM need an OpenAI key in `OPENAI_API_KEY` (environment or a `.env` file): the PyDI LLM schema matcher in the P1 notebooks of Companies, Games, and Music; the LLM members of P2's schema matching committee (`--no-llm-sm` removes them) and of the committees; and knobs 1 and 2 of the variant generator, which rewrite values and generate entities with an LLM. Scoring needs no key.

---

## The tasks

Each task is a folder `use cases/<domain>/<tier>/` with `<domain>` one of `games`, `companies`, `music`, `products`, `papers` and `<tier>` one of `base`, `easy`, `medium`, `hard`. A system receives:

- the **source tables** in `input/data/` (base tasks: with one metadata file per source that records provenance, columns, and publication date),
- the **target schema** `input/schemamatching/target_schema.json`, a JSON Schema that defines each target attribute with its type and value constraints, and the **taxonomies** (`input/schemamatching/*_Taxonomy.csv`) onto which categorical attributes are mapped,
- optionally, the labeled **training and validation sets** for entity matching, normalization, and fusion.

It returns one fused table that conforms to the target schema and contains one row per real-world entity. For the step-wise evaluation, it also reports its intermediate results (next section). [`use cases/README.md`](use%20cases/README.md) describes the folder layout, the file formats, and the statistics of all 20 tasks.

---

## Evaluating a system

Validation splits are for tuning; report the test splits. Record ids must be the ids of the source tables (their first column; see [`use cases/README.md`](use%20cases/README.md)).

**Scoring a system in one go.** Put the system's outputs for one task into one folder (any subset of the files)

```
submission/
├── sm_mapping.csv            schema matching: source_dataset, source_column, target_dataset, target_column, score
├── normalization/            normalized source tables, one CSV per source
├── blocking/candidates.csv   all candidate pairs: id1, id2
├── correspondences.csv       matched record pairs: id1, id2[, score]
├── fused.csv                 fused table: _id and the target attributes
└── membership.csv            record_id, source, cluster_id (cluster_id = the _id of the fused row)
```

and score every step that is present:

```bash
python -m reproduction.scoring.score_submission --task "use cases/games/base" --submission submission/ [--e2e] [--json scores.json]
```

The layout is that of the P4 runs plus `normalization/`, and the scorer is the one that scored them. For a check: on P2's Games outputs (`fused.csv` and `correspondences.csv` from `results/best of breeds/games/baseline/`) it gives the entity matching F1 (0.6730) and the fusion accuracy (0.6570) of Tables 6 and 7. The table below summarizes the steps; the subsections give the details and show how to call the scorers directly.

| Step | What the system submits | Gold file (in the task folder) | Metric | Code |
|---|---|---|---|---|
| Schema matching | correspondences between source columns and target attributes | `input/schemamatching/sm_mapping_gold.json` (base), `input/schemamatching/sm_mapping.csv` (variants) | F1 over correspondences | PyDI `SchemaMappingEvaluator` |
| Normalization | the normalized source tables | `input/normalization/test.csv` | accuracy over the test cells | `madi-score-normalization` |
| Blocking | all candidate pairs | entity matching test splits | pair completeness and reduction ratio, mean over source pairs | PyDI `EntityMatchingEvaluator.evaluate_blocking` |
| Entity matching | matched record pairs | entity matching test splits | F1, mean over source pairs | PyDI `EntityMatchingEvaluator.evaluate_matching` |
| Data fusion | fused table and membership table | fusion test split in `input/fusion/` | all-gold accuracy | `madi-score-fusion` |
| End-to-end | fused table and membership table | fusion test split (ground truth); P1's fused output (silver reference, base tasks) | the metrics of Table 3 of the paper | PyDI `compute_e2e_panel` |

PyDI's evaluators are used at the pinned commit `cd51e25`; [`reproduction/scoring/`](reproduction/scoring/) holds the code that applied them for the paper's numbers (file loading, id handling, averaging).

### Schema matching

- **Submit** a CSV with the columns `source_dataset, source_column, target_dataset, target_column, score`, one row per correspondence between a source column and a target attribute, including each source's identifier column (target attribute `id`). `source_dataset` is the source file name without extension (`metacritic`, `open_alex`). The gold of the Products base task names the sources `products_1` to `products_4` (files `dataset_1.json` to `dataset_4.json`).
- **Gold:** base tasks `input/schemamatching/sm_mapping_gold.json` (the list `mappings`; `unmapped_source_columns` lists the source columns that have no target attribute), variants `input/schemamatching/sm_mapping.csv` (same columns, every row a correspondence). The variants rename the source columns, so each has its own mapping.
- **Metric:** precision, recall, and F1 over all correspondences of the task; a submitted correspondence that is not in the gold is a false positive (`SchemaMappingEvaluator.evaluate(submitted, gold, complete=True)`). PyDI compares all four name columns; the scoring for the paper ignores `target_dataset` (the gold uses `target_schema` in the base tasks and the domain name in the variants) and accepts `dataset_N` for `products_N`.

```python
import json
import pandas as pd
from PyDI.schemamatching import SchemaMappingEvaluator

cols = ["source_dataset", "source_column", "target_dataset", "target_column"]
gold = pd.DataFrame(json.load(open("use cases/games/base/input/schemamatching/sm_mapping_gold.json"))["mappings"])
mine = pd.read_csv("sm_mapping.csv")   # your correspondences; target_dataset spelled as in the gold
print(SchemaMappingEvaluator.evaluate(mine, gold[cols], complete=True)["f1"])
```

### Value normalization

- **Submit** the normalized source tables: every source record with its values mapped to the target attributes and normalized. Either a directory with one CSV per source, named like the source (`metacritic.csv`, `dataset_1.csv`; Products also accepts `products_N` and `prodN`), each with an `id` or `record_id` column holding the source table's own ids (with or without the `<source>_` prefix) and the target attributes as columns; or one CSV with `source` and `record_id` columns. Files without an id column are skipped and listed.
- **Gold:** `input/normalization/test.csv` (and `validation.csv` for tuning). Each row names a source cell, the raw value, and the `expected_value` that a correct normalizer produces from the raw value alone. Categories: `identity` (the raw value already is correct), `normalization` (a rule or a vocabulary that the task declares recovers it: units, dates, casing, aliases, the shipped taxonomies; column `rule` names the minimal rule), and `knowledge` (recoverable only with outside knowledge, for example expanding a journal abbreviation).
- **Metric:** accuracy over all cells of the test set (reported also for the transformation cells alone, per attribute, and per rule, with the share of cells whose record was found). A cell is correct when the normalized value equals the expected value as a string; list attributes (companies `keypeople`, games `genres`, music `tracks`, papers `authors`) are compared as lists in any accepted serialization (JSON list, `|` or `;` separated, a single bare value), in order except `keypeople`. Nothing else is forgiven: numbers compare as text (`1888` is not `1888.0`), whitespace and case count, and a record that is not found counts as wrong.

```bash
madi-score-normalization --task "use cases/games/base" --tables my_normalized_tables/ [--split validation] [--json result.json]
```

### Blocking and entity matching

- **Test files** (one per source pair): base tasks the plain `*_test.csv` files; variants the `*_test_corner_filled.csv` files, with `*_train_corner_filled.csv` and `*_val_corner_filled.csv` as their training and validation splits. The plain-named `*_train/val/test.csv` files in the variant folders are copies of the base task's splits and may reference records that the variant dropped; do not evaluate on them. [`use cases/README.md`](use%20cases/README.md#entity-matching-files) lists the files of every domain and their formats.
- **Submit** for blocking all candidate pairs as a CSV `id1,id2` (the complete candidate set: pair completeness is computed over exactly what is submitted), and for entity matching the matched pairs as `id1,id2` (a `score` column is allowed and not used).
- **Metrics,** computed per source pair and then averaged without weights over the source pairs of the task (as in Table 6 of the paper):
  - *pair completeness*: the share of the test split's matching pairs that are in the candidate set;
  - *reduction ratio*: 1 − (candidate pairs between the two sources) / (rows of the first source × rows of the second source);
  - *precision, recall, F1* of the match class. A submitted pair counts as a false positive only when the test split labels it a non-match; submitted pairs that are not in the test split are not scored (PyDI's convention).
- **Pair direction:** the test files give each pair in the order of the file name (`forbes_2_dbpedia`: Forbes id first). The scoring for the paper sorts the two ids of every pair on both sides, so direction does not matter there; when calling PyDI directly, orient the pairs as in the test file.

```python
import pandas as pd
from PyDI.entitymatching import EntityMatchingEvaluator

pred = pd.read_csv("results/best of breeds/music/baseline/correspondences.csv", dtype=str)   # P2's matches
pred["score"] = 1.0   # PyDI requires the column; it is not used without a threshold
f1 = []
for pair in ["musicbrainz_2_discogs", "musicbrainz_2_lastfm"]:
    test = pd.read_csv(f"use cases/music/base/input/entitymatching/{pair}_test.csv",
                       header=None, names=["id1", "id2", "label"], dtype=str)
    test["label"] = test["label"].str.lower().eq("true").astype(int)
    f1.append(EntityMatchingEvaluator.evaluate_matching(pred, test)["f1"])
print(sum(f1) / len(f1))   # 0.9484, the P2 Music cell of Table 6
# blocking: EntityMatchingEvaluator.evaluate_blocking(candidates, test,
#               total_possible_pairs=len(source_a) * len(source_b))
```

### Data fusion

- **Submit** the fused table (one row per fused entity, an `_id` column and the target attributes) and a membership table `record_id, source, cluster_id` that says which source records went into which row (`cluster_id` = the row's `_id`). Instead of the membership table, the fused table may carry a `_fusion_sources` column with the member record ids of each row, as PyDI's fuser writes it. Ids are compared as strings; read the tables with `madi_bench.evaluation.load_table` (or `dtype=str`) so that numeric-looking ids are not changed. List-valued attributes may be written as a JSON list or `|`-separated.
- **Gold:** the fusion test split in `input/fusion/` (file names per domain and tier: [`use cases/README.md`](use%20cases/README.md#fusion-files)), 100 human-verified entities per split. Every task carries version 2 of the gold (`input/fusion/GOLD_VERSION` = `v2`): the verified values, with every cell blanked that no source record of the entity could yield, even after normalization; blanked cells are not graded. The base test splits grade 793 (games), 610 (companies), 726 (music), 836 (products), and 922 (papers) cells; the variants keep the entities and the gold values of their base task.
- **Metric:** the *all-gold accuracy*, which the paper reports as fusion accuracy (Table 7)

      all-gold accuracy = accuracy on the graded gold cells × gold coverage × attribute cell coverage

  The accuracy pools all graded cells of all attributes. *Gold coverage* is the share of the 100 gold entities that the submission reached: an entity the system did not produce counts as wrong. *Attribute cell coverage* is the share of graded cells whose attribute the submission carries at all: dropping a column costs the cells it held. A missing fused value is wrong.
- **Alignment:** every gold record and every fused row is keyed on its *anchor*: the member record from the highest-priority source present, recognized by the record-id prefix (companies: the Forbes URL > the DBpedia URL > `fullcontact_`; games: `metacritic_` > `dbpedia_` > `sales_`; music: `mbrainz_` > `discogs_` > `lastFM_`; papers: `dblp-` > `crossref-` > `open_alex-`; products: `products_1_` > ... > `products_4_`, a prefix that the scorer adds to the ids of the Products base sources). Within the primary source the anchor is the first member listed (for the XML gold files, the scorer sorts the members by id first), within a fallback source the lexicographically smallest id. A gold record is reached when a fused row carries its anchor; if the fused table has a `_fusion_sources` column, the record is also graded through a fused row that lists its anchor there, as PyDI's evaluator does.
- **Variants:** records that a variant added never serve as anchors; the scorer finds them by comparing with the base task, so keep `use cases/<domain>/base/` next to the variant folders. Where a variant removed the anchor record of a gold entity (mostly in the hard variants), the gold record is keyed on a surviving member.
- **Comparison per attribute:** the strict rule tables in [`madi_bench/evaluation/fusion_rules.py`](madi_bench/evaluation/fusion_rules.py) (`STRICT_RULES`). A comparator tolerates how a value is written, but not disagreement about the fact, and not the repairs that the normalization step is expected to make (annotation suffixes, a music release date to the day, typography, taxonomy mapping). Other implementations can be used if they follow these tables:

<details>
<summary>Per-domain rule tables (from <code>STRICT_RULES</code>)</summary>

"Same words" means the same set of lower-cased words with punctuation removed (word order does not matter). "Identical" means string equality after the same light preparation on both sides (for example, `42.0` and `42` are the same year or count).

| Domain | Attribute: comparison |
|---|---|
| Companies | `name`, `country`, `city`: same words · `revenue`, `assets`: within 2 % · `keypeople`: the same set of names after accent and case folding · `founded`: same year · `industry`: identical |
| Games | `name`, `publisher`: same words · `platform`, `developer`, `ESRB`: identical · `releaseYear`: same year · `criticScore`: within ±1 · `userScore`: within ±0.1 · `genres`: every gold genre present, ignoring case (extra genres allowed) · `series`: not graded |
| Music | `name`, `artist`, `release-country`: same words · `duration`: within ±5 seconds · `release-date`: the same full date (a year or a year and month alone does not match) · `label`: every gold label present, ignoring case (extra labels allowed) · `tracks`: the same set of track names, ignoring case only · `genre`: identical |
| Products | `brand`, `product_type`, `color`: identical ignoring case · `vram_gb`, `storage_gb`, `read_speed_mb_s`, `write_speed_mb_s`, `width_mm`, `length_mm`, `height_mm`, `weight_g`: within 2 % · `chipset_name`, `interface_type`, `memory_type`: the same sequence of numbers, and the letters and digits of one value contain those of the other · `bus_type`: as before, after mapping the USB-IF renames of one standard onto each other (USB 3.0 = USB 3.1 Gen 1 = USB 3.2 Gen 1; USB 3.1 = USB 3.1 Gen 2 = USB 3.2 Gen 2) · `model`, `model_number`: identical after case folding, with each run of characters other than letters and digits read as one space · `storage_connection_type`: same words · `form_factor`: identical · `title`, `description`, `price`, `priceCurrency`, `url`: not graded |
| Papers | `type`, `volume`, `issue`, `first_page`, `last_page`, `referenced_works_count`: identical · `title`: the same set of words after Unicode, dash, and case folding (punctuation kept) · `authors`: the same set of authors, each name compared as its set of words after accent and case folding (`J.` and `J`, and the order of first and last name, do not matter; a missing or extra name or initial does) · `publication_year`: same year · `journal`: same words · `cited_by_count`: within 10 % or 2, whichever is larger |

</details>

```bash
madi-score-fusion --task "use cases/games/base" --fused fused.csv --membership membership.csv [--split validation] [--json result.json]
madi-score-fusion --domain games --tier hard --fused fused.csv --membership membership.csv
```

From Python: `from madi_bench.evaluation import score_fusion, score_normalization`; both take a task folder and return a dict. `--tasks-root` (or `MADI_BENCH_USECASES`) points the tools at another copy of the task tree. As a check, `madi-score-fusion --task "use cases/games/base" --fused "results/best of breeds/games/baseline/fused.csv"` gives 0.6570, the P2 Games cell of Table 7.

### End-to-end metrics

- **Submit** the same fused table and membership table as for fusion.
- **References:** *ground truth* = the fusion test split; *silver* = P1's fused output `use cases/<domain>/base/output/data_fusion/fused.csv` (Papers: `fused.csv.gz`), restricted to its clusters with at least two source records. Only the base tasks have a silver reference.
- **Metrics:** the reference-free metrics (record gain, density gain, output density, fusion ratio, schema validity) and the reference-based metrics (entity recovery, value drift, value-density Δ, schema validity Δ, BCubed precision, recall, and F1, fusion accuracy, fully-correct rate) defined in Table 3 of the paper. They are computed by `PyDI.evaluation.compute_e2e_panel` at the pinned commit, with two exceptions in the paper's ground-truth rows: the fusion accuracy is the all-gold accuracy of `madi-score-fusion` (Table 7), and the fully-correct rate is computed with the same strict comparison rules (`reproduction/tables/table8/`). `score_submission --e2e` gives the reference-free and ground-truth rows; `python -m reproduction.scoring.e2e_panels --domain <domain> --out <folder>` computes the reference-free, silver-reference, and ground-truth rows of Table 8. The panel also reports a composite score, which the paper does not use.

---

## Reference pipelines and results

| | Pipeline | Code | Outputs |
|---|---|---|---|
| P1 | Human-designed PyDI pipelines, one notebook per base task, written by master and PhD students | `use cases/<domain>/base/*.ipynb` | `use cases/<domain>/base/output/` (see its `README.md`) |
| P2 | Best-of-breed pipeline: a committee of methods per step, the validation winner of each step is passed on | [`pipelines/`](pipelines/) | [`results/best of breeds/`](results/best%20of%20breeds/) |
| P3 | LLM workflow (Steiner and Bizer, *Automatic End-to-End Data Integration using Large Language Models*, ICDE 2026 Workshops) | [`baselines/llm-pipeline/`](baselines/llm-pipeline/) | [`results/llm pipeline/`](results/llm%20pipeline/) |
| P4 | Claude Code 2.1.280 with Claude Opus 5.5, without PyDI and without labeled data | skills and prompts in [`baselines/claude-code/`](baselines/claude-code/) | [`results/claude code/`](results/claude%20code/) |

[`results/scores_v2.json`](results/scores_v2.json) holds the scores of the stored outputs of P1 to P3 under the rules above, `results/claude code/scores/` those of P4, and `results/paper_tables/` the measurements behind the other cells of the paper's tables.

**Running the pipelines.**

- **P1:** open the notebook of a base task in `use cases/<domain>/base/` (`companies_workflow.ipynb`, `games_workflow.ipynb`, `music_workflow.ipynb`, `papers_workflow_minimal.ipynb`, `products_workflow_minimal.ipynb`) with the task folder as the working directory and run all cells. The notebooks write into the task's `output/` folder, so run them on a copy of the repository, or headless with `reproduction/p1/run_p1.sh <domain> <run_root>`, which runs a copy of the task folder under `<run_root>`, a folder outside `use cases/`. The notebooks of Companies, Games, and Music call an OpenAI model for schema matching; export `OPENAI_API_KEY` first.
- **P2:** train the Ditto and SC-Block checkpoints of a domain, then run the pipeline per task (the SLURM scripts write their logs into `logs/`):

  ```bash
  mkdir -p logs
  DOMAIN=games PYDI_VENV=.venv sbatch cluster/slurm/run_bob_em_train.sbatch
  # once the training job has finished; VARIANT: baseline, easy, medium, hard
  DOMAIN=games VARIANT=baseline PYDI_VENV=.venv sbatch cluster/slurm/run_best_of_breed.sbatch
  ```

  Without SLURM, `DOMAIN=games PYDI_VENV=.venv bash cluster/slurm/run_bob_em_train.sbatch` trains the checkpoints and `python pipelines/scripts/run_best_of_breed.py --config pipelines/configs/games.yaml --variant baseline --out <folder> --ditto-checkpoint-override pipelines/games/checkpoints/em_matching/ditto/baseline/best --sc-block-checkpoint-override pipelines/games/checkpoints/em_blocking/sc_block/baseline/best` runs the pipeline (for a variant: `--variant easy` and the `variant_easy` checkpoints; without the two overrides, the Ditto and SC-Block committee members are left out).
- **P3:** `baselines/llm-pipeline/scripts/run_pipeline.py`; see [`baselines/llm-pipeline/README.md`](baselines/llm-pipeline/README.md).
- **P4:** Claude Code with the skills in `baselines/claude-code/skills/`, the orientation prompt `baselines/claude-code/prompts/orientation.md`, and the task brief and submission contract of the task (base tasks: `baselines/claude-code/prompts/<domain>/`; the copies of all 20 runs: `results/claude code/<domain>/<tier>/prompts/`); see [`baselines/claude-code/README.md`](baselines/claude-code/README.md).

The outputs are scored with the tools of [Evaluating a system](#evaluating-a-system).

---

## Generating variants

The variants were generated with the code in [`usecases_synthetic/`](usecases_synthetic/) (`madi-generate-variant --domain <domain> --level <easy|medium|hard>`; see [`usecases_synthetic/README.md`](usecases_synthetic/README.md) and the knob cards in [`knobs/`](knobs/)). The generator writes its output into `use cases/<domain>/<level>/` and so overwrites the shipped task; run it on a copy of the repository. The committees that measure the difficulty of the variants (Tables 11 and 12 of the paper) write into `usecases_synthetic/baselines/` and `usecases_synthetic/validation/`, which hold the measurements of the paper, so run them on a copy as well:

```bash
mkdir -p logs
DOMAIN=games PYDI_VENV=.venv sbatch cluster/slurm/run_em_train_committee.sbatch               # Ditto and SC-Block checkpoints
DOMAIN=games PYDI_VENV=.venv WITH_LLM=true sbatch cluster/slurm/run_measure_validate.sbatch   # madi-measure-baseline, then madi-validate-variant per level
```

---

## Citation

If you use MaDI-Bench, please cite:

```bibtex
@misc{steiner2026madibench,
  title         = {MaDI-Bench: An End-to-End Data Integration Benchmark},
  author        = {Steiner, Aaron and Peeters, Ralph and Bizer, Christian},
  year          = {2026},
  eprint        = {2606.30371},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  url           = {https://arxiv.org/abs/2606.30371}
}
```
