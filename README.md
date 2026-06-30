# MaDI-Bench: An End-to-End Data Integration Benchmark

**Aaron Steiner\*, Ralph Peeters\*, Christian Bizer**
Data and Web Science Group, University of Mannheim, Germany
*\*These authors contributed equally to this work.*

**Paper:** https://arxiv.org/abs/2606.30371 · **Website:** https://wbsg-uni-mannheim.github.io/MaDI-Bench/ 

---

## Abstract

> Data integration combines heterogeneous data sets into a single, coherent representation. Data integration involves a sequence of interdependent tasks including schema matching, value normalization, entity blocking, entity matching, and data fusion. Existing benchmarks either evaluate these steps in isolation or cover only incomplete versions of the data integration pipeline, omitting specific steps. The lack of public end-to-end data integration benchmarks hinders research on data integration methods that address the integration process as a whole. The Mannheim Data Integration Benchmark (MaDI-Bench) fills this gap. MaDI-Bench is the first benchmark for the end-to-end integration of relational tables covering all steps of the integration process. MaDI-Bench contributes (i) a set of base end-to-end data integration tasks spanning several application domains, each requiring the full schema matching, value normalization, entity matching, and conflict resolution pipeline; and (ii) a generic method for deriving task variants that mitigates rapid benchmark saturation as data integration systems advance. We validate the benchmark using human-engineered pipelines, a best-of-breed pipeline, and an LLM-based pipeline. The validation demonstrates the utility of the benchmark for measuring the step-wise as well as the end-to-end performance of data integration pipelines. All benchmark artifacts are available for public download..

**Keywords:** data integration, data lakes, schema matching, entity matching, data fusion, end-to-end evaluation, large language models

---

## What is MaDI-Bench?

The Mannheim Data Integration Benchmark (MaDI-Bench) is the first public benchmark for evaluating the **full, end-to-end integration of relational tables**. Each MaDI task takes several heterogeneous source tables as input and asks the system under test to produce a single output table that conforms to a given target schema and contains only a single record per real-world entity. Solving a task requires to handle all steps of the data integration process together:

**Schema Matching → Value Normalization → Entity Blocking → Entity Matching → Data Fusion**

MaDI-Bench provides ground truth for every step of the integration process *and* for the final output, so a system can be scored step-by-step and end-to-end. As errors propagate through the pipeline, end-to-end scoring reveals their impact on the final integrated result..

What the benchmark provides:

- **20 integration tasks across 5 domains** — Games, Companies, Music, Products, and Scientific Papers. Each domain ships one base task plus *easy*, *medium*, and *hard* variants.
- **Ground truth for every step** — a gold schema mapping, labeled entity-matching train/validation/test splits, and human-verified data fusion validation and test sets.
- **A variant-generation method** built on eight controllable *difficulty knobs*, which derive easier and harder versions of each base task. The easy variants keep simpler, cheaper methods competitive; the hard variants preserve headroom as systems improve, so the benchmark stays useful over time.
- **Validation runs from three reference pipelines** — a human-engineered pipeline, a best-of-breed pipeline, and an LLM-based pipeline.

Across the five base tasks, the artifacts include more than **93,000 labeled record pairs** for entity matching, **1,000 human-verified fused records** carrying close to **11,000 human-verified attribute values**, and a gold schema mapping per task. All artifacts are released using common formats (CSV, JSON, XML).

The data integration pipelines that are used for the validation of the benchmark build on the **[PyDI - Data Integration Framework](https://github.com/wbsg-uni-mannheim/PyDI)**, which provides alternative integration methods and a specialized evaluator classes for each step of the data integration pipeline.

---

## Repository structure

| Path | What it holds |
|---|---|
| [`use cases/`](use%20cases/) | **The benchmark itself** — all 20 integration tasks, with inputs, ground truth, and reference outputs. |
| [`results/`](results/) | **Validation runs** of the three reference pipelines: [`best of breeds/`](results/best%20of%20breeds/) and [`llm pipeline/`](results/llm%20pipeline/). |
| [`knobs/`](knobs/) | Specification of the eight difficulty knobs used to generate the task variants. |
| [`difficulty_dimensions.md`](difficulty_dimensions.md) | The underlying per-stage difficulty dimensions the knobs operationalize. |
| [`usecases_synthetic/`](usecases_synthetic/) | Code, configuration, and tests for the variant-generation pipeline. |


### `use cases/` — the benchmark tasks

The core of the repository. One directory per domain, and within each domain one directory per difficulty level:

```
use cases/<domain>/<base|easy|medium|hard>/
├── input/
│   ├── data/              # source tables (CSV) + a schema.org metadata file per source (JSON)
│   ├── schemamatching/    # target_schema.json (JSON Schema with value constraints),
│   │                      # sm_mapping_gold.json (gold schema mapping), taxonomy CSVs
│   ├── entitymatching/    # labeled pair splits: <srcA>_2_<srcB>_{train,val,test}.csv
│   └── fusion/            # annotated fusion records plus *_better_readability.csv views
├── config/                # difficulty.yaml for the variant (variants only)
└── output/                # reference outputs from the human pipeline (metrics, schema
                           # matching, blocking evaluation, cluster analysis, data fusion,
                           # dataset profiles, logs)
```

The `base/` task additionally ships a `<domain>_workflow.ipynb` notebook showing the integration workflow end to end.

**Inputs to a task:** the source tables with their metadata, the target schema with its constraints and taxonomies, and the labeled entity-matching and fusion validation sets.

**Expected output:** a single fused table that conforms to the target schema and contains one record per real-world entity. For step-level evaluation, a system additionally reports the proposed schema mapping and the matched record pairs.

### `results/` — validation runs

Outputs of the two automated reference pipelines, organized as `<domain>/<baseline|easy|medium|hard>/`:

- **[`results/best of breeds/`](results/best%20of%20breeds/)** — the best-of-breed pipeline (P2), which runs a committee of competing methods at each step, scores them on the validation set, and chains the per-step winners. Each run keeps the fused output table, predicted correspondences, per-stage scores, and the end-to-end panel.
- **[`results/llm pipeline/`](results/llm%20pipeline/)** — the LLM-based pipeline (P3), which uses an LLM to configure each integration step. Each run keeps the pipeline metrics, the end-to-end report, and the per-step fusion, schema-matching, and entity-resolution outputs.

The human-engineered pipeline (P1) is the silver reference; its outputs are stored as the reference `output/` inside each task under `use cases/`.

---


## Getting started

The tasks are plain files, so any integration system can consume them directly:

- **Source tables** are CSV files under each task's `input/data/`, each with a schema.org metadata file describing provenance, columns, and publication date.
- **Target schema** is a JSON Schema (`input/schemamatching/target_schema.json`) that fixes the output columns, per-attribute value constraints, and the taxonomy level at which categorical values should be fused.
- **Gold schema mapping**, **entity-matching splits**, and **fusion validation/test sets** provide the ground truth for scoring each step.

To reproduce or compare against the reference pipelines and to use the per-step evaluator classes, see **[PyDI](https://github.com/wbsg-uni-mannheim/PyDI)**. The base-task notebooks under `use cases/<domain>/base/` are a good starting point for understanding a full workflow.

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
