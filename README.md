# MaDI-Bench: An End-to-End Data Integration Benchmark

**Aaron Steiner\*, Ralph Peeters\*, Christian Bizer**
Data and Web Science Group, University of Mannheim, Germany
*\*These authors contributed equally to this work.*

🌐 **Website:** https://wbsg-uni-mannheim.github.io/MaDI-Bench/ · 🧰 **Pipeline framework (PyDI):** https://github.com/wbsg-uni-mannheim/PyDI

---

## Abstract

> Data integration combines heterogeneous data sets into a single, coherent representation. Data integration involves a sequence of interdependent tasks including schema matching, value normalization, entity blocking, entity matching, and data fusion. Existing benchmarks either evaluate these steps in isolation or target only incomplete versions of the data integration pipeline leaving out specific steps. The lack of public end-to-end data integration benchmarks hinders research on data integration methods. This paper fills this gap by introducing the Mannheim Data Integration Benchmark (MaDI-Bench), the first benchmark for the end-to-end integration of relational tables covering all steps of the integration process. MaDI-Bench contributes (i) a set of baseline end-to-end data integration tasks spanning several application domains, each requiring the full schema matching, value normalization, entity matching, and conflict resolution pipeline; and (ii) a generic method for deriving variants of these tasks to prevent the quick saturation of the benchmark as data integration systems progress. We validate the benchmark using human-engineered pipelines, a best-of-breed pipeline, and an LLM-based pipeline. The validation shows the utility of the benchmark for measuring the step-wise as well as the end-to-end performance of data integration pipelines. All benchmark artifacts are available for public download.

**Keywords:** data integration, data lakes, schema matching, entity matching, data fusion, end-to-end evaluation, large language models

---

## What is MaDI-Bench?

MaDI-Bench is the first public benchmark for the **full, end-to-end integration of relational tables**. Each task takes several heterogeneous source tables in a domain and asks a system to produce a single fused target table that conforms to a given target schema and contains one record per real-world entity. Solving a task exercises every step of the data integration pipeline together:

**Schema Matching → Value Normalization → Entity Blocking → Entity Matching → Data Fusion**

Rather than evaluating each step on its own dataset, MaDI-Bench provides ground truth for every step *and* for the integrated output, so a system can be scored step-by-step and end-to-end on the same task.

What the benchmark provides:

- **20 integration tasks across 5 domains** — Games, Companies, Music, Products, and Scientific Papers. Each domain ships one base task plus *easy*, *medium*, and *hard* variants.
- **Ground truth for every step** — a gold schema mapping, labeled entity-matching train/validation/test splits, and hand-annotated fusion validation/test records, in addition to the integrated reference output.
- **A variant-generation method** built on eight controllable *difficulty knobs*, which derive easier and harder versions of each base task. The easy variants keep simpler, cheaper methods competitive; the hard variants keep headroom as systems improve, so the benchmark stays useful over time.
- **Validation runs from three reference pipelines** — a human-engineered pipeline, a best-of-breed pipeline, and an LLM-based pipeline.

Across the five base tasks, the artifacts include more than **93,000 labeled record pairs** for entity matching, **1,000 hand-annotated fusion records** carrying close to **11,000 verified attribute values**, and a gold schema mapping per task. All artifacts are released in common formats (CSV, JSON, XML).

The tasks build on and are evaluated with **[PyDI](https://github.com/wbsg-uni-mannheim/PyDI)**, which provides the integration functions and a specialized evaluator class for each step, so a system can be developed and scored without annotating any data of its own.

---

## Repository structure

| Path | What it holds |
|---|---|
| [`use cases/`](use%20cases/) | **The benchmark itself** — all 20 integration tasks, with inputs, ground truth, and reference outputs. |
| [`results/`](results/) | **Validation runs** of the three reference pipelines: [`best of breeds/`](results/best%20of%20breeds/) and [`llm pipeline/`](results/llm%20pipeline/). |
| [`knobs/`](knobs/) | Specification of the eight difficulty knobs used to generate the task variants. |
| [`difficulty_dimensions.md`](difficulty_dimensions.md) | The underlying per-stage difficulty dimensions the knobs operationalize. |
| [`usecases_synthetic/`](usecases_synthetic/) | Code, configuration, and tests for the variant-generation pipeline. |
| [`website/`](website/) | Source of the project website (deployed via GitHub Pages). |

### `use cases/` — the benchmark tasks

The core of the repository. One directory per domain, and within each domain one directory per difficulty level:

```
use cases/<domain>/<base|easy|medium|hard>/
├── input/
│   ├── data/              # source tables (CSV) + a schema.org metadata file per source (JSON)
│   ├── schemamatching/    # target_schema.json (JSON Schema with value constraints),
│   │                      # sm_mapping_gold.json (gold schema mapping), taxonomy CSVs
│   ├── entitymatching/    # labeled pair splits: <srcA>_2_<srcB>_{train,val,test}.csv
│   └── fusion/            # validation_set.xml, test_set.xml (annotated fusion records)
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

## The three reference pipelines

MaDI-Bench is validated with three pipelines that span the design space from hand-built to fully automated:

1. **Human-engineered pipeline (P1)** — domain-specific workflows built on PyDI by data engineers. Scores 100% on the base tasks and serves as the *silver* reference.
2. **Best-of-breed pipeline (P2)** — chains the strongest available method per step (e.g., COMA and Magneto for schema matching; Ditto and Magellan for entity matching; truth-discovery and PyDI heuristics for fusion), selected on the validation set.
3. **LLM-based pipeline (P3)** — an LLM configures each integration step, building its own machine-labeled training data, so the pipeline scales while staying automatic.

The validation reports both **step-level metrics** (schema-matching F1; blocking pair completeness and reduction ratio; entity-matching F1; fusion accuracy) and **end-to-end metrics** organized along three quality dimensions (Coverage, Consistency, Correctness) and three reference levels (reference-free, silver, ground-truth).

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
@inproceedings{steiner2026madibench,
  title     = {MaDI-Bench: An End-to-End Data Integration Benchmark},
  author    = {Steiner, Aaron and Peeters, Ralph and Bizer, Christian},
  year      = {2026},
  note      = {Data and Web Science Group, University of Mannheim}
}
```

> **Note:** publication venue and full citation details to be added once available.

---

## License

See [`LICENSE`](LICENSE). <!-- TODO: add a license file (see notes below). -->
