# LLM workflow (P3)

P3 is the end-to-end data integration workflow of Steiner and Bizer [1]. It uses
the methods of the [PyDI](https://github.com/wbsg-uni-mannheim/PyDI) framework,
and an LLM configures them. In MaDI-Bench, P3 uses GPT-5.5 for schema matching,
taxonomy mapping, and the fusion configuration, and GPT-5.2 for labeling. Its
outputs and scores on the 20 tasks are in
[`results/llm pipeline`](../../results/llm%20pipeline).

## Steps

1. **Schema matching:** one prompt with the source columns, sample values, and
   the target schema.
2. **Normalization:** the LLM maps values to the taxonomies of the task. A column
   profiler selects a normalization function for the other attributes.
3. **Entity matching:** an embedding blocker generates candidate pairs. The
   workflow labels a sample of them with the LLM and builds its own training and
   validation sets. It trains and validates rule-based matchers and
   similarity-feature models such as random forests and XGBoost on them. It does
   not use the training and validation sets of the benchmark.
4. **Fusion:** the LLM assigns each attribute a conflict-resolution function of
   PyDI. The workflow builds several such configurations, some of them tuned on
   the fusion validation set of the task, and keeps the one with the highest
   accuracy on that set.

## Contents

| Path | Contents |
|---|---|
| `scripts/run_pipeline.py` | Entry point of the workflow |
| `scripts/pipeline_utils.py` | Helper functions of the entry point |
| `scripts/prepare_inputs.py` | Converts the five base tasks to the input format of the workflow |
| `scripts/sync_schema_gold.py` | Schema-matching gold of the converted base tasks (used by `prepare_inputs.py`) |
| `manifests/` | One manifest per task: input folders, source pairs, entity matching test sets, excluded columns |
| `prompts/` | The LLM prompts, by step |
| `PyDI/` | The PyDI version the workflow runs with |
| `pyproject.toml` | Package definition and dependencies |

## Installation

The workflow brings its own PyDI version. Install it in a separate Python 3.12
environment:

```bash
cd baselines/llm-pipeline
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e . faiss-cpu xgboost
```

The workflow calls the OpenAI API. Set `OPENAI_API_KEY` in the environment or in
a `.env` file in `baselines/llm-pipeline/`.

## Running a task

Run all commands from `baselines/llm-pipeline/`. First convert the base tasks:

```bash
python scripts/prepare_inputs.py
```

The script writes `DI-Bench/usecases/<domain>/input/`. The sources of Papers and
Products become CSV files, and their fusion validation and test sets become XML
files. The variants need no conversion. Their manifests read `use cases/`
directly.

Then run a task with its manifest:

```bash
python scripts/run_pipeline.py --manifest manifests/games.json \
    --training-set-construction active --compare-provided-validation \
    --fusion-case all --fusion-val-generation-mode all
```

The manifests are `manifests/<domain>.json` for the base tasks and
`manifests/<domain>_<tier>.json` for the variants (`easy`, `medium`, `hard`). A
base run writes to `DI-Bench/runs/<domain>/`, a variant run to
`DI-Bench/runs/augmented/<domain>/<tier>/`.

With `--fusion-case all`, the run builds every fusion configuration and writes
one fused table per configuration. `fusion/best_case_comparison.json` compares
them on the fusion validation set. The results in `results/llm pipeline` use the
configuration with the highest accuracy on that set. The runtimes in the paper
were measured with `--official-validation-fusion-only`, which restricts the
fusion step to the validation set of the task, and with cached LLM labels
(`results/llm pipeline/timing/`).

The models are set with `PYDI_SCHEMA_LLM_MODEL`, `PYDI_LABELING_LLM_MODEL`, and
`PYDI_FUSION_LLM_MODEL`. The defaults are `gpt-5.5`, `gpt-5.2`, and the schema
model. The outputs are scored with the tools in
[Evaluating a system](../../README.md#evaluating-a-system).

## Reference

[1] Aaron Steiner and Christian Bizer. Automatic End-to-End Data Integration
using Large Language Models. Beyond SQL Workshop 2026: AI for Complex Data
Management, co-located with ICDE 2026.
