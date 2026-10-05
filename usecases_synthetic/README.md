# usecases_synthetic

Code, configuration, and tests for the variant-generation pipeline that derives the *easy*, *medium*, and *hard* task variants from the base tasks.

The generated tasks themselves live in [../use cases/](../use%20cases/); this directory holds the generator that produces them. See [OVERVIEW.md](OVERVIEW.md) for the design.

**Start here:** [PIPELINE.md](PIPELINE.md) — the ordered list of scripts to run in sequence. Read that file first; it's the runbook.

## Install

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs this repository in editable mode via the root
`pyproject.toml`. The variant pipeline depends on
[`PyDI`](https://github.com/wbsg-uni-mannheim/PyDI) for the data-integration
base classes and evaluators; the default install resolves it from GitHub as
`uma-pydi[synthetic]` at the commit that `pyproject.toml` pins (cd51e25).

If you have a sibling PyDI checkout at that commit and want to use it instead:

```bash
python -m pip install -e ../PyDI[synthetic]
python -m pip install -e . --no-deps
```

The install exposes the main reproducibility entry points:

```bash
madi-generate-variant --domain companies --level easy
madi-measure-baseline --domain companies
madi-validate-variant --domain companies --level easy
```

The same modules can still be run directly:

```bash
python -m usecases_synthetic.scripts.generate_variant --domain companies --level easy
python -m usecases_synthetic.scripts.measure_baseline --domain companies
python -m usecases_synthetic.scripts.validate_variant --domain companies --level easy
```

## Layout

```
usecases_synthetic/
  scripts/
    build_pool.py          Build the pooled-positives CSV per domain from the
                           EM gold, the P1 matcher's correspondences and Ditto
                           predictions (companies, games, music)
  pools/
    <domain>/
      pooled_positives.csv Merged pool in PyDI (id1, id2, score) + extra cols
      pool_stats.json      Counts, agreement breakdown, coverage notes
```

## Scope 

- **Domains covered:** companies, games, music, products, and papers.
- **Pool sources:** companies, games and music combine the positives of the EM gold, the correspondences of the P1 notebook's rule-based matcher and the predictions of a per-domain Ditto matcher, with an LLM deciding the pairs on which P1 and Ditto disagree ([scripts/build_pool.py](scripts/build_pool.py)); products uses the WDC `cluster_id` grouping ([scripts/build_pool_products.py](scripts/build_pool_products.py)); papers uses exact DOI matches and the EM gold positives ([scripts/build_pool_papers.py](scripts/build_pool_papers.py)).
- **Protection-set semantics** (not replacement gold): see [../knobs/cross_cutting.md](../knobs/cross_cutting.md#gold-standard-incompleteness-and-pooling).

## Building the pools

```bash
python usecases_synthetic/scripts/build_pool.py --domain companies
python usecases_synthetic/scripts/build_pool.py --domain games
python usecases_synthetic/scripts/build_pool.py --domain music
```

Or all at once:

```bash
python usecases_synthetic/scripts/build_pool.py --all
```

Outputs land in `usecases_synthetic/pools/<domain>/`.
