# usecases_synthetic

Code, configuration, and tests for the variant-generation pipeline that derives the *easy*, *medium*, and *hard* task variants from the base tasks.

The generated tasks themselves live in [../use cases/](../use%20cases/); this directory holds the generator that produces them. See [OVERVIEW.md](OVERVIEW.md) for the design.

**Start here:** [PIPELINE.md](PIPELINE.md) — the ordered, status-tracked list of scripts to run in sequence. Read that file first; it's the runbook and the source of truth for what's built vs. planned.

## Layout

```
usecases_synthetic/
  scripts/
    build_pool.py          Build pooled-positives CSV per domain by merging
                           existing matcher outputs from two pipelines
  pools/
    <domain>/
      pooled_positives.csv Merged pool in PyDI (id1, id2, score) + extra cols
      pool_stats.json      Counts, agreement breakdown, coverage notes
```

## Scope (v1 prototype)

- **Domains covered:** companies, games, music, products, and papers.
- **Pool sources (reuse only, no new matching runs):**
  1. **PLM-based pipeline** — `automatic-data-integration/scripts/output/<domain>_0302/entity_resolution/matching/`
  2. **Human-baseline pipeline** — `usecases_new/output/<domain>/cluster_analysis/detailed_cluster_info.json`
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
