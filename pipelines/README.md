# pipelines/

Best-of-breed sequential data-integration pipeline framework.

At each pipeline stage (SM → Norm → EM blocking → EM matching →
Refinement → Fusion) every committee member runs on the upstream stage's
winner output, and the member with the best validation score wins the
stage. The winner's output flows to the next stage; the loser outputs are
discarded. In replay mode (the default, used for the paper's runs) each
member runs with the hyperparameters of its committee YAML; sweep mode
additionally searches a member's hyperparameter grid where a sweep is
implemented and locks the val-best setting. The Ditto and SC-Block
members load checkpoints that are trained beforehand
(`cluster/slurm/run_bob_em_train.sbatch`).

This is **separate** from the synthetic-pipeline committee tuning
under [usecases_synthetic/](../usecases_synthetic/). That tuning runs
on per-stage "perfect" inputs (each stage independently against its
own gold). The best-of-breed framework reuses the committee
hyperparameter grids but writes its own sweep results + model
checkpoints under `pipelines/<domain>/`, and re-trains learned
matchers (Ditto, Magellan classifier) from scratch on the
chained-pipeline state.

The commands are in the [main README](../README.md#reference-pipelines-and-results).

## Layout

```
pipelines/
├── README.md                  # this file
├── configs/
│   ├── products.yaml          # per-domain pipeline config
│   └── ...
├── lib/
│   ├── pipeline.py            # BestOfBreedPipeline orchestrator
│   ├── bundle.py              # PipelineState + bundle loader
│   ├── stage_runners.py       # per-stage runners (YAML-default mode)
│   ├── sweep_harness.py       # per-stage chained-sweep runners
│   ├── sweep.py               # reads val-best hyperparameters from usecases_synthetic tuning caches
│   └── report.py              # artifact writers
├── scripts/
│   ├── run_best_of_breed.py            # main entrypoint
│   └── compare_to_human_baseline.py    # panel vs notebook side-by-side
├── tests/
│   └── ...
└── <domain>/                  # run outputs (the P2 runs of the paper are in results/best of breeds/)
    ├── sweeps/
    │   ├── sm/
    │   │   ├── sweep.json     # every (member, HP) → val + test
    │   │   └── winners.json   # per-member val-best HP + winner
    │   ├── norm/
    │   ├── em_blocking/
    │   ├── em_matching/
    │   ├── refinement/
    │   └── fusion/
    ├── checkpoints/
    │   ├── em_matching/
    │   │   └── ditto/
    │   │       ├── <hp_hash>/   # per-HP-combo Ditto checkpoint
    │   │       └── winner/      # symlinks/copies of val-best checkpoint
    │   └── ...
    ├── e2e_panel/                  # final metric panel artifacts
    ├── per_stage_summary.csv
    ├── comparison.md               # vs human-baseline notebook
    └── summary.md
```

## Running

```
# Full chained sweep + pipeline run (long; one-time per domain)
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/products.yaml \
    --mode sweep \
    --out pipelines/products/

# Replay: every member with the hyperparameters of its committee YAML (default; deterministic)
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/products.yaml \
    --mode replay \
    --out pipelines/products/replay_<id>/
```

## Conventions

- **No model reuse** from `usecases_synthetic/cache/`. Ditto / sc_block
  / Magellan classifier are trained from scratch with the
  pipeline-specific stage k-1 winner output, checkpointed under
  `pipelines/<domain>/checkpoints/`.
- **HP grids reused, results not.** Sweep grids come from the
  existing `_tune_<stage>_committee.py SPECS` dicts (imported
  directly). Sweep outputs go to `pipelines/<domain>/sweeps/`.
- **Run outputs.** Sweeps, checkpoints, and run folders go to
  `pipelines/<domain>/`; the P2 runs of the paper are in
  `results/best of breeds/`.
