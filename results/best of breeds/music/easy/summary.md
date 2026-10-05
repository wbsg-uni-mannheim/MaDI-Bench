# Best-of-breed pipeline run — music

Total runtime: 3089.6 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `coma_hybrid` | f1 | 1.0000 | 1.0000 | 201.4 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.8952 | 0.8952 | 1.0 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9993 | 0.9993 | 1871.3 |
| em_matching | `ditto_plm` | f1 | 0.9486 | 0.9486 | 1871.3 |
| refinement | `baseline` | f1 | 0.9772 | 0.9471 | 1.1 |
| fusion | `truthfinder_only` | macro_accuracy | 0.7106 | 0.8543 | 1135.4 |

## End-to-end metric panel


## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-26, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
