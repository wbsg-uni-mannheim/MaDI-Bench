# Best-of-breed pipeline run — products

Total runtime: 1557.7 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `duplicate_majority` | f1 | 1.0000 | 1.0000 | 447.3 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.6439 | 0.6439 | 2.9 |
| em_blocking | `embedding_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9877 | 0.9877 | 1371.2 |
| em_matching | `ditto_plm` | f1 | 0.9693 | 0.9693 | 1371.2 |
| refinement | `baseline` | f1 | 0.5845 | 0.5914 | 0.2 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.7383 | 0.7319 | 104.9 |

## End-to-end metric panel


## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-26, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
