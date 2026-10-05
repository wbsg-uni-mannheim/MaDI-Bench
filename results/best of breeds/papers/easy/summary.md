# Best-of-breed pipeline run — papers

Total runtime: 4878.0 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `coma_hybrid` | f1 | 1.0000 | 1.0000 | 376.0 |
| norm | `passthrough` | macro_f1 | 0.6466 | 0.6466 | 7.3 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9985 | 0.9985 | 4326.7 |
| em_matching | `magellan` | f1 | 0.9945 | 0.9945 | 4326.7 |
| refinement | `baseline` | f1 | 0.9924 | 0.9898 | 10.8 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.5551 | 0.5807 | 433.2 |

## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-27, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
