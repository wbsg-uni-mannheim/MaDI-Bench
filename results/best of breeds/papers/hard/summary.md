# Best-of-breed pipeline run — papers

Total runtime: 15872.7 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 318.9 |
| norm | `passthrough` | macro_f1 | 0.5682 | 0.5682 | 3.1 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9499 | 0.9499 | 15457.1 |
| em_matching | `ditto_plm` | f1 | 0.9671 | 0.9671 | 15457.1 |
| refinement | `baseline` | f1 | 0.7028 | 0.6827 | 7.0 |
| fusion | `accusim_only` | macro_accuracy | 0.3768 | 0.3779 | 344.0 |

## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-26, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
