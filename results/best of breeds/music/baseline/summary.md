# Best-of-breed pipeline run — music

Total runtime: 2841.4 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 159.8 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.9884 | 0.9884 | 1.3 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9976 | 0.9976 | 1917.6 |
| em_matching | `ditto_plm` | f1 | 0.9265 | 0.9484 | 1917.6 |
| refinement | `baseline` | f1 | 0.9592 | 0.9484 | 0.6 |
| fusion | `accusim_only` | macro_accuracy | 0.8784 | 0.7692 | 843.4 |

## End-to-end metric panel


## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-26, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
