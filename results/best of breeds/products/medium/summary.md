# Best-of-breed pipeline run — products

Total runtime: 1837.5 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 426.3 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.5336 | 0.5336 | 2.5 |
| em_blocking | `embedding_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9889 | 0.9889 | 1667.8 |
| em_matching | `ditto_plm` | f1 | 0.8602 | 0.8602 | 1667.8 |
| refinement | `baseline` | f1 | 0.6082 | 0.5266 | 0.2 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.6170 | 0.6342 | 87.7 |

## End-to-end metric panel


## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-26, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
