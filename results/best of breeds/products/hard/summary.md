# Best-of-breed pipeline run — products

Total runtime: 891.6 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9800 | 0.9800 | 438.6 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.5227 | 0.5227 | 1.9 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 1.0000 | 1.0000 | 590.1 |
| em_matching | `ditto_plm` | f1 | 0.7922 | 0.7922 | 590.1 |
| refinement | `baseline` | f1 | 0.6541 | 0.5705 | 0.1 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.3915 | 0.4778 | 49.0 |

## End-to-end metric panel


## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-26, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
