# Best-of-breed pipeline run — games

Total runtime: 7302.8 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `coma_hybrid` | f1 | 1.0000 | 1.0000 | 188.5 |
| norm | `passthrough` | macro_f1 | 0.9608 | 0.9608 | 0.6 |
| em_blocking | `standard_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9905 | 0.9905 | 2978.8 |
| em_matching | `ditto_plm` | f1 | 0.6686 | 0.6686 | 2978.8 |
| refinement | `baseline` | f1 | 0.0000 | 0.6610 | 1.5 |
| fusion | `accusim_only` | macro_accuracy | 0.6666 | 0.6214 | 4271.2 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-27, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
