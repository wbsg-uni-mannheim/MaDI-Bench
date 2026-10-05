# Best-of-breed pipeline run — papers

Total runtime: 27988.5 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9859 | 0.9859 | 324.6 |
| norm | `passthrough` | macro_f1 | 0.5981 | 0.5981 | 34.0 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9955 | 0.9955 | 25650.6 |
| em_matching | `magellan` | f1 | 0.9923 | 0.9923 | 25650.6 |
| refinement | `baseline` | f1 | 0.9955 | 0.9923 | 65.8 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.4490 | 0.4766 | 2042.9 |

## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-26, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
