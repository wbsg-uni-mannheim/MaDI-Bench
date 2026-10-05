# Best-of-breed pipeline run — companies

Total runtime: 866.1 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 0.9767 | 0.9767 | 68.4 |
| norm | `passthrough` | macro_f1 | 0.9388 | 0.9388 | 3.6 |
| em_blocking | `bm25_blocker` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9596 | 0.9596 | 457.4 |
| em_matching | `magellan` | f1 | 0.6902 | 0.8929 | 457.4 |
| refinement | `baseline` | f1 | 0.9082 | 0.8929 | 0.6 |
| fusion | `prefer_higher_trust_only` | macro_accuracy | 0.5333 | 0.4590 | 325.1 |

## End-to-end metric panel


## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-26, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
