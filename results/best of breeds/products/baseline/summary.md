# Best-of-breed pipeline run — products

Total runtime: 489.8 s

## Per-stage winners

| Stage | Winner | Metric | Val score | Test score | Runtime (s) |
|---|---|---|---|---|---|
| sm | `llm_openai` | f1 | 1.0000 | 1.0000 | 469.3 |
| norm | `rule_per_attribute_optimal` | macro_f1 | 0.5855 | 0.5855 | 2.3 |
| em_blocking | `sc_block` | pair_completeness (>=0.97 floor; reduction_ratio tiebreak) | 0.9933 | 0.9933 | 300.0 |
| em_matching | `ditto_plm` | f1 | 0.0000 | 0.8548 | 300.0 |
| refinement | `baseline` | f1 | 0.8676 | 0.8548 | 0.2 |
| fusion | `pydi_per_attribute_optimal` | macro_accuracy | 0.7201 | 0.7084 | 95.2 |

## End-to-end metric panel


## Panel warnings

- Source-attribution and synthesis-rate metrics skipped against gold (gold.cell_provenance is None).

## Caveats

- The SM stage (stage_1_sm_selection.json and the sm row) ran separately from the other stages, on 2026-09-26, on the shipped column names (usecases_synthetic/lib/sm_view.py); the total runtime above is that of the other stages' run. The later stages do not read the SM winner.
- Greedy per-stage selection is locally optimal; no joint search across stages.
- **Norm selection was vacuous** (spread=0.0008 < epsilon). Norm members produced near-identical outputs on this input.
