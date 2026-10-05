# Best-of-Breed Results

Outputs of the **best-of-breed pipeline (P2)**, one of the four validation pipelines of MaDI-Bench. At each integration step, P2 runs a committee of competing methods from the literature, scores them on the validation set, and chains the per-step winners into a single pipeline.

## Layout

```text
results/best of breeds/<domain>/<variant>/
```

- **Domains:** `companies`, `games`, `music`, `papers`, `products`.
- **Variants:** `baseline`, `easy`, `medium`, `hard` (`baseline` is the base task).

Each run directory keeps the result artifacts for that domain and variant:

| Artifact | Contents |
|---|---|
| `fused.csv` | The fused output table. |
| `correspondences.csv` | The matched record pairs (entity matching output). |
| `per_stage_summary.csv` | Per-stage scores for the run. |
| `stage_*_selection.json` | The method selected at each pipeline stage (schema matching, normalization, blocking, matching, refinement, fusion). |
| `effective_committees/` | The committee members evaluated per stage. |
| `em_per_pair_test_f1.json` | Companies, Games, Music, and Papers base runs: entity-matching F1 per source pair on the test split, of the matchers `ditto_plm` and `magellan` (written by `pipelines/scripts/recompute_em_per_pair.py`). |
| `e2e_panel/` | Companies, Games, Music, and Products runs: the end-to-end metric panel that the run computed (reference-free and ground-truth views). |
| `summary.md` | Human-readable run summary. |

The end-to-end values of Table 8 of the paper are in `results/paper_tables/table8/`; `reproduction/scoring/e2e_panels.py` computes them.
