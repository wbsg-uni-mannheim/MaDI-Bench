# Best-of-Breed Results

Outputs of the **best-of-breed pipeline (P2)**, one of the three reference pipelines used to validate MaDI-Bench. At each integration step, P2 runs a committee of competing methods from the literature, scores them on the validation set, and chains the per-step winners into a single pipeline.

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
| `correspondences.csv` | Predicted schema correspondences. |
| `per_stage_summary.csv` | Per-stage scores for the run. |
| `stage_*_selection.json` | The method selected at each pipeline stage (schema matching, normalization, blocking, matching, refinement, fusion). |
| `effective_committees/` | The committee members evaluated per stage. |
| `em_per_pair_test_f1.json` | Entity-matching F1 on the test pairs. |
| `e2e_panel/`, `e2e_panel_sr/`, `e2e_panel_fixed/` | End-to-end metric panels (reference-free, silver-reference, and ground-truth views). |
| `summary.md` | Human-readable run summary. |
