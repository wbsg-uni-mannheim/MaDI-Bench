# P1 outputs: Products base task

P1 is the paper's human-written pipeline, the notebook `products_workflow_minimal.ipynb` in this task folder.
The files below are the outputs of the P1 run that the paper reports: a headless run of that notebook on a
CPU node on 2026-09-30.

| File | Content | Used for |
|---|---|---|
| `data_fusion/fused.csv` | The fused table: one row per entity, with the target-schema attributes and PyDI's fusion columns (`_id`; `_fusion_sources`, the ids of the source records merged into the row; `_fusion_source_datasets`; `_fusion_confidence`; `_fusion_metadata`). | P1 fusion accuracy (Table 7); silver reference of the end-to-end metrics (Table 8) |
| `data_fusion/fusion_scores_v2.json` | Fusion accuracy of `fused.csv` against this task's fusion gold, validation and test split (v2 gold, strict comparison rules), written by the notebook. The paper prints the test split's `overall_accuracy_all_gold`. | Table 7 |
| `normalization/products_1.csv` to `normalization/products_4.csv` | P1's normalized source tables, one per source. | P1 normalization accuracy (Table 5) |

`results/scores_v2.json` (`results.p1_human.products_base`) holds the same scores. To recompute them, run
from the repository root:

```bash
madi-score-fusion --task "use cases/products/base" --fused "use cases/products/base/output/data_fusion/fused.csv"
madi-score-normalization --task "use cases/products/base" --tables "use cases/products/base/output/normalization" --no-details
```

To run the notebook headless, use `reproduction/p1/run_p1.sh products <run_root>` (see the header of
that script).

The other files in this folder are diagnostics and intermediate files of the notebook and of the preparation
notebooks in `../utils/`.
