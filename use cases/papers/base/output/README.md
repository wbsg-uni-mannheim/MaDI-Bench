# P1 outputs: Papers base task

P1 is the paper's human-written pipeline, the notebook `papers_workflow_minimal.ipynb` in this task folder.
The files below are the outputs of the P1 run that the paper reports: a headless run of that notebook on a
CPU node on 2026-09-24.

| File | Content | Used for |
|---|---|---|
| `data_fusion/fused.csv.gz` | The fused table, gzip-compressed (see below): one row per entity, with the target-schema attributes and PyDI's fusion columns (`_id`; `_fusion_sources`, the ids of the source records merged into the row; `_fusion_source_datasets`; `_fusion_confidence`; `_fusion_metadata`). | P1 fusion accuracy (Table 7); silver reference of the end-to-end metrics (Table 8) |
| `data_fusion/fusion_scores_v2.json` | Fusion accuracy of the fused table against this task's fusion gold, validation and test split (v2 gold, strict comparison rules), written by the notebook. The paper prints the test split's `overall_accuracy_all_gold`. | Table 7 |
| `normalization/crossref.csv`, `normalization/dblp.csv`, `normalization/open_alex.csv` | P1's normalized source tables, one per source. | P1 normalization accuracy (Table 5) |

The notebook writes the fused table as `fused.csv` (263 MB). That is above GitHub's file size limit of 100 MB,
so the repository stores it compressed as `fused.csv.gz` (23 MB). The fusion scorer below reads the compressed
file directly. `gunzip -k "use cases/papers/base/output/data_fusion/fused.csv.gz"` restores `fused.csv`; its
SHA-256 is `0d0454b76a482274b116e03cfd4efafde9a1c2a6d7a1de5dc25fade2b273adf9`, the value that
`results/scores_v2.json` records for this input.

`results/scores_v2.json` (`results.p1_human.papers_base`) holds the same scores. To recompute them, run from
the repository root:

```bash
madi-score-fusion --task "use cases/papers/base" --fused "use cases/papers/base/output/data_fusion/fused.csv.gz"
madi-score-normalization --task "use cases/papers/base" --tables "use cases/papers/base/output/normalization" --no-details
```

To run the notebook headless, use `reproduction/p1/run_p1.sh papers <run_root>` (see the header of that
script).

The other files in this folder are diagnostics and intermediate files of the notebook.
