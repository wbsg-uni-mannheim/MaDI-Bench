# P1 outputs: Games base task

P1 is the paper's human-written pipeline, the notebook `games_workflow.ipynb` in this task folder. The files
below are the outputs of the P1 run that the paper reports: a headless run of that notebook on a CPU node on
2026-09-24.

| File | Content | Used for |
|---|---|---|
| `data_fusion/fused.csv` | The fused table: one row per entity, with the target-schema attributes and PyDI's fusion columns (`_id`; `_fusion_sources`, the ids of the source records merged into the row; `_fusion_source_datasets`; `_fusion_confidence`; `_fusion_metadata`). | P1 fusion accuracy (Table 7); silver reference of the end-to-end metrics (Table 8) |
| `data_fusion/fusion_scores_v2.json` | Fusion accuracy of `fused.csv` against this task's fusion gold, validation and test split (v2 gold, strict comparison rules), written by the notebook. The paper prints the test split's `overall_accuracy_all_gold`. | Table 7 |
| `normalization/dbpedia.csv`, `normalization/metacritic.csv`, `normalization/sales.csv` | P1's normalized source tables, one per source. | P1 normalization accuracy (Table 5) |

`results/scores_v2.json` (`results.p1_human.games_base`) holds the same scores. To recompute them, run from
the repository root:

```bash
madi-score-fusion --task "use cases/games/base" --fused "use cases/games/base/output/data_fusion/fused.csv"
madi-score-normalization --task "use cases/games/base" --tables "use cases/games/base/output/normalization" --no-details
```

To run the notebook headless, use `reproduction/p1/run_p1.sh games <run_root>` (see the header of that
script). The notebook calls an OpenAI model for schema matching; export `OPENAI_API_KEY` first.

The other files in this folder are diagnostics and intermediate files of the notebook.
