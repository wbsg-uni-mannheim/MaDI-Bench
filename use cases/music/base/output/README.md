# P1 outputs: Music base task

P1 is the paper's human-written pipeline, the notebook `music_workflow.ipynb` in this task folder. The files
below are the outputs of the P1 run that the paper reports: a headless run of that notebook on a CPU node on
2026-09-24. The normalized tables were built from that run's output (see the note below the table).

| File | Content | Used for |
|---|---|---|
| `data_fusion/fused.csv` | The fused table: one row per entity, with the target-schema attributes and PyDI's fusion columns (`_id`; `_fusion_sources`, the ids of the source records merged into the row; `_fusion_source_datasets`; `_fusion_confidence`; `_fusion_metadata`). | P1 fusion accuracy (Table 7); silver reference of the end-to-end metrics (Table 8) |
| `data_fusion/fusion_scores_v2.json` | Fusion accuracy of `fused.csv` against this task's fusion gold, validation and test split (v2 gold, strict comparison rules), written by the notebook. The paper prints the test split's `overall_accuracy_all_gold`. | Table 7 |
| `normalization/discogs.csv`, `normalization/lastfm.csv`, `normalization/musicbrainz.csv` | P1's normalized source tables, one per source. | P1 normalization accuracy (Table 5) |

The normalized tables were built from the tables the run exported directly after schema translation, by
applying the two cells of the notebook's section "Step 5: Translate and Normalize" that parse the track lists
and complete partial release dates (for example `2012` becomes `2012-01-01`); matching and fusion use the
tables after these two cells. For every attribute, their values equal the values that the run's fusion step
received. Table 5 reports their accuracy. The export cell of `music_workflow.ipynb` writes the tables after
the two cells.

`results/scores_v2.json` (`results.p1_human.music_base`) holds the same scores. To recompute them, run from
the repository root:

```bash
madi-score-fusion --task "use cases/music/base" --fused "use cases/music/base/output/data_fusion/fused.csv"
madi-score-normalization --task "use cases/music/base" --tables "use cases/music/base/output/normalization" --no-details
```

To run the notebook headless, use `reproduction/p1/run_p1.sh music <run_root>` (see the header of that
script). The notebook calls an OpenAI model for schema matching; export `OPENAI_API_KEY` first.

The other files in this folder are diagnostics and intermediate files of the notebook.
