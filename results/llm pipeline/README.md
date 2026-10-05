# LLM workflow results (P3)

Outputs of the LLM workflow (P3), one of the four pipelines used to validate
MaDI-Bench. The code and the run commands are in
[`baselines/llm-pipeline`](../../baselines/llm-pipeline).

## Layout

```text
results/llm pipeline/<domain>/<variant>/
```

- **Domains:** `companies`, `games`, `music`, `papers`, `products`.
- **Variants:** `baseline`, `easy`, `medium`, `hard` (`baseline` is the base task).

Each run folder holds the outputs of one task:

| Artifact | Contents |
|---|---|
| `schema_matching/` | Schema mappings and the translated source tables |
| `entity_resolution/` | Blocking, matching, and the generated training and validation sets |
| `fusion/` | Fused tables, fusion rules, and the comparison of the fusion configurations on the validation set (`best_case_comparison.json`) |
| `pipeline_metrics.{csv,json}` | Scores per step |
| `end_to_end_metrics.json`, `end_to_end_report.{csv,txt}` | End-to-end metrics |
| `metrics/`, `reporting/` | Aggregated metrics and report tables |
| `scoring/` | Products only: step scores computed with the benchmark scorer |
| `pipeline.log.gz` | Products only: log of the run |

Two further folders:

- `e2e_panels/`: the end-to-end panels behind the P3 cells of Table 8, per
  domain against the silver reference (`silver/`) and the ground truth
  (`rf_gt/`). `cells.json` lists the table values and the SHA-256 of every input.
- `timing/`: runtime measurements of the base tasks. Table 9 uses
  `base_cached_strict_20260609/` for Games, Companies, and Music,
  `papers_cached_alias_20260609/` for Papers, and
  `products_cached_replay_20260930/` for Products. The other two folders hold
  timings with other cache settings.
