# LLM Pipeline Results

Outputs of the **LLM-based pipeline (P3)**, one of the three reference pipelines used to validate MaDI-Bench. P3 uses an LLM to configure each integration step and generates its own machine-labeled training and validation data, so the pipeline stays automatic while scaling.

## Layout

```text
results/llm pipeline/<domain>/<variant>/
```

- **Domains:** `companies`, `games`, `music`, `papers`, `products`.
- **Variants:** `baseline`, `easy`, `medium`, `hard` (`baseline` is the base task).

Each run directory keeps the available result artifacts for that domain and variant:

| Artifact | Contents |
|---|---|
| `pipeline_metrics.{csv,json}` | Step-level scores for the run. |
| `end_to_end_metrics.json`, `end_to_end_report.{csv,txt}` | End-to-end quality metrics and report. |
| `schema_matching/` | Proposed schema mapping outputs. |
| `entity_resolution/` | Blocking, matching, training, and validation outputs. |
| `fusion/` | Fused output and fusion evaluation. |
| `metrics/`, `reporting/` | Aggregated metrics and reporting tables. |
| `pipeline.log` | Run log. |

The `timing/` directory holds runtime measurements for the pipeline.

> Cached embedding artifacts (`entity_resolution/embeddings/`) are intentionally excluded — they are large caches rather than result tables.
