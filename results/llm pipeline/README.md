# LLM Pipeline Results

This directory contains the LLM pipeline outputs copied from
`/Users/aaronsteiner/Documents/GitHub/unsupervised-data-integration/DI-Bench/runs`.

Layout:

```text
results/llm pipeline/<domain>/<variant>/
```

Domains: `companies`, `games`, `music`, `papers`, `products`.
Variants: `baseline`, `easy`, `medium`, `hard`.

The source pipeline names the base case `base`; this repository stores that
case as `baseline` to match the rest of the MaDI-Bench result layout.

Each canonical run directory preserves the available result artifacts for that
domain and variant, including `pipeline_metrics.*`, `pipeline.log`,
`end_to_end_report.*`, `end_to_end_metrics.json`, `fusion/`, `metrics/`,
`reporting/`, `schema_matching/`, and non-embedding `entity_resolution/`
outputs such as matching, training, and validation artifacts.

The `entity_resolution/embeddings/` folders were intentionally excluded. They
are cached embedding artifacts rather than result tables and are large enough to
make the repository unnecessarily heavy.

Source mapping:

| Domain | Variant | DI-Bench source run |
|---|---|---|
| companies | baseline | `runs/companies` |
| companies | easy | `runs/augmented/companies/easy` |
| companies | medium | `runs/augmented/companies/medium` |
| companies | hard | `runs/augmented/companies/hard` |
| games | baseline | `runs/games` |
| games | easy | `runs/augmented/games/easy` |
| games | medium | `runs/augmented/games/medium` |
| games | hard | `runs/augmented/games/hard` |
| music | baseline | `runs/music` |
| music | easy | `runs/augmented/music/easy` |
| music | medium | `runs/augmented/music/medium` |
| music | hard | `runs/augmented/music/hard` |
| papers | baseline | `runs/papers` |
| papers | easy | `runs/augmented/papers/easy` |
| papers | medium | `runs/augmented/papers/medium` |
| papers | hard | `runs/augmented/papers/hard` |
| products | baseline | `runs/products` |
| products | easy | `runs/augmented/products/easy` |
| products | medium | `runs/augmented/products/medium` |
| products | hard | `runs/augmented/products/hard` |

Additional copied outputs:

| Directory | Source |
|---|---|
| `aggregated_results/` | `runs/aggregated_results` |
| `timing/` | `runs/timing` |
| `auxiliary/papers-augmented-easy/` | `runs/papers-augmented-easy` |
| `auxiliary/papers-augmented-medium/` | `runs/papers-augmented-medium` |
| `auxiliary/papers-augmented-hard/` | `runs/papers-augmented-hard` |
| `auxiliary/schema_matching_heuristic_baselines/` | `runs/schema_matching_heuristic_baselines` |
| `auxiliary/schema_matching_heuristic_baselines_fair_instance/` | `runs/schema_matching_heuristic_baselines_fair_instance` |
| `auxiliary/schema_matching_source_to_source_baselines/` | `runs/schema_matching_source_to_source_baselines` |
| `auxiliary/schema_matching_target_schema_20_examples/` | `runs/schema_matching_target_schema_20_examples` |
