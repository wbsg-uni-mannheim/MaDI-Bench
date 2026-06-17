# DI-Bench Metrics

This folder keeps the paper-facing metrics short. The root files are the ones
to cite; `details/` keeps audit/debug tables.

- `paper_results.csv` and `.json`: one concise table for paper reporting.
- `schema_matching.csv`: official schema-matching precision/recall/F1 against `sm_mapping_gold.json`, plus coverage diagnostics.
- `entity_matching.csv`: official held-out test F1/precision/recall for the final selected configuration, plus an `all_pairs` macro F1 row.
- `fusion_accuracy.csv`: official test accuracy for the selected fusion strategy.
- `consistency.json`: reference-free schema consistency on `fusion/fused_clean.csv`.

Detailed validation sweeps, model comparisons, per-attribute fusion accuracy,
per-column consistency, raw schema mappings, and structural E2E metrics are in
`details/`.
