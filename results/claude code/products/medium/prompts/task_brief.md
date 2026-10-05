# Task brief: products

Integrate the source tables under `task/input/data/` into one consolidated
table conforming to `task/input/schemamatching/target_schema.json`.

## What counts as one entity

One entity is one specific product configuration as its manufacturer sells it, not a product family and not a single seller's offer. Every offer of that configuration belongs to one entity; a variant with a different specification is a different entity.

## Workspace inventory

- task/input/data/products_1.csv (455,633 bytes)
- task/input/data/products_2.csv (393,649 bytes)
- task/input/data/products_3.csv (343,682 bytes)
- task/input/data/products_4.csv (297,231 bytes)
- task/input/schemamatching/GPU_Memory_Taxonomy.csv (364 bytes)
- task/input/schemamatching/Product_Type_Taxonomy.csv (624 bytes)
- task/input/schemamatching/Storage_Interface_Taxonomy.csv (677 bytes)
- task/input/schemamatching/target_schema.json (6,316 bytes)

All permitted source data is in this workspace. No labeled training, validation or test data is provided. This baseline has no labeling or fusion-teacher service and no access to label pools. Do not create or request a labeled training/evaluation set. Do not train or fine-tune Ditto or another supervised matcher. Use direct matching decisions, rules, similarities and label-free diagnostics; report their limitations.

## Deliverables (details in refs/SUBMISSION_SPEC.md)

Write into `submission/`:
- `sm_mapping.csv` — your schema mapping
- `correspondences.csv` — matched record pairs across sources
- `membership.csv` — record-to-cluster assignment
- `fused.csv` — the consolidated table (one row per entity as defined above)
- `report.md` — your method notes and label-free diagnostic results
Optional: `blocking/candidates.csv` (your blocking candidate pairs).

## Budget

- Monetary budget: $100.00 of LLM spend; the episode ends when it is exhausted.
- Time budget: 18 hours of wallclock; the episode ends when it runs out.
- You will receive NO interim status messages: finalize submission/ before the budget runs out.

## Notes

none
