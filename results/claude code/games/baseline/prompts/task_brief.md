# Task brief: games

Integrate the source tables under `task/input/data/` into one consolidated
table conforming to `task/input/schemamatching/target_schema.json`.

## What counts as one entity

One entity is one video game on one specific platform. The same game released on another platform is a different entity.

## Workspace inventory

- task/input/data/dbpedia.csv (4,708,497 bytes)
- task/input/data/dbpedia_metadata.json (2,532 bytes)
- task/input/data/metacritic.csv (2,507,396 bytes)
- task/input/data/metacritic_metadata.json (2,811 bytes)
- task/input/data/sales.csv (789,396 bytes)
- task/input/data/sales_metadata.json (3,107 bytes)
- task/input/schemamatching/ESRB_Rating_Taxonomy.csv (1,330 bytes)
- task/input/schemamatching/Gaming_Platforms_Taxonomy.csv (6,725 bytes)
- task/input/schemamatching/Video_Game_Genres_Taxonomy.csv (9,023 bytes)
- task/input/schemamatching/target_schema.json (4,304 bytes)

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
