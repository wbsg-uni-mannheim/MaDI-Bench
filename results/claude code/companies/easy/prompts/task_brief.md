# Task brief: companies

Integrate the source tables under `task/input/data/` into one consolidated
table conforming to `task/input/schemamatching/target_schema.json`.

## What counts as one entity

One entity is one company that is listed or reported in its own right. Members of the same corporate group that are listed separately are separate entities, and a part of a company — a subsidiary, regional arm, division, or brand — is not the company itself.

## Workspace inventory

- task/input/data/dbpedia.csv (1,203,315 bytes)
- task/input/data/forbes.csv (305,942 bytes)
- task/input/data/fullcontact.csv (112,614 bytes)
- task/input/schemamatching/CLDR_Country_Taxonomy.csv (4,278 bytes)
- task/input/schemamatching/GICS_Industry_Taxonomy.csv (17,668 bytes)
- task/input/schemamatching/target_schema.json (4,333 bytes)

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
