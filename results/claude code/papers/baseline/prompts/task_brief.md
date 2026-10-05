# Task brief: papers

Integrate the source tables under `task/input/data/` into one consolidated
table conforming to `task/input/schemamatching/target_schema.json`.

## What counts as one entity

One entity is one scholarly publication: a paper as it appeared in one venue on one occasion. Records from different libraries that describe that same publication are one entity. A paper published again elsewhere, for example a conference paper and its later journal version, is a separate publication and therefore a separate entity.

## Workspace inventory

- task/input/data/crossref.jsonl (53,275,285 bytes)
- task/input/data/crossref_metadata.json (3,961 bytes)
- task/input/data/dblp.jsonl (21,609,304 bytes)
- task/input/data/dblp_metadata.json (3,172 bytes)
- task/input/data/open_alex.jsonl (27,679,550 bytes)
- task/input/data/open_alex_metadata.json (3,713 bytes)
- task/input/schemamatching/target_schema.json (4,450 bytes)

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
