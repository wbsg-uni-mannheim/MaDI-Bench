# Task brief: music

Integrate the source tables under `task/input/data/` into one consolidated
table conforming to `task/input/schemamatching/target_schema.json`.

## What counts as one entity

One entity is one specific release of an album, EP, or single: a particular issue of that title, not the recording or the album in general. Different issues of the same title by the same artist are different entities.

## Workspace inventory

- task/input/data/discogs.csv (7,270,540 bytes)
- task/input/data/lastfm.csv (1,715,075 bytes)
- task/input/data/musicbrainz.csv (1,410,340 bytes)
- task/input/schemamatching/Music_Genres_Taxonomy.csv (18,462 bytes)
- task/input/schemamatching/target_schema.json (2,691 bytes)

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
