# Claude Code baseline (P4)

This directory publishes the seven knowledge skills and prompt snapshots used by
the reported Claude Code baseline: Claude Code 2.1.280 with Claude Opus 5.5,
without PyDI or labeled training and validation data.

## Skills

- [Schema matching](skills/schema-matching/SKILL.md)
- [Normalization](skills/normalization/SKILL.md)
- [Blocking](skills/blocking/SKILL.md)
- [Entity matching](skills/entity-matching/SKILL.md)
- [Refinement and clustering](skills/refinement-clustering/SKILL.md)
- [Fusion](skills/fusion/SKILL.md)
- [Pipeline diagnosis](skills/pipeline-diagnosis/SKILL.md)

The [skill index](skills/INDEX.md) is the index supplied in the run workspace.
All seven skills and the index were verified to be byte-identical across the
five selected base runs. The PyDI-specific skill was not supplied in this
experimental condition and is not included here.

## Prompts and task contracts

The shared [orientation prompt](prompts/orientation.md) describes the available
inputs, restrictions, iteration process, and record keeping. It is also identical
across the five selected base runs. Each task has its original brief and
submission contract:

| Base task | Task brief | Submission contract |
|---|---|---|
| Games | [Brief](prompts/games/task_brief.md) | [Contract](prompts/games/SUBMISSION_SPEC.md) |
| Companies | [Brief](prompts/companies/task_brief.md) | [Contract](prompts/companies/SUBMISSION_SPEC.md) |
| Music | [Brief](prompts/music/task_brief.md) | [Contract](prompts/music/SUBMISSION_SPEC.md) |
| Products | [Brief](prompts/products/task_brief.md) | [Contract](prompts/products/SUBMISSION_SPEC.md) |
| Papers | [Brief](prompts/papers/task_brief.md) | [Contract](prompts/papers/SUBMISSION_SPEC.md) |

Each brief specifies a budget of USD 100 and 18 hours. The agent receives source
tables and metadata, the target schema, taxonomies, and the entity definition.
Web search and fetching are disabled. An embeddings-only endpoint may be
available; labeling services and supervised matcher training are excluded.

## Provenance and scope

The archived skills and prompts are unmodified. [provenance.json](provenance.json)
records the source run identifiers, relative to the AgenticDI repository, and
SHA-256 hashes of the archived files. The runs took place on September 23–24,
2026. Skill and orientation copies were checked against all five run workspaces.

Paths inside these snapshots refer to the original agent workspace: `skills/`,
`refs/task_brief.md`, `refs/SUBMISSION_SPEC.md`, `task/input/`, and `submission/`.
The task contracts also reference harness utilities such as
`scripts/self_check.py`. This release contains the skills and prompts; the
execution harness, task workspaces, generated pipeline code, and evaluator are
not included in this directory. In particular, the original task briefs retain
their recorded input inventories and are not templates for changed datasets.
