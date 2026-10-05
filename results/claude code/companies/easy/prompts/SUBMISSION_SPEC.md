# Submission contract

All files go into `submission/`. Use the record ids EXACTLY as they appear in
the source tables' id columns — do not invent, rename, or re-prefix ids.
Ids a loader generated for you (`alpha-0000` style row indices) are for YOUR
internal joins only; a submission using them scores ZERO because nothing can
join them back to the sources.
`scripts/self_check.py` verifies your ids
appear in the source tables — run it after every submission update.
Dataset names are the source file stems (e.g. `alpha.csv` → `alpha`).

## sm_mapping.csv
CSV with header: `source_dataset,source_column,target_dataset,target_column,score`
One row per (source column → target attribute) correspondence.
`target_dataset` may be any consistent label for the target schema.
`score` in [0,1] (your confidence; 1.0 is fine for rule-based decisions).

## correspondences.csv
CSV with header: `id1,id2,score`
One row per matched record PAIR across two different sources (only pairs you
assert are the same entity, as defined in the task brief). Direction does
not matter.

## membership.csv
CSV with header: `record_id,source,cluster_id`
One row per source record that belongs to a fused entity. `cluster_id` must
equal the `_id` of the corresponding row in `fused.csv`. Singleton clusters
(one-record entities) are allowed and encouraged for unmatched records.

## fused.csv
CSV with header: `_id` + the target-schema attribute names.
One row per entity as defined in the task brief, values resolved from the
member records.
Attributes the target schema declares as `type: array` may be serialized
as a JSON list (`["A", "B"]`) or as a `|`-separated string (`A|B`) —
both are accepted equivalently. Do not invent other separators.

## How fused values are compared

A fused value counts as correct when it matches the expected value
under the comparison for its attribute — not by string equality.
Attributes not listed are not part of the fusion score.

| attribute | how it is compared |
|---|---|
| `name` | token-based: word order and punctuation ignored |
| `revenue` | 2% (relative) |
| `assets` | 2% (relative) |
| `keypeople` | compared as a SET of names — every name must match, in any order, ignoring case and accents; a JSON list and a single name are the same set |
| `founded` | compared by YEAR only |
| `country` | token-based: word order and punctuation ignored |
| `city` | token-based: word order and punctuation ignored |
| `industry` | exact match after trimming |

Use these SAME comparisons when you evaluate your own fusion.


## blocking/candidates.csv (required)
CSV with header: `id1,id2` — your FULL blocking candidate set, never a
sample: blocking quality (pair completeness) is scored over exactly what you
submit, so a truncated file reads as catastrophic blocking, and a missing
file leaves the blocking stage unmeasured (self_check fails on it).

## report.md
Free-form: method per stage, your own measured numbers, known weaknesses.

## work/rebuild.sh (required)
One entrypoint that regenerates every data file in `submission/` from your
workspace — your scripts, cached intermediates, and saved model
checkpoints. No interactive steps, no network calls, no model training
(run from saved checkpoints and cached scores). At acceptance it is
executed in a fresh copy of your workspace with `submission/` absent, and
what it produces is compared to what you declared: row order and float
formatting do not matter; differing pairs, membership rows, or fused
values do. A submission its own rebuild cannot reproduce is refused, so
keep `work/rebuild.sh` current as your pipeline evolves — a script you
edited after your last full run is the usual way the two drift apart.
