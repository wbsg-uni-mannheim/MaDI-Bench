---
name: fusion
kind: knowledge
description: Resolve attributes from cluster-member evidence with deterministic, schema-aware rules and provenance.
---

# Data fusion without a validation teacher

Fuse the final membership, not an earlier clustering. Use the canonical tables
created by normalization. For each attribute choose a resolver based on its
semantics: normalized vote for equivalent categorical values, source-supported
representatives for text, carefully justified numeric aggregation, or set union
only when the attribute genuinely represents multiple independent values.

Do not average identifiers, mutually exclusive model specifications or numbers
with unresolved units. Longer text is not automatically more correct. Majority
agreement is weaker if sources copied one another. Without independent evidence
do not invent source accuracy weights. State deterministic tie-breaks and the
limitations of whichever heuristic you use.

Ignore empty strings in voting, but preserve meaningful false/zero values.
Canonicalize aliases before counting votes. Honor canonical taxonomy spelling
and distinguish exhaustive lists from partial vocabularies. For unresolved values,
prefer honest missingness over a fabricated answer. Never enrich from another
model, an external website or a cached reference set in this profile.

Keep per-cell provenance or an auditable rule explaining the chosen value. Re-read
the emitted table and verify schema types, target columns, list representation,
taxonomy membership and _id/membership alignment. Sample-check multi-source groups
to ensure that values belong to their own members; id-set agreement alone cannot
detect rows accidentally permuted during a join.

Report disagreement rate, missingness and schema validity as diagnostics, not
fusion accuracy. No fusion-evaluation service or correctness dataset is available.
