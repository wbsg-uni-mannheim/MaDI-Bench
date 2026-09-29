---
name: refinement-clustering
kind: knowledge
description: Form coherent entity clusters while preserving provenance and diagnosing transitive over-merges.
---

# Refinement and clustering

Pairwise decisions induce a graph, but connected components can amplify one weak
bridge into a large false merge. Inspect large components, incompatible model
specifications and weak edges connecting otherwise distinct subgroups.

Use constraints supported by the task, not by a desired number of entities.
One-record-per-source may be appropriate if sources contain one row per entity;
check this assumption rather than making it universal. When resolving alternatives,
preserve evidence and use deterministic tie-breaks. Do not randomly discard records
to satisfy a constraint. Retain unmatched records as singletons.

Track cluster-size distribution, sources per cluster, within-source multiplicity,
singleton share and conflicting identity attributes. Review bridges and alternate
assignments when a small threshold change causes a giant structural change.
Explicitly distinguish a normalization disagreement from genuine entity conflict.

Export membership with resolvable source-native record identifiers. Every source
record should be represented as required by the submission contract; each record
must have one unambiguous assignment. Regenerate correspondences consistently
with the final cluster decisions and preserve the full candidate set.

After a cluster revision rerun fusion and check each fused row against its actual
members. Structural coherence is useful evidence, not proof of precision or recall.
