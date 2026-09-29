---
name: pipeline-diagnosis
kind: knowledge
description: Diagnose integration failures from source coverage, graph structure, schema validity and provenance.
---

# Label-free pipeline diagnosis

After the first complete pipeline and every downstream rerun, compute a consistent
diagnostic panel from saved artifacts. Treat diagnostics as symptoms, not ground
truth. There are no independently judged matches or fusion values in this profile.

Check native-id resolution, source-record coverage, correspondences contained in
candidates, cluster sizes, sources per cluster, singleton share and giant components.
Compare membership cluster ids with fused _id values in both directions. Quantify
schema/type/taxonomy validity and density per attribute, treating whitespace-only
strings as missing. Review parse failures and unmapped canonical values by source.

For every sampled multi-member cluster, trace its fused values back to its own
members. This catches row permutations and wrong joins that pass id-set checks.
Inspect high-similarity rejected pairs and contradictory accepted pairs as cases
for reasoning; do not silently promote them into trusted reference annotations.

Route symptoms to likely causes:

- Almost no cross-source groups: inspect candidate coverage and identifying fields.
- Giant components: inspect weak graph bridges, thresholds and variant distinctions.
- One source almost all singletons: inspect its schema mapping and block keys.
- Dense but invalid values: inspect normalization and taxonomy aliases.
- Low density: inspect mapping, null-handling and overly destructive resolvers.
- Fused values absent from members: inspect row alignment and provenance.

Change one responsible stage, rerun everything downstream, and recompute the same
panel. Preserve the prior revision. Do not optimize for density or fusion ratio
alone: over-merging and invented values inflate both. Report limitations instead
of claiming matching F1, blocking PC or fusion accuracy that you cannot measure.
