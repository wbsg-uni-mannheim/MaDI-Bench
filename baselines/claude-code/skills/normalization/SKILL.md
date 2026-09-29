---
name: normalization
kind: knowledge
description: Canonicalize values symmetrically using schema taxonomies, explicit units and measured coverage.
---

# Value normalization

Start from translated columns and classify attributes as controlled categories,
scalars with units, identifiers, dates, lists or free text. Apply the same canonical
representation to every source; keep original values and native ids for audit.

Read every taxonomy reference declared by the target schema, including paths,
canonical columns, exhaustiveness flags and aliases.
Create work/taxonomy_plan.json documenting paths, canonical columns, aliases,
list serialization and unmapped-value policy. An exhaustive vocabulary is a
closed set; a non-exhaustive vocabulary is not proof that other values are wrong.

Measure source, attribute, non_null, canonical, unmapped and canonical_rate in
work/taxonomy_coverage.csv. Inspect common unmapped values and correct justified
aliases. Do not force wrong categories merely to increase coverage. Re-read the
written canonical tables to prove that the mappings reached downstream artifacts.

Normalize explicit unit differences and locale notation. Avoid interpreting
ambiguous decimals, dates or bare numbers without context. Measure parse failures
and keep non-destructive fallbacks. Preserve model suffixes, capacity, generation,
form factor and other details that distinguish entities. Case-fold comparison
copies only; fused output must retain canonical or source-supported spelling.

Recompute block keys, matching and fusion after changes. Inspect symmetry across
sources, unexpected null increases, collapsed distinctions and stale file paths.
Use parse rates, taxonomy coverage and schema checks, not unmeasurable accuracy.
