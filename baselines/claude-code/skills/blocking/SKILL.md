---
name: blocking
kind: knowledge
description: Construct complementary candidate sets and inspect label-free coverage and computational cost.
---

# Blocking

Blocking limits comparisons; matching decides identity. Inspect source sizes and
compute the full cross-source comparison count before choosing the candidate
budget. Preserve source-native ids and explicitly identify each source pair.

Combine complementary approaches appropriate to the data: reliable shared keys,
normalized token blocks, type/brand/specification keys, sorted neighborhoods or
nearest neighbors over general representations. Union and deduplicate their
candidates. Avoid blocking on a field that is mostly missing or incompatible
across sources; missing keys must not create a giant all-null block.

Measure candidate counts by source pair, per-record candidate-count distributions,
zero-candidate records, largest blocks, duplicate pairs and reduction ratio.
Inspect whether one source or entity category is almost completely excluded.
Add a conservative fallback for sparse records where justified. Compare alternate
blocking methods' overlap and unique contributions to see what they miss.

You have no known matching pairs, so candidate coverage is NOT pair completeness
or recall. Similar-looking excluded examples are review cases, not a labeled
evaluation set. Never claim PC from your own matching predictions.

Export the complete candidate set, then confirm every correspondence belongs to
it. After normalization or key changes regenerate candidates and all downstream
artifacts. Keep blocking simple enough to rerun within the task's CPU budget.
