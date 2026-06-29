# Fusion Ground Truth

This directory holds the hand-annotated data-fusion validation and test records for this task.

For a first look, prefer the `*_better_readability.csv` files. They keep the `source_ids` helper column followed by the target-schema value columns (from `../schemamatching/target_schema.json`) in schema order, and drop provenance and other source-helper fields from the original splits.

In the papers task, `source_ids` lists the source records that describe the fused paper, for example `dblp-...`, `crossref-...`, and `open_alex-...`. DOI values were used only to derive the entity-matching pairs and fusion splits; they are not part of the released source data, fusion gold, or target schema.

The `cited_by_count` values in the papers fusion gold are interpreted as citation counts as of `2026-05-30` (May 30, 2026).

The original fusion files remain the authoritative artifacts. They may carry source attributes, provenance annotations, raw left/right values, URLs, and other fields useful for tracing values and running the evaluation. Where a target-schema field is missing from an original split, the Better Readability CSV leaves that column empty rather than inferring a value.
