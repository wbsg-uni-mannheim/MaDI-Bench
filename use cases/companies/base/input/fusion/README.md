# Fusion Ground Truth

This directory holds the hand-annotated data-fusion validation and test records for this task.

For a first look, prefer the `*_better_readability.csv` files. They keep only the target-schema columns (from `../schemamatching/target_schema.json`), in schema order, and drop the provenance and source-helper fields from the original splits.

The original fusion files remain the authoritative artifacts. They may carry source attributes, provenance annotations, raw left/right values, URLs, and other fields useful for tracing values and running the evaluation. Where a target-schema field is missing from an original split, the Better Readability CSV leaves that column empty rather than inferring a value.
