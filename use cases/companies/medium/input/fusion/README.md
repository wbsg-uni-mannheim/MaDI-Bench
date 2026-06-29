# Fusion Ground Truth

This directory holds the hand-annotated data-fusion validation and test records for this task.

For a first look, prefer the `*_better_readability.csv` files. They keep only the target-schema columns (from `../schemamatching/target_schema.json`), in schema order, and drop the provenance and source-helper fields from the original splits.

The Companies fusion gold was curated on `2026-04-26`. For time-dependent company attributes such as `assets`, `revenue`, and `keypeople`, use `2016` as the task-level temporal target rather than current company profiles. The values were selected to reflect the historical source-era context; the source metadata was published/updated in 2016, with underlying coverage described as DBpedia `2012-01-01/2014-04-30`, Forbes `2013-01-01/2013-12-31`, and FullContact `2014-01-01/2014-06-01`.

The original fusion files remain the authoritative artifacts. They may carry source attributes, provenance annotations, raw left/right values, URLs, and other fields useful for tracing values and running the evaluation. Where a target-schema field is missing from an original split, the Better Readability CSV leaves that column empty rather than inferring a value.
