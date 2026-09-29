---
name: schema-matching
kind: knowledge
description: Align source attributes by semantics, types, units and examples without reference mappings.
---

# Schema matching

Read the target schema and the task's entity definition. For every source, list
attribute names, observed types, null rates and representative values. Map by
meaning rather than position alone; identical column order is evidence to check,
not proof. Preserve a source-native id field throughout the pipeline.

Distinguish model name, model number and product title; dates from years; price
from currency; storage capacity from memory capacity. Detect unit differences
before interpreting numeric agreement. Many-to-one mappings require an explicit
transformation, not silent overwriting. A field missing in a source is missing,
not a reason to shift the remaining columns or invent a value.

Write a mapping inventory with source field, target field, transformation and
reason. Validate that all mapping sources actually exist and all target names
belong to the schema. Review populated-but-unmapped columns and targets with no
contributors. Export sm_mapping.csv in the submission specification's format.

Run a small translation first and inspect rows from every source. If an upstream
mapping changes, invalidate normalized tables and everything downstream. Your
coverage and type checks are not schema-matching F1; there is no reference mapping
available to measure that score in this baseline.
