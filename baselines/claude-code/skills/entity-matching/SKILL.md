---
name: entity-matching
kind: knowledge
description: Make direct identity decisions with rules and similarities; no supervised training or reference labels.
---

# Entity matching without supervised training

Read the task's entity definition. Separate identifiers and discriminative
specifications from generic description text. Two records describing neighboring
variants are not identical merely because their titles are highly similar.

Compute interpretable evidence: normalized exact identifiers, token similarities,
edit distances, numeric differences with domain-appropriate tolerances, category
compatibility and general embedding similarity where useful. Missing values are
unknown, not matches. Avoid counting several derived versions of one field as
independent evidence. Descriptive embeddings alone can confuse nearby variants.

Use deterministic rules or an explicitly reasoned score to make direct decisions.
Strong contradictory model, capacity, version or other identity-defining values
can outweigh broad title agreement, but first rule out normalization errors.
Record evidence and decision rationale for uncertain cases. Compare a few
plausible threshold settings through graph and source-coverage diagnostics,
not an invented F1 score. Keep uncertainty explicit.

This profile has no labeling function, cached labels or externally judged
validation. Do not create pseudo-label splits or fit any supervised model.
Ditto training and fine-tuning are excluded. You can reason about individual
pairs as part of producing correspondences, but those decisions are not an
independent reference dataset for evaluating or training another matcher.

Inspect score distributions, ambiguous alternatives, high-similarity rejected
pairs and low-evidence accepted pairs. Refine your rule only for a documented
failure pattern, not to guarantee a predetermined cluster count. Pass scored
decisions to clustering and assess downstream contradictions before accepting a
revision. Record which thresholds and rules produced the saved correspondences.
