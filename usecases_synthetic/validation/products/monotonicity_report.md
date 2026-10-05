# Monotonicity + Collapse Report — products

Cross-level analysis of easy/medium/hard variants against the baseline. See `knobs/cross_cutting.md` for the protocol.

## Cumulative Cross-Level Slope (load-bearing verdict)

Per-stage committee metric across the **cumulative** variant levels (every knob on at each level). This is the cross-level verdict: committee scores should weakly decrease easy -> medium -> hard, and baseline (a reference value) should land no harder than medium. It makes no single-knob isolation assumption, so this -- not the per-knob signals below -- is the headline difficulty verdict.

| Stage | Metric | baseline | easy | medium | hard | easy->hard | Mono (e>=m>=h) | BasePos |
|---|---|---|---|---|---|---|---|---|
| sm | `aggregated.macro_f1` | 0.717 | 0.791 | 0.721 | 0.657 | -0.134 | [ok] | [ok] |
| norm | `aggregated.macro_f1` | 0.457 | 0.451 | 0.470 | 0.466 | 0.015 | [!!] | [!!] |
| em_blocking | `aggregated.macro_pair_recall` | 0.851 | NaN | NaN | NaN | NaN | [!!] | [ok] |
| em_matching | `aggregated.macro_f1_variant_model_on_regen_test` | 0.897 | 0.899 | 0.791 | 0.812 | -0.086 | [!!] | [ok] |
| fusion | `aggregated.overall_accuracy` | 0.741 | 0.740 | 0.698 | 0.562 | -0.178 | [ok] | [ok] |

## Per-Knob Expected Signals (indicative)

> These are **per-knob** expectations from `knob_expected_signals.yaml`, evaluated against the **cumulative** variants (every knob on at each level). They cannot isolate one knob, so a `flat` expectation for a stage that *another* knob also drives reads `[!!]` by construction (e.g. SM is not flat for knob_01 because K8 naming is also on). Treat this as indicative of combined effect; the load-bearing verdict is the Cumulative Cross-Level Slope above. For true per-knob isolation, run `generate_variant --only-knob <K>` ablations.

| Knob | Signals | OK direction | OK range | Notes |
|---|---|---|---|---|
| knob_01 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_01_surface_augmentation.md) |
| knob_02 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_02_niche_density.md) |
| knob_03 | 4 | 2/4 [!!] | 0/— | [card](knobs/knob_03_attribute_drop.md) |
| knob_04 | 4 | 1/4 [!!] | 0/— | [card](knobs/knob_04_coverage_skew.md) |
| knob_05 | 6 | 2/6 [!!] | 0/— | [card](knobs/knob_05_format_unit.md) |
| knob_06 | 5 | 2/5 [!!] | 0/— | [card](knobs/knob_06_value_noise.md) |
| knob_08 | 7 | 4/7 [!!] | 0/— | [card](knobs/knob_08_schema_naming.md) |
| knob_10 | 6 | 1/6 [!!] | 0/— | [card](knobs/knob_10_source_reliability.md) |

## Per-Signal Results

| Knob | Signal | Stage | Metric | Dir | baseline | easy | medium | hard | delta | target | Mono | Rng | BasePos | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knob_01 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.717 | 0.791 | 0.721 | 0.657 | -0.061 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.791 -> 0.721 -> 0.657 |
| knob_01 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.897 | 0.806 | 0.720 | 0.784 | -0.113 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.806 -> 0.720 -> 0.784 |
| knob_01 | em_blocker_spread | em_blocking | `spread:per_member.standard_blocker.metrics.pair_recall:per_member.embedding_blocker.metrics.pair_recall` | down | -0.340 | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |
| knob_01 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.897 | 0.926 | 0.889 | 0.858 | -0.038 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.926 -> 0.889 -> 0.858 |
| knob_02 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.897 | 0.806 | 0.720 | 0.784 | -0.113 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.806 -> 0.720 -> 0.784 |
| knob_02 | em_pool_precision_tightens | em_matching | `aggregated.macro_precision` | down | 0.891 | 0.884 | 0.782 | 0.837 | -0.054 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.884 -> 0.782 -> 0.837 |
| knob_02 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.741 | 0.740 | 0.698 | 0.562 | -0.179 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.740 -> 0.698 -> 0.562 |
| knob_02 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.897 | 0.926 | 0.889 | 0.858 | -0.038 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.926 -> 0.889 -> 0.858 |
| knob_03 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.851 | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |
| knob_03 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.897 | 0.806 | 0.720 | 0.784 | -0.113 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.806 -> 0.720 -> 0.784 |
| knob_03 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.741 | 0.740 | 0.698 | 0.562 | -0.179 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.740 -> 0.698 -> 0.562 |
| knob_03 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.897 | 0.926 | 0.889 | 0.858 | -0.038 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.926 -> 0.889 -> 0.858 |
| knob_04 | em_flat_or_shift | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.897 | 0.806 | 0.720 | 0.784 | -0.113 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.806 -> 0.720 -> 0.784 |
| knob_04 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.741 | 0.740 | 0.698 | 0.562 | -0.179 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.740 -> 0.698 -> 0.562 |
| knob_04 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.038 | 0.054 | 0.153 | 0.109 | 0.070 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.054 -> 0.153 -> 0.109 |
| knob_04 | em_flat_or_shift_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.897 | 0.926 | 0.889 | 0.858 | -0.038 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.926 -> 0.889 -> 0.858 |
| knob_05 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.717 | 0.791 | 0.721 | 0.657 | -0.061 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.791 -> 0.721 -> 0.657 |
| knob_05 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.897 | 0.806 | 0.720 | 0.784 | -0.113 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.806 -> 0.720 -> 0.784 |
| knob_05 | em_pool_recall_drops_for_lexical | em_blocking | `per_member.standard_blocker.metrics.pair_recall` | down | 0.660 | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |
| knob_05 | fusion_spread_is_signal | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.038 | 0.054 | 0.153 | 0.109 | 0.070 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.054 -> 0.153 -> 0.109 |
| knob_05 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.741 | 0.740 | 0.698 | 0.562 | -0.179 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.740 -> 0.698 -> 0.562 |
| knob_05 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.897 | 0.926 | 0.889 | 0.858 | -0.038 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.926 -> 0.889 -> 0.858 |
| knob_06 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.717 | 0.791 | 0.721 | 0.657 | -0.061 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.791 -> 0.721 -> 0.657 |
| knob_06 | em_monotone_drop | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | down | 0.897 | 0.806 | 0.720 | 0.784 | -0.113 | qualitative | [!!] | [ok] | [ok] | not weakly down: 0.806 -> 0.720 -> 0.784 |
| knob_06 | em_pool_recall_drops | em_blocking | `aggregated.macro_pair_recall` | down | 0.851 | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |
| knob_06 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.741 | 0.740 | 0.698 | 0.562 | -0.179 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.740 -> 0.698 -> 0.562 |
| knob_06 | em_monotone_drop_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | down | 0.897 | 0.926 | 0.889 | 0.858 | -0.038 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.926 -> 0.889 -> 0.858 |
| knob_08 | sm_monotone_drop | sm | `aggregated.macro_f1` | down | 0.717 | 0.791 | 0.721 | 0.657 | -0.061 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.791 -> 0.721 -> 0.657 |
| knob_08 | sm_label_collapses | sm | `per_member.label_jw.metrics.f1` | down | 0.595 | 0.814 | 0.629 | 0.317 | -0.277 | qualitative | [ok] | [ok] | [!!] | weakly down: 0.814 -> 0.629 -> 0.317 |
| knob_08 | sm_spread_is_signal | sm | `spread:per_member.llm_openai.metrics.f1:per_member.label_jw.metrics.f1` | up | 0.405 | 0.186 | 0.371 | 0.663 | 0.257 | qualitative | [ok] | [ok] | [!!] | weakly up: 0.186 -> 0.371 -> 0.663 |
| knob_08 | sm_instance_steady | sm | `per_member.instance_tf_cosine.metrics.f1` | flat | 0.528 | 0.537 | 0.537 | 0.582 | 0.054 | qualitative | [ok] | [ok] | [ok] | weakly flat: 0.537 -> 0.537 -> 0.582 |
| knob_08 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.897 | 0.806 | 0.720 | 0.784 | -0.113 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.806 -> 0.720 -> 0.784 |
| knob_08 | fusion_flat | fusion | `aggregated.overall_accuracy` | flat | 0.741 | 0.740 | 0.698 | 0.562 | -0.179 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.740 -> 0.698 -> 0.562 |
| knob_08 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.897 | 0.926 | 0.889 | 0.858 | -0.038 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.926 -> 0.889 -> 0.858 |
| knob_10 | sm_flat | sm | `aggregated.macro_f1` | flat | 0.717 | 0.791 | 0.721 | 0.657 | -0.061 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.791 -> 0.721 -> 0.657 |
| knob_10 | em_flat | em_matching | `aggregated.macro_f1_baseline_model_on_regen_test` | flat | 0.897 | 0.806 | 0.720 | 0.784 | -0.113 | qualitative | [!!] | [ok] | [!!] | not weakly flat: 0.806 -> 0.720 -> 0.784 |
| knob_10 | fusion_monotone_drop | fusion | `aggregated.overall_accuracy` | down | 0.741 | 0.740 | 0.698 | 0.562 | -0.179 | qualitative | [ok] | [ok] | [ok] | weakly down: 0.740 -> 0.698 -> 0.562 |
| knob_10 | fusion_spread_widens | fusion | `spread:aggregated.max_accuracy:aggregated.min_accuracy` | up | 0.038 | 0.054 | 0.153 | 0.109 | 0.070 | qualitative | [!!] | [ok] | [ok] | not weakly up: 0.054 -> 0.153 -> 0.109 |
| knob_10 | fusion_voting_drops_faster_than_trust_on_name | fusion | `spread:per_attribute.name.prefer_higher_trust_only:per_attribute.name.voting_only` | up | NaN | NaN | NaN | NaN | NaN | qualitative | [!!] | [ok] | [ok] | metric missing at levels: easy, medium, hard |
| knob_10 | em_flat_pruned | em_matching | `aggregated.macro_f1_baseline_model_on_baseline_test` | flat | 0.897 | 0.926 | 0.889 | 0.858 | -0.038 | qualitative | [!!] | [ok] | [ok] | not weakly flat: 0.926 -> 0.889 -> 0.858 |

## Collapses

No members fell below the collapse threshold.

## Best-Member Ceiling

Per-stage best-member F1 across baseline -> easy -> medium -> hard. A valid difficulty signal must depress the *ceiling* (the user-attainable member), not just the committee mean. A flat / rising ceiling means the user-selected matcher never sees the synthetic difficulty (committee-mean drift can be masked by weak-member degradation alone).

| Stage | baseline | easy | medium | hard | delta | non-increasing | winner trail | Reason |
|---|---|---|---|---|---|---|---|---|
| sm | 1.000 | 1.000 | 1.000 | 0.980 | -0.020 | [ok] | `llm_openai -> duplicate_majority -> llm_openai -> llm_openai` | best-member ceiling non-increasing: 1.000 -> 1.000 -> 1.000 -> 0.980  (llm_openai -> duplicate_majority -> llm_openai -> llm_openai) |
| norm | 0.852 | 0.866 | 0.907 | 0.913 | 0.061 | [!!] | `llm_only -> llm_only -> llm_only -> llm_only` | best-member ceiling did NOT decline: 0.852 -> 0.866 -> 0.907 -> 0.913  (llm_only -> llm_only -> llm_only -> llm_only) — difficulty dial may be invisible to the user-selected matcher |
| em_blocking | 1.000 | NaN | NaN | NaN | NaN | [!!] | `token_blocker -> ? -> ? -> ?` | best-member ceiling missing at levels: easy, medium, hard |
| em_matching | 0.955 | 0.952 | 0.907 | 0.924 | -0.031 | [!!] | `ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm` | best-member ceiling did NOT decline: 0.955 -> 0.952 -> 0.907 -> 0.924  (ditto_plm -> ditto_plm -> ditto_plm -> ditto_plm) — difficulty dial may be invisible to the user-selected matcher |
| fusion | 0.741 | 0.740 | 0.698 | 0.562 | -0.179 | [ok] | `pydi_per_attribute_optimal -> llm_only -> llm_only -> llm_only` | best-member ceiling non-increasing: 0.741 -> 0.740 -> 0.698 -> 0.562  (pydi_per_attribute_optimal -> llm_only -> llm_only -> llm_only) |

## Open Questions

Signals that are direction-correct but magnitude-unspecified by the knob card.

| Knob | Signal | Stage | observed delta | Card notes |
|---|---|---|---|---|
| knob_01 | em_monotone_drop_pruned | em_matching | -0.038 | Monotone F1 drop across levels. Sharper for lexical blockers than for embedding matchers. [baseline_pruned surface companion] |
| knob_02 | em_monotone_drop_pruned | em_matching | -0.038 | Monotone drop. Sharper for similarity-threshold matchers than for learned matchers. Niche collisions pressurise precision. [baseline_pruned surface companion] |
| knob_03 | fusion_monotone_drop | fusion | -0.179 | Fusion accuracy drops monotonically given conflict-preserving constraint and survivor cap. |
| knob_03 | em_monotone_drop_pruned | em_matching | -0.038 | Monotone F1 drop. Learned matchers with missing-value handling degrade less than rule-based comparators. [baseline_pruned surface companion] |
| knob_04 | fusion_monotone_drop | fusion | -0.179 | Primary target. Voting-family strategies degrade faster than trust-weighted / single-best-source strategies. |
| knob_05 | fusion_monotone_drop | fusion | -0.179 | Monotone drop for naive strategies. Best-strategy accuracy (the max across strategies) may hold if a canonicalizing strategy is available, but overall_accuracy averages all strategies and therefore drops. |
| knob_05 | em_monotone_drop_pruned | em_matching | -0.038 | Monotone drop for non-normalizing comparators; minimal for type-aware comparators. [baseline_pruned surface companion] |
| knob_06 | fusion_monotone_drop | fusion | -0.179 | Monotone drop given survivor floors. Provenance-aware fusers should be rewarded over cell-local. |
| knob_06 | em_monotone_drop_pruned | em_matching | -0.038 | Monotone drop. Sharp for rule-based comparators, mild for learned/embedding matchers. [baseline_pruned surface companion] |
| knob_08 | sm_monotone_drop | sm | -0.061 | Primary target. Monotone drop expected. |
| knob_08 | sm_label_collapses | sm | -0.277 | Label-based string-similarity matchers collapse fast on cryptic/anonymized names. |
| knob_08 | sm_spread_is_signal | sm | 0.257 | Per the card, "the spread between matcher types IS the K8 difficulty signal". Instance-based / embedding / LLM matchers hold while label-based collapse. Spread (llm_openai.f1 - label_jaccard.f1) widens monotonically. |
| knob_08 | sm_instance_steady | sm | 0.054 | Instance-based matchers degrade more gracefully than label-based. Note: baseline instance-based F1 for companies is already 0.0 (no overlap in sampled values). This signal may be untestable on companies because the baseline is already at the floor. |
| knob_10 | fusion_monotone_drop | fusion | -0.179 | Primary target. Cell-local strategies (voting, most_complete, per-attribute trust) lose accuracy as unreliability grows. |

## Provenance

- Domain: products
- Expectations: `usecases_synthetic/config/knob_expected_signals.yaml`
- Baseline: `usecases_synthetic/baselines/products/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/products/<level>/metrics.json`
