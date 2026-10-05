# Cross-cutting policies

Policies that apply to every knob. Per-knob cards reference this file rather than restating these.

## Profile model — absolute target bands

Each domain ships four artifacts: `baseline` (reference, untouched) plus `easy` / `medium` / `hard` augmented variants. Each variant is generated to hit an *absolute* target difficulty band for the knob, which may require *adding* heterogeneity (baseline below target) or *reducing* it (baseline above target — e.g. normalize-down for easy). Comparability across domains is a **soft goal**: same intent per level and monotone easy→medium→hard shape required; absolute committee numbers need not align across domains.

## Per-value provenance (mandatory)

For every augmentation, emit a row-level provenance record with at minimum:

```
(entity_id, source, attribute, original_value, new_value, transform_fn, transform_params, knob, level)
```

Knobs that operate on entities (Knob 2) or columns (Knobs 8, 9) emit entity-scoped or column-scoped records instead. Per-knob cards specify the exact `transform_fn` values they use.

## Test-set treatment

- **Fusion test set:** entity membership and gold values frozen. Only mutate when unavoidable, and only via the committee-driven gold replace/extend rules below.
- **EM splits:** regenerated per variant as per-source-pair per-split files in two versions: `<src1>_2_<src2>_{train,val,test}_baseline_pruned.csv` (the base pairs whose records survive in the variant) and `<src1>_2_<src2>_{train,val,test}_corner_filled.csv` (the surviving pairs refilled toward the base split's size and positive ratio with corner-mined pairs; the backfill serves the splits in the order test → val → train). See [Protection set semantics](#protection-set-semantics-not-replacement-gold) for the reported and diagnostic EM surfaces.
- **SM mappings:** unchanged in S1 unless Knob 8 is actively perturbing headers. Regenerated per variant in S2 (Knobs 8 and 9).

## Committee-validated augmentation

Knobs that touch values (1, 5, 6, 7, 10) risk pushing the dataset outside the regime where the gold is recoverable. Resolution: wrap augmentation in a measurement loop using **committees of standard PyDI methods**, separately at the fusion, SM, and EM stages.

1. **Establish baseline.** Run the committee on the *original* dataset under lenient evaluation. Record per-attribute / per-stage baseline accuracy / F1.
2. **Apply augmentation** with the chosen knob settings.
3. **Re-measure.** Run the same committee on the augmented dataset.
4. **Compare.** Expected outcome: a controlled monotone *drop* (the difficulty signal). Unexpected: collapse to ~random.
5. **Fix on collapse**, in preference order:
   - **Soften source augmentation locally** for the offending entities/attributes (preserves the original gold).
   - **Update the gold** (replace with closest augmented variant, or extend to an accepted set) only if softening can't preserve the difficulty signal.

The committee mechanism is the **safety net** (prevents impossible tasks), the **calibration tool** (turns qualitative scales into measured deltas), and the **validation evidence** for the variants.

### Per-knob fix-strategy defaults

| Knob | Fix on collapse |
|---|---|
| 1 (paraphrase) | Replace-or-extend gold to accepted set |
| 5 (formats) | Trivial — canonical-form comparison absorbs format/unit differences |
| 6 (noise) | **Reject** the augmentation; gold artifact never touched (typos must not be promoted into the gold contract) |
| 7 (ambiguity) | Extend gold to accepted sets where homonyms/collisions occur (not applied: the released variants do not use Knob 7) |
| 10 (reliability) | No gold change needed; gold is reshuffled across sources, not perturbed |

### Committee composition (fusion)

Nine members, each an end-to-end fusion approach that produces a complete fused table:

- **`pydi_per_attribute_optimal`:** per attribute, the validation-best PyDI resolver from a per-type candidate list (e.g. `voting`, `longest_string`, `most_complete` for strings; `median`, `trimmed_mean` for numerics; `union`, `intersection` for lists; `earliest`, `most_recent` for dates).
- **`llm_only`:** an LLM judge on every attribute.
- **Truth discovery:** `fusionquery_only`, `truthfinder_only`, `ltm_only`, `casefusion_only`, `accusim_only`.
- **Single-resolver baselines:** `voting_only`, `prefer_higher_trust_only`.

SM and EM committee compositions are configured in `usecases_synthetic/config/committees/`.

## Bootstrap order (committees → baseline → calibration)

The committee designs must exist before the per-source baseline measurement (which Knob 10 in particular consumes). **Committee design → baseline measurement → per-knob calibration.** Per-domain baselines used by Knobs 3, 6, 8, 10 are measured after the committees are designed.

## Gold standard incompleteness and pooling

The per-domain EM gold is a **sampled subset** of the true match set, not a complete enumeration. Records outside the labeled pairs are not confirmed non-matches — they are *unlabeled*. This matters for every knob that samples from the "non-matched" population (Knob 2 entity selection; Knob 2 + Knob 1 hard-negative mining) and for the committee loop above, which scores committee output against a test set that undercounts real positives.

### Pooled positives set

For each domain we build a **pooled positives set**: the labeled EM positives plus likely matches found over the full source data. The pool is serialized as `usecases_synthetic/pools/<domain>/pooled_positives.csv`.

**Companies, games, music** (`usecases_synthetic/scripts/build_pool.py`): three evidence streams per source pair — the positives of the EM gold splits (train / val / test), the correspondences of the rule-based matcher of the P1 notebook (the human-designed pipeline), and the predictions of a per-domain Ditto PLM (score ≥ 0.5) over the candidate pairs of a blocker chosen per source pair from a sweep of five blockers (by pair recall on the gold positives, target 0.97 as in the EM blocking committee, and reduction ratio) and over the P1 and gold pairs. The P1 and Ditto streams are each closed transitively across the source pairs. Gold positives are kept (`decision_path = gold`), pairs that P1 and Ditto both declare are kept (`agreement`), and a pair that only one of the two declares is decided by an LLM (`gpt-5.4`, temperature 0; kept pairs: `plm_check_llm_yes`). Columns: `id1`, `id2`, `source_1`, `source_2`, `score`, `in_gold`, `in_human`, `in_ditto`, `decision_path`.

**Products** (`usecases_synthetic/scripts/build_pool_products.py`): the WDC `cluster_id` grouping in the generator's copy of the products sources; `pool_agreement` counts the sources of a cluster.

**Papers** (`usecases_synthetic/scripts/build_pool_papers.py`): exact DOI matches across the sources, unioned with the EM gold positives.

**Egregious-cluster filter (products and papers).** These two builders drop connected components whose size exceeds `max(ceil(P99 of observed component sizes), 3 * n_sources)`. The P99 adapts to each domain's distribution; the `3 * n_sources` structural floor reflects the upper bound on plausible cross-source duplicate clusters (n_sources members plus slack for in-source duplicates). `build_pool.py` applies no size filter and reports the component-size distribution in `pool_stats.json`.

```
expanded_positives = test_gold ∪ train_gold ∪ val_gold ∪ pooled_positives
```

Pool construction is a prerequisite for the committee loop and for Knob 2 calibration.

### Protection set semantics (not replacement gold)

The pool is used as a **"probably-positive, do not perturb" protection set**, never as a replacement gold. **Pool-as-replacement-gold remains forbidden** — using the pool's declared positives as the scoring oracle would rubber-stamp the pooling systems (every pool member would score ≈1.0 against its own declaration on its own inductive bias).

Concretely, the pool constrains the *generator*, not the evaluator:

1. **Knob 2 entity selection:** entities appearing in any `expanded_positives` pair are protected — never dropped (easy removal), never used as parent seeds for single-source distractor interpolation (hard level). This extends the fusion-gold floor into a pooled-positives floor.
2. **Hard-negative mining (Knobs 1, 2):** synthetic hard negatives are drawn only from pairs *outside* `expanded_positives`, with an additional score-based safety margin around the pool systems' similarity thresholds (pair must sit below every pool system's decision boundary by at least margin δ, δ set per domain as `plm_margin_delta` in the Knob 2 configuration).
3. **Committee loop diagnostic signal:** compute a "committee-vs-pool agreement" rate as a diagnostic. If primary F1 collapses but pool agreement stays high, the collapse is probably hidden-positive noise rather than real difficulty — soften the augmentation. If both drop together, the difficulty signal is real. Pool agreement is never a reported number.
4. **Knob 6 (noise) is unchanged** — typos never get promoted into any gold contract, pooled or otherwise.

#### Reportable EM F1 — primary vs. secondary

Under S1 the committee validation scores EM predictions closed-set (`score_em_correspondences_closed_set`): predictions outside a split's judged pairs (positives and explicit negatives) are **out of scope** and do not count as FPs. This closed-set scoping prevents precision collapse when the matcher scans a larger pair space than the benchmark covers.

- **Primary — `<src1>_2_<src2>_test_corner_filled.csv`.** The surviving base test pairs plus the corner-mined backfill: cross-source pairs of K2-interpolated near-twins (positives) and cross-cluster pairs gated by the score-margin hard-negative policy (negatives). The backfill labels are **known by construction** — the rubber-stamp argument that bars pool-as-gold does not apply, because the pairs are not drawn from any matcher's own decision boundary. This surface carries the EM committee's macro_f1 monotonicity check.
- **Secondary — `<src1>_2_<src2>_test_baseline_pruned.csv`.** The surviving base test pairs only; a per-level reference value.
- **Diagnostic — pool agreement.** `pool_precision` / `pool_recall` inform the collapse-vs-hidden-positive check above.

The `val` `corner_filled` split serves as an internal val/test agreement check; where a variant has no val split (Games), a stratified hold-out of the training split takes its place.

Committee composition is frozen across variants (same blockers, matchers, and thresholds), so cross-variant deltas on the regenerated F1 isolate the knob's difficulty signal.

**Contamination mitigation.** The knob intensities were tuned against committee F1 on the regenerated validation splits, and the hard-negative gate uses a Ditto PLM to filter negatives, so Ditto's F1 on those splits is a ceiling estimate. For downstream users the evaluation sets are the base task's `*_test.csv` and each variant's `*_test_corner_filled.csv` (top-level README); the plain-named copies of the base splits in the variant folders are not evaluation sets.

**Aggregate vs. best-member reporting.** Committee macro_f1 is the unweighted mean across all enabled members of a stage. That number can mask one strong individual or one disabled-by-failure member dragging the mean down. When reporting per-stage signals, **always also report the best-member F1 alongside the committee macro_f1**. The best-member ceiling is what an end user could obtain by picking the strongest single matcher; the committee mean is what the difficulty validation guarantees against averaging. Both numbers are needed to interpret a delta: a -0.10 macro drop with a -0.20 best-member drop is real; a -0.10 macro drop with a flat best-member is one weak member regressing.

### Residual limitations

- Pooling gives a **recall lower bound**, not completeness: matches that the pool builders miss remain hidden.
- The cost of the protection set is **reduced augmentation headroom** — the generator has fewer entities/pairs to work with. For domains with small labeled gold and large record counts (companies, games), this may materially shrink Knob 2's downward range; per-domain headroom estimates must be re-measured against the pooled set, not the raw EM gold.
