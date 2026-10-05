# usecases_synthetic — Pipeline manifest

Ordered list of scripts that together reproduce the synthetic use cases. Doubles as an execution runbook — the reader should be able to see what runs when and what depends on what.

For the design rationale behind each stage, see [../knobs/](../knobs/).

---

## Phase 0 — Pool construction

### 1. `scripts/build_pool.py`

Builds a "pooled positives" set per domain, used downstream as a protection set during Knob 2 augmentation (never as replacement gold — see [../knobs/cross_cutting.md](../knobs/cross_cutting.md#gold-standard-incompleteness-and-pooling)).

**Inputs** (per source pair)
- the positives of the EM gold splits (`use cases/<domain>/base/input/entitymatching/`)
- the correspondences of the rule-based matcher of the domain's P1 notebook (the human-designed pipeline)
- the predictions of a per-domain Ditto checkpoint over the candidate pairs of a blocker sweep and over the P1 and gold pairs

**Outputs**
- `pools/<domain>/pooled_positives.csv` — columns `id1, id2, source_1, source_2, score, in_gold, in_human, in_ditto, decision_path`
- `pools/<domain>/pool_stats.json` — blocker sweep, candidate counts, bucket breakdown, transitive-closure and cluster-size telemetry

**Covered domains:** companies, games, music.
**Products** uses a separate pool builder ([scripts/build_pool_products.py](scripts/build_pool_products.py)) that derives the pool directly from each record's WDC `cluster_id` in the generator's copy of the sources (`usecases/products/input/data/products_*.json`) rather than from matcher outputs. The task data (`use cases/products/`) omit `cluster_id`, since it is the gold grouping. **Papers** uses [scripts/build_pool_papers.py](scripts/build_pool_papers.py), which matches records across sources on their DOI and adds the EM gold positives.

**Key decisions baked in**
- **Three evidence streams.** Gold positives are kept (`decision_path = gold`); pairs that both the P1 matcher and Ditto (score ≥ 0.5) declare are kept (`agreement`); a pair that only one of the two declares is decided by an LLM (`gpt-5.4`, temperature 0), and the pairs it confirms are kept (`plm_check_llm_yes`).
- **Ditto candidates come from a blocker sweep** per source pair over five blockers, chosen by pair recall on the gold positives (target 0.97, as in the EM blocking committee) and reduction ratio, under a cap of one million candidates; the P1 and gold pairs are scored as well.
- **Transitive closure across source pairs**, per evidence stream; the pairs that the closure adds are scored with Ditto.
- **No cluster-size filter.** `pool_stats.json` records the connected-component sizes as telemetry.
- Same-source pairs are dropped.
- Pair IDs are ordered lexicographically so directional duplicates dedupe.

**Run**
```bash
source .venv/bin/activate
python usecases_synthetic/scripts/build_pool.py --all          # all three domains
python usecases_synthetic/scripts/build_pool.py --domain games # one domain
```

| Domain | Pool size | Gold | P1 and Ditto agree | Confirmed by the LLM |
|---|---|---|---|---|
| companies | 1910 | 1212 | 531 | 167 |
| games | 19938 | 610 | 9453 | 9875 |
| music | 8717 | 5513 | 2161 | 1043 |

---

## Phase 0.5 — Domain downsampling (optional)

### `scripts/downsample_domain.py`

Produces a reduced clone of an existing domain under a new name (e.g. `companies` → `companies-small`) so that expensive multi-knob ablation matrices (Phase 3) can be exercised in minutes instead of hours. The new domain reuses the source domain's per-knob YAMLs via `knob_config_alias` in the target domain YAML, so no knob configs are duplicated. Per-knob values can still be tuned on the small domain without forking the alias via a `knob_config_overrides: {knob_NN_<name>: {key: value, ...}}` block in the target domain YAML — values deep-merge onto the aliased knob config. See [lib/domain_config.py](lib/domain_config.py) `load_knob_config` and `_resolve_knob_config_overrides`.

**Protection policy**
- Every ID referenced by any EM gold CSV (all splits, both columns) is preserved.
- Every ID referenced by the fusion gold XMLs is preserved — both the top-level `<id>` text and each `provenance=`-split source ID.
- Additional non-gold rows are sampled deterministically up to `gold_multiplier * |protected|` per source (subject to `--min-rows` floor and optional `--max-rows` cap).

**Inputs / outputs**
- Reads: `use cases/<source_domain>/base/input/{data,entitymatching,fusion,schemamatching}/` (products: `usecases_synthetic/usecases/products/input/`)
- Writes: `use cases/<target_domain>/base/input/...` (products: `usecases_synthetic/usecases/<target_domain>/input/...`; downsampled data files, verbatim gold), `usecases_synthetic/config/domains/<target_domain>.yaml` with `knob_config_alias: <source_domain>`, and the filtered pool `usecases_synthetic/pools/<target_domain>/pooled_positives.csv`

**Run**
```bash
source .venv/bin/activate
python usecases_synthetic/scripts/downsample_domain.py --source-domain companies --target-domain companies-small --gold-multiplier 1.5 --seed 0
```

Domain configurations for `companies-small`, `games-small`, `music-small` and `products-small` are in `config/domains/`, and their pools in `pools/<domain>/`.

---

## Phase 1 — Baseline measurement

**Goal:** establish per-domain / per-stage committee baselines on the *original* data, before any augmentation. This is the reference point the committee-validated augmentation loop subtracts from to detect "difficulty deltas" (see [../knobs/cross_cutting.md](../knobs/cross_cutting.md#committee-validated-augmentation), §Bootstrap order).

**Committees:** SM (7 members), normalization (3 members), EM blocking (6 members), EM matching (4 members + pool diagnostic), fusion (9 members). See [config/committees/](config/committees/).

**Scripts**
- `scripts/measure_baseline.py` — runs the SM, normalization, EM and fusion committees on each domain's original data, writes `baselines/<domain>/baseline_metrics.json` + `baseline_report.md`.

**Covered domains:** companies, games, music, papers and products (baselines at [baselines/](baselines/)).

---

## Phase 2 — Scenario 1 (augmented use cases)

**Goal:** apply the knobs to the original use cases in the canonical order defined in [../knobs/README.md](../knobs/README.md#canonical-knob-application-order) to produce `easy`/`medium`/`hard` variants.

**Canonical order (S1):**
`Knob 2 (niche density) → Knob 4 (coverage skew) → Knobs 1/5/6 (value perturbations, jointly per cell) → Knob 3 (attribute drop) → Knob 10 (reliability reshuffle) → Knob 8 (header rename)`

**Per-knob scripts.** Each exposes a pure `apply_knob_XX(...)` entry point plus a CLI for standalone runs; the master orchestrator (`generate_variant.py`) calls the pure entry points in canonical order.

- `scripts/apply_knob_02_niche.py` — consumes pooled positives as protection floor; removes or interpolates entities toward the target `corner_case_ratio`. First knob applied.
- `scripts/apply_knob_04_coverage.py` — per-entity source coverage skew, takes Knob 2's placements as fixed.
- `scripts/apply_values_joint.py` — Knobs 1, 5, 6 applied jointly per cell with collision-index coordination to preserve conflict-preservation constraints.
- `scripts/apply_knob_03_drop.py` — per-source attribute drop, runs after value perturbations so drops happen on perturbed data. Cross-level nesting (`D_easy ⊆ D_medium ⊆ D_hard`) is enforced structurally: all three drop masks are computed in one pass, constraints (fusion floor, conflict preserve, single-source cap) are applied at each level, then `easy`/`medium` are shrunk against `hard` via `_enforce_nesting`. Propagate-fill runs after nesting so filled cells are never re-dropped.
- `scripts/apply_knob_10_reliability.py` — source reliability reshuffle via fusion-gold realignment, no raw gold change.
- `scripts/apply_knob_08_naming.py` — header-only schema naming divergence, orthogonal, runs last.

**Orchestrator + packaging**
- `scripts/generate_variant.py` — master CLI. Runs all eight knobs in canonical S1 order for a `(domain, level)`, consolidates provenance, writes `difficulty.yaml`, and (when run with `--level all`) produces a cross-level monotonicity audit covering K3 drop nesting plus per-knob intensity checks for K2/K4/K5/K6/K8/K10. K4 audit skips levels whose `target_coverage_histogram` is `null` (identity by design, not a failure). K8 audit uses summed `rapidfuzz.distance.Levenshtein` over provenance rows (column: `knob_08_naming_edit_distance`) rather than row counts, which mislead because a mild rung can rename more columns, with smaller edits, than a harsher one. K1/K2 LLM calls are served from the content-hash caches under `cache/` (`knob_01_paraphrases/`, `knob_02_interpolations/`, `knob_02_non_corner/`, `llm_cache/`) when they are present. The orchestrator does **not** enable strict-cache: `generate_variant()` defaults both `strict_cache_k1`/`strict_cache_k2` to `False` (never fail on a miss) and exposes no strict-cache CLI flag. On a cache miss it degrades gracefully — deterministic operators / blender when no `OPENAI_API_KEY` is set, or an OpenAI call that fills the cache when a key is present. (Only the standalone `apply_values_joint.py` CLI forces strict-cache at `hard` level — `apply_values_joint.py:551` — which would hard-error on a miss; the orchestrator never takes that path.)
- `scripts/package_variant.py` — assembles the per-level variant directory under `use cases/<domain>/<level>/` with `input/{data,schemamatching,entitymatching,fusion}`, `output/provenance/` (per-knob provenance and `provenance_all.csv`), `output/baselines/`, and `config/difficulty.yaml`.

**Run**
```bash
source .venv/bin/activate
python usecases_synthetic/scripts/generate_variant.py --domain companies --level easy
python usecases_synthetic/scripts/generate_variant.py --domain companies --level all   # runs easy+medium+hard and writes monotonicity_audit.csv
```

**Scope:** companies, games, music, papers and products.

---

## Phase 3 — Committee validation


**Goal:** re-run committees on the augmented variants and compare to Phase 1 baselines. Controlled monotone drops = real difficulty signal. Collapse = back off via the fix-strategy table in [../knobs/cross_cutting.md](../knobs/cross_cutting.md#per-knob-fix-strategy-defaults).

- `scripts/validate_variant.py` — runs the SM, normalization, EM and fusion committees against a packaged variant (one `(domain, level)` at a time), loads the baseline, and persists per-level metrics with baseline+delta twins on every leaf, plus a per-pair EM CSV, a per-attribute fusion CSV, and a `level_report.md` rollup. Refuses to run if committee YAML hashes diverge from `baseline_metrics.json`'s recorded versions (belt-and-braces drift guard). Uses the baseline's `fusion_input_member` as-is — no per-variant re-selection. Measurement-only: monotonicity and collapse judgement live in `analyze_monotonicity.py`.
- `scripts/analyze_monotonicity.py` — cross-level monotonicity + collapse detection. Consumes the per-level `metrics.json` files produced above plus the baseline and writes `validation/<domain>/monotonicity_report.md` + `.csv`.
- `scripts/run_ablation_validation.py` — per-knob ablation runner. For each active knob (K1/K2/K3/K4/K5/K6/K8/K10), generates a single-knob-hard variant and runs `validate_variant` against it. Writes `validation/<domain>/ablation/knob_<id>/metrics.json`. Expensive: 8 knobs × ~20-40 min each.
- `scripts/analyze_ablation.py` — per-knob ablation analyzer. Consumes baseline, full-hard, and per-knob ablation metrics; writes `validation/<domain>/ablation/ablation_report.md` + `.csv` with per-signal deltas and four interaction flags (cross-stage leakage, primary under/over-signal, direction mismatch).

**Run**
```bash
source .venv/bin/activate
python usecases_synthetic/scripts/validate_variant.py --domain companies --level easy
python usecases_synthetic/scripts/validate_variant.py --domain companies --level medium
python usecases_synthetic/scripts/validate_variant.py --domain companies --level hard
python usecases_synthetic/scripts/validate_variant.py --domain companies --level baseline  # sanity: all deltas should be 0
```

**Outputs**
- `validation/<domain>/<level>/metrics.json` — full per-stage / per-member / per-pair / per-attribute metrics with `_baseline` + `_delta` twins
- `validation/<domain>/<level>/level_report.md` — human-readable stage tables with delta columns
- `validation/<domain>/<level>/em_per_pair.csv` — one row per (member, pair)
- `validation/<domain>/<level>/fusion_per_attribute.csv` — one row per fusion attribute

---

## Notes on convention

- All scripts import PyDI the same way tests do — absolute imports from `PyDI`.
- All scripts should be runnable standalone from the repo root (`python usecases_synthetic/scripts/<name>.py`) — no `cd` required, use `Path(__file__).resolve().parents[N]` to locate the repo root.
- All scripts must preserve `DataFrame.attrs` provenance when transforming frames.
- No emojis in console output.
- NumPy-style docstrings.
