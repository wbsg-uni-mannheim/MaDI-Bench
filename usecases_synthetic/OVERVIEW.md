# Synthetic Use Case Pipeline — Overview

A tour of how `usecases_synthetic/` turns an original
use case (companies, games, music, papers, products) into three difficulty-graded
synthetic variants — `easy`, `medium`, `hard` — and validates each level
against a frozen committee of matchers / normalizers / fusion strategies.

For the commands of each phase see [PIPELINE.md](PIPELINE.md) (runbook). This file is meant to give a new
reader a self-contained mental model of the pipeline.

---

## End-to-end flow

```
 ┌──────────────────┐    ┌─────────────────────┐    ┌──────────────────────────┐
 │ Original use     │    │ Phase 0             │    │ Phase 1                  │
 │ case data        │───▶│ Pool construction   │───▶│ Baseline measurement     │
 │ (5 domains)      │    │ (assemble a set of  │    │ (5 committees on         │
 │                  │    │  known-match pairs) │    │  unperturbed data)       │
 └──────────────────┘    └─────────────────────┘    └────────────┬─────────────┘
                                                                 │
                                       ┌─────────────────────────┴────────────────────────┐
                                       ▼                         ▼                        ▼
                          ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
                          │ Phase 2 / easy       │  │ Phase 2 / medium     │  │ Phase 2 / hard       │
                          │ generate_variant.py  │  │ generate_variant.py  │  │ generate_variant.py  │
                          │ (knob stack below)   │  │ (knob stack below)   │  │ (knob stack below)   │
                          └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
                                     ▼                         ▼                         ▼
                          ┌──────────────────────────────────────────────────────────────────────────┐
                          │ Phase 3 — Validation per level                                           │
                          │   validate_variant.py  →  SM / Norm / EM-block / EM-match / Fusion       │
                          │   analyze_monotonicity.py  →  cross-level: macro_f1, best-member ceiling │
                          │                               + per-knob realised intensity audits       │
                          └────────────────────────────────────┬─────────────────────────────────────┘
                                                               ▼
                                                  validation/<domain>/monotonicity_report.md
```

---

## Phase 0 — Pool construction

### Why we need a pool

Every use case ships with an **entity-matching gold standard** — labeled
cross-source record pairs used to measure matcher performance. Those golds
are typically *incomplete*: a real-world
EM gold covers only a small fraction of the true match set (often single-
to low-double-digit recall against the actual match population), because
exhaustive cross-source labelling at scale is infeasible.

Incompleteness is fine when you're scoring matchers against the gold — you
just need a representative sample. But it becomes a problem when you start
*perturbing* the data to create harder variants (Phase 2 below). If the
perturbation step that thins out crowded entity neighbourhoods only knew
about gold pairs, it could happily delete a pair that's genuinely a match
in the real world but happens to be missing from the gold sample. The
resulting "harder" variant would then have *fewer matchable entities*, not
*harder-to-match entities* — the difficulty signal you'd measure later
would partially be noise from accidentally destroyed matches rather than
the dial setting you set.

The pool fills that gap. It augments the gold's protection coverage with
**likely matches found over the full source data**, even when the labeled
gold doesn't mention them.

### How the pool is built

For companies, games and music, [scripts/build_pool.py](scripts/build_pool.py)
combines three evidence streams per source pair:

- the positives of the **EM gold** splits (train / val / test);
- the correspondences of the **P1 matcher** — the rule-based matcher of
  the domain's P1 notebook (the human-designed pipeline);
- the predictions of a per-domain **Ditto** matcher (score ≥ 0.5) over the
  candidate pairs of a blocker chosen per source pair from a sweep of five
  blockers (by pair recall on the gold positives, target 0.97 as in the EM
  blocking committee, and reduction ratio) and over the P1 and gold pairs.

The P1 and Ditto streams are each closed transitively across the source
pairs; the pairs that the closure adds are scored with Ditto. Combination
rules:

- A gold positive lands in the pool (`decision_path = gold`).
- A pair that both P1 and Ditto declare lands in the pool
  (`decision_path = agreement`).
- A pair that only one of the two declares goes to an LLM (`gpt-5.4`,
  temperature 0); the pairs it confirms land in the pool
  (`decision_path = plm_check_llm_yes`).

`build_pool.py` applies no cluster-size filter; `pool_stats.json` records
the connected-component sizes as telemetry.

Products uses a separate pool builder,
[build_pool_products.py](scripts/build_pool_products.py), that derives
clusters directly from each record's WDC `cluster_id` in the generator's
copy of the sources (`usecases/products/input/data/products_*.json`), so no
matcher is needed. The task data (`use cases/products/`) omit `cluster_id`,
since it is the gold grouping. Papers uses
[build_pool_papers.py](scripts/build_pool_papers.py), which matches
records across sources on their DOI and adds the EM gold positives.

Both builders apply an **egregious-cluster filter** against
transitive-chain artefacts (A↔B, B↔C, C↔D edges that are each
individually plausible but together form a blob that clearly isn't one
entity): any cluster strictly larger than `max(ceil(P99), 3 × n_sources)`
is removed, where P99 is the 99th percentile of the domain's cluster sizes
and `3 × n_sources` is a structural floor (a legitimate cross-source
cluster could plausibly contain up to ~3 rows per source: one matched row
plus slack for in-source duplicates). In the products and papers pools
the filter removed no cluster.

### What the pool is used for

The pool is fed forward into Phase 2 as a **protected set** that the
perturbation steps consult before touching any entity. Concretely, the
step that controls how many similar-but-distinct entities cluster
together (K2 — explained below in the knob stack) refuses to remove any
entity that appears in the pool, even when its niche dial points at a
sparser target. The corresponding step for source coverage (K4) and the
cell-drop step (K3) apply analogous protection rules. The net effect is
that **the difficulty signal we measure in Phase 3 reflects the dials we
set, not noise from accidentally destroying known matches**.

### Outputs

Per domain under [pools/](pools/):
- `pooled_positives.csv` — one row per pair, columns
  `id1, id2, source_1, source_2, score, in_gold, in_human, in_ditto,
  decision_path` (products: `id1, id2, source_1, source_2, pool_agreement`,
  where `pool_agreement` counts the sources of the cluster).
- `pool_stats.json` — build statistics: per-source-pair counts (for
  companies, games and music also the blocker sweep and the bucket
  breakdown) and the cluster-size distribution.

| Domain    | Pool size | Gold  | P1 and Ditto agree | Confirmed by the LLM |
|-----------|----------:|------:|-------------------:|---------------------:|
| companies | 1910      | 1212  | 531                | 167                  |
| games     | 19938     | 610   | 9453               | 9875                 |
| music     | 8717      | 5513  | 2161               | 1043                 |

---

## Phase 1 — Baseline measurement

[scripts/measure_baseline.py](scripts/measure_baseline.py) runs the five
committees against each domain's *original* data and writes
`baselines/<domain>/baseline_metrics.json` + `baseline_report.md`. Committee
YAML SHAs are recorded in the metrics file; validate_variant refuses to run
later if those SHAs drift, so a frozen baseline is paired with a frozen
committee for the entire variant lifetime.

The committees (rosters from
[config/committees/](config/committees/), all members
`enabled_by_default: true` unless noted):

- **SM** — schema matching, 7 members spanning four signal families:
  `duplicate_majority` (duplicate), `label_jw` (label string similarity),
  `instance_tf_cosine` (instance tf-cosine), `embedding_sbert` (SBERT
  embedding), `llm_openai`, `magneto_slm_llm`, `coma_hybrid` (COMA-style
  hybrid).
- **Norm** — normalization, per-domain roster of 3 members (see
  `normalization_committee_<domain>.yaml`): `rule_per_attribute_optimal`
  (one rule per attribute among `text_clean`, `date_iso`, `number_locale`,
  `country_iso` and `taxonomy_lookup`), `llm_only` and `passthrough`.
- **EM-block** — entity-matching blocking, 6 members:
  `token_blocker`, `standard_blocker`, `embedding_blocker`,
  `sorted_neighbourhood_blocker`, `bm25_blocker`, `sc_block` (SC-Block,
  a supervised-contrastive encoder with nearest-neighbour retrieval;
  requires a per-domain trained checkpoint).
- **EM-match** — entity-matching matching, 4 members + a pool diagnostic:
  `ditto_plm` (DITTO PLM), `magellan` (Magellan-style comparator stack),
  `llm_matcher` (zero-shot LLM), `comem` (ComEM, a two-stage LLM matcher
  that selects candidates and then confirms each pair).
- **Fusion** — 9 members, each an end-to-end fusion approach that
  produces a complete fused table: `pydi_per_attribute_optimal` (per
  attribute, the validation-best PyDI resolver from a per-type candidate
  list such as `voting`, `longest_string`, `most_complete`, `median`,
  `trimmed_mean`, `union`), `llm_only` (an LLM judge, `gpt-5.4-mini`, on
  every attribute), five truth-discovery members (`fusionquery_only`,
  `truthfinder_only`, `ltm_only`, `casefusion_only`, `accusim_only`;
  where a method does not cover an attribute type, the validation-best
  PyDI resolver fills in), and two single-resolver baselines
  (`voting_only`, `prefer_higher_trust_only`).

See [config/committees/](config/committees/) for member parameters.

---

## Phase 2 — Variant generation (S1 augmented use cases)

[scripts/generate_variant.py](scripts/generate_variant.py) is the master
orchestrator. For a given `(domain, level)` it imports each `apply_knob_*`
pure entry point and applies them in canonical S1 order. Two knobs share an
LLM cache (K1, K2); on a cache miss the generator calls the LLM when
`OPENAI_API_KEY` is set and otherwise falls back to deterministic operators
(see PIPELINE.md).

### Knob stack

```
sources (post-Phase-0 protection set in scope)
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K2 niche density            (first — sets the entity population)            │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K4 coverage skew            (which sources have which entities)             │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K1 + K5 + K6  JOINT cell pass via apply_values_joint.py                     │
      │   K1 surface  →  K5 format  →  K6 noise                                     │
      │   Shared CollisionIndex: K5 skips K1+K4 cells; K6 skips K1+K5 but does     │
      │   touch K4-fabricated cells.                                                │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K3 per-source attribute drop  (column-level NaN masking per row)            │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K10 source reliability        (which source carries the gold-aligned cell)  │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  ┌─────────────────────────────────────────────────────────────────────────────┐
      │ K8 schema naming              (header renames only)                         │
      └─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
use cases/<domain>/<level>/
   ├── input/{data, schemamatching, entitymatching, fusion}
   ├── output/baselines/knob_NN_realised.csv     (per-knob intensity audit)
   ├── output/provenance/provenance_all.csv      (every modified cell)
   └── config/difficulty.yaml                    (resolved per-knob params)
```

### What each knob actually does

#### K2 — Entity niche density   [knob_02_niche_density.md](../knobs/knob_02_niche_density.md) · [apply_knob_02_niche.py](scripts/apply_knob_02_niche.py)

Controls how many similar-but-distinct entities cluster in the same semantic
neighborhood (franchises, discographies, sequels). The dial we actually set
is `corner_case_ratio` — the fraction of EM pairs that are near-twin
pairs — which acts as a measurable proxy for niche density: dense
neighborhoods produce more near-twin EM pairs, sparse neighborhoods
produce fewer. Two shared sub-systems on one multi-metric substrate
(lexical + embedding + attribute-overlap + label-collision): a
**consensus-biased RRF scorer** ranks entities by niche density (it picks
the seeds of interpolated near-twins and the low-density entities removed
to keep source sizes stable), a **recall-biased per-metric union** mines
the corner-case pairs that decide between dropping and interpolating and
that feed EM split regeneration and the hard-negative budget.

##### How "density" is computed

Each entity gets a single density score via four-metric Reciprocal Rank
Fusion (RRF) plus a label-collision boost. The metrics live in
[lib/niche_metrics.py](lib/niche_metrics.py); the fusion lives in
[lib/niche_scorer.py](lib/niche_scorer.py).

1. **Four ranking metrics**, each producing a top-K list of similar
   entities per entity (`metric_top_k = 30` per the music config):
   - `lexical_extended_jaccard` — generalised Jaccard over tokenised
     primary labels, with Levenshtein-ratio inner-token matching so
     K6-injected typos don't erase near-twins.
   - `tfidf` — sklearn `TfidfVectorizer` document-term matrix → cosine
     similarity.
   - `embedding` — sentence-transformers `BAAI/bge-base-en-v1.5` embeddings
     (cached on disk per domain) → cosine similarity.
   - `attribute_overlap` — weighted Jaccard over the categorical-
     attribute bag, with per-domain column weights.
2. **RRF fusion.** For every neighbour `n` that any metric ranked in
   entity `e`'s top-K, the fused score is
   `Σ_m  1 / (k₀ + rank_m(n | e))` with `k₀ = 60` (standard RRF
   damping). An *agreement count* tracks how many of the four metrics
   ranked `n` in `e`'s top-K.
3. **Consensus filter** (the "consensus-biased" bit). A neighbour with
   agreement count below `c_min = 2` contributes zero to density —
   one metric flagging `n` isn't enough; at least two have to agree.
4. **Label-collision boost.** Entities whose primary label normalises
   byte-identical (lowercase + accent-fold + strip punctuation + drop
   bracketed suffixes) to ≥1 other entity get a fixed `+5.0` boost.
   This deterministically pulls franchises with exact-match titles
   (`John Williams`, `Crash`) into the same density bucket even when
   the ranking metrics individually disagree.

The result is a per-entity density score: **high = tight cluster of
similar entities; low = uniquely positioned in the population**.

##### How the algorithm decides drop vs interpolate

The dispatcher first measures the *current* corner-case ratio in the
baseline data (using the same four metrics through a separate recall-
biased miner — `t_match` / `t_nonmatch` thresholds per metric, union
across metrics), then compares to the level's target with a ±0.02
tolerance band:

- **`baseline > target + 0.02`** → drop the entities that touch the most
  corner-case pairs (protected entities are skipped) and refill each drop
  with an LLM-synthesised non-corner entity (`non_corner_refill` in the K2
  configs) to keep the canonical set size stable; a refill whose label
  collides with a real entity is rejected. Removing dense entities
  instead would drift the ratio *further* from target (it shrinks the
  denominator faster than the numerator). With the refill disabled, K2
  no-ops and reports the baseline as the realised ratio.
- **`baseline < target - 0.02`** → **paired LLM interpolation**: for
  each step, generate a near-twin entity seeded from a dense cluster,
  then remove one low-density entity to keep the per-source row count
  close to the original (size-invariant by construction). Capped at
  `max_interp_fraction × n_entities` (`0.60` for music).
- **`|baseline − target| ≤ 0.02`** → no-op.

##### Per-level targets

The K2 configs set `target_corner_case_ratio` to 0.20 (easy), 0.50
(medium) and 0.80 (hard); music uses 0.20 / 0.30 / 0.35. The rule above
picks the operator from each variant's measured baseline ratio, and
`knob_02_realised.csv` records it. At easy the baseline ratio lies above
0.20 in games, music (~0.25), papers and products, so K2 drops and
refills; in companies (~0.20) K2 is a no-op. The LLM calls (interpolation
and refill, `gpt-5.4-mini`) are cached. Interpolation places half of the
near-twins across sources (hard positives) and half in a single source
(pure hard negatives): `placement_split` 0.5.

**Side-effect:** regenerates the EM test split per variant so the corner-case
ratio is tracked end-to-end. **Audit:** `knob_02_realised.csv` (baseline /
target / final ratio + per-guardrail rejection counters).

#### K4 — Per-entity source coverage skew   [knob_04_coverage_skew.md](../knobs/knob_04_coverage_skew.md) · [apply_knob_04_coverage.py](scripts/apply_knob_04_coverage.py)

Shifts the per-entity source coverage histogram (group size distribution: how
many sources cover each entity). Operates on whole rows, not cells.

- **Easy** (uniform): for entities missing from a source, copies the
  entity's row from a source that has it and paraphrases the copy with
  K1's medium operators (abbreviation table, EDA `random_swap` /
  `random_delete`; no LLM).
- **Medium**: identity — baseline preserved.
- **Hard** (long-tail): removes entity rows to create singletons, gated by a
  fusion-gold floor and a conflict-preserving removal rule.
  `within_source_duplicate_rate` also adds same-entity duplicate rows via K1's
  paraphrase pipeline at hard.

| Music dial — `target_coverage_histogram` (share of entities covered by N sources) | easy | medium | hard |
|---|---:|---:|---:|
| Covered by 1 source (singleton)   |  0% | (identity) | 55% |
| Covered by 2 sources              | 10% | (identity) | 30% |
| Covered by 3 sources (all)        | 90% | (identity) | 15% |
| `within_source_duplicate_rate`    |  0% |    0%      |  2% |

Easy aims for near-uniform coverage so voting strategies always see ≥2
values per attribute; hard creates a long tail so the majority of
entities are visible in only one source and fusion must fall back on
trust signals.

**Audit:** `knob_04_realized_vs_target.csv`.

#### K1 — Surface augmentation   [knob_01_surface_augmentation.md](../knobs/knob_01_surface_augmentation.md) · [apply_knob_01_surface.py](scripts/apply_knob_01_surface.py)

Plausible free-text rewriting of cell values (paraphrase, abbreviate,
reorder). Both forms are correct — the boundary against K6 is "legitimate
variant" vs "error." Tier-C hybrid generator:

- **Easy**: deterministic `normalize-to-canonical` via
  `baseline_above_target_rules` (used when natural baseline is already above
  the target paraphrase rate — pulls back toward a per-entity canonical form).
- **Medium**: deterministic table-driven abbreviation + EDA `random_swap` /
  `random_delete`.
- **Hard**: medium operators ∪ cached LLM paraphrase (`gpt-5.4-mini`) with
  contamination guardrails.

Per-attribute-class rates control what fraction of cells in each class
gets touched. A rate of 0.12 on `primary` means ~12% of primary-attribute
cells (e.g. song titles, album names) are paraphrased.

| Music dial — fraction of cells touched | easy | medium | hard |
|---|---:|---:|---:|
| `paraphrase_rate_primary` (e.g. `name`)              |  0% |  4% | 12% |
| `paraphrase_rate_key` (e.g. `artist`, `release-country`) |  0% |  8% | 18% |
| `paraphrase_rate_secondary` (e.g. `label`, `release-date`) |  0% | 16% | 30% |
| `paraphrase_rate_categorical` (e.g. `genre`)         |  0% |  8% | 18% |

The `paraphrase_long` rate from the K1 spec is not used: no domain
config routes an attribute to it.

#### K5 — Format / unit diversity   [knob_05_format_unit.md](../knobs/knob_05_format_unit.md) · [apply_knob_05_format.py](scripts/apply_knob_05_format.py)

Structured-format and unit rewriting — every value stays machine-parseable
and semantically exact, just in a different form. Classifies each attribute
by format family (date / number / currency / locale), draws a format
assignment, applies the operator, and verifies a round-trip parse.

- **Easy**: per-(source, attribute) draw from a pool of ≤2 formats; one
  format per source, consistent within a source.
- **Medium**: per-(source, attribute) draw from 2–3 formats; still consistent
  within a source.
- **Hard**: per-(row, attribute) draw from 3+ formats — a single source mixes
  formats row-to-row. Multiple coexisting units (USD/EUR; thousands/millions;
  minutes/hh:mm:ss). Excludes deliberately ambiguous values (those belong to
  K7).

| Music dial — `format_pools_per_level` | easy | medium | hard |
|---|---|---|---|
| `release-date` formats | ISO, `YYYY` year-only | + EU `DD.MM.YYYY` | + per-row mix of 4 formats |
| `duration` formats     | `seconds_int`, `mm:ss` | + `hh:mm:ss` | + `human_xm_ys` (e.g. `17m 35s`), per-row |
| Pool size per attr     | 2 | 3 | 4 |

#### K6 — Value noise   [knob_06_value_noise.md](../knobs/knob_06_value_noise.md) · [apply_knob_06_noise.py](scripts/apply_knob_06_noise.py)

Cell-level corruption using the FEBRL / Christen-Vatsalan operator suite —
char-level edits, OCR confusions (`O↔0`, `l↔1`, `rn↔m`, `cl↔d`), truncations,
whitespace/punctuation corruption, case corruption. These are *errors*, not
variants. Per-attribute-class rates (`noise_rate_primary / _key /
_secondary`). Source rows only — the fusion-gold artifact is never touched.
Easy / medium / hard scale the rates per attribute class; hard also
introduces noise into the primary label, which is zero at easy/medium.

| Music dial — fraction of cells corrupted | easy | medium | hard |
|---|---:|---:|---:|
| `noise_rate_primary`   |  0% | 0% |  1% |
| `noise_rate_key`       |  0% | 2% |  6% |
| `noise_rate_secondary` |  1% | 4% | 12% |

The operator mix also widens with level: easy uses only whitespace +
case corruption, medium adds typos / OCR confusions / taxonomy walks,
hard adds truncation and allows up to three edits per cell.

#### K3 — Per-source attribute drop   [knob_03_attribute_drop.md](../knobs/knob_03_attribute_drop.md) · [apply_knob_03_drop.py](scripts/apply_knob_03_drop.py)

Cell masking (NaN injection) at the (row, column) level. Per-attribute-class
rates (`drop_rate_primary / _key / _secondary`).

- **Cross-level nesting** `D_easy ⊆ D_medium ⊆ D_hard` is enforced
  structurally via shared per-cell uniform draws: all three masks are
  computed in one pass and easy/medium are shrunk against hard so a cell
  dropped at easy is also dropped at medium/hard.
- **Per-source shape:** transforms a *measured-baseline* missingness vector
  — easy `compress` (propagate values cross-source toward the
  min-missingness source), medium `identity`, hard `stretch`.
- **Constraints** applied at each level: fusion floor (cells used by fusion
  gold are protected), conflict preservation, single-source survivor cap.

| Music dial — fraction of cells dropped | easy | medium | hard |
|---|---:|---:|---:|
| `drop_rate_primary`   | 0% |  0% |  3% |
| `drop_rate_key`       | 2% | 10% | 25% |
| `drop_rate_secondary` | 5% | 15% | 35% |

**Audit:** `knob_03_baseline_missingness.csv`.

#### K10 — Source reliability differentiation   [knob_10_source_reliability.md](../knobs/knob_10_source_reliability.md) · [apply_knob_10_reliability.py](scripts/apply_knob_10_reliability.py)

Pure permutation reshuffle. After K1/K5/K6 have produced multiple variants
of the same (entity, attribute) cell, K10 picks **which source** carries the
gold-aligned variant. No new perturbation; no gold mutation (a SHA sentinel
verifies the fusion gold file is byte-identical before / after).

- **`per_attribute_concentration`** — at easy, the per-attribute winner gets
  the gold variant on a high share of cells (trust strategies trivially
  work); at hard the share drops (diffuse trust per attribute).
- **`error_correlation`** — at hard, perturbations burst along the (source,
  entity) axis: if source X mis-handled entity Y on attribute A, elevated
  probability of mis-handling on B too. Models "this source confused Y with a
  near-twin and got everything wrong."
- The per-attribute winner is re-derived on every run from the
  freshly-measured baseline `B[s, a]`; the YAML gives each source's share
  by name, and the loader logs a warning when the measured winner's share
  rises from one level to the next (easy → medium → hard).

| Music dial — share of `name` cells carrying gold variant (winner: musicbrainz) | easy | medium | hard |
|---|---:|---:|---:|
| musicbrainz (winner) | 85% | 65% | 40% |
| discogs              | 10% | 20% | 30% |
| lastfm               |  5% | 15% | 30% |

Per-attribute targets are authored per source × level (same shape for
`artist`, `release-date`, etc., each with its own measured winner).
Concentration shrinks toward equal-share at hard so trust-weighted
fusion strategies can no longer rely on a single winner.

`error_correlation` (burstiness along the source × entity axis):
`easy=0.0 / medium=0.20 / hard=0.50`.

**Audit:** `knob_10_realised.csv` (rate-based: `swap_rate = swap_cells /
reshufflable_count`, invariant to K3 pool shrinkage).

#### K8 — Schema naming divergence   [knob_08_schema_naming.md](../knobs/knob_08_schema_naming.md) · [apply_knob_08_naming.py](scripts/apply_knob_08_naming.py)

Column headers only — no cell values touched. Per-source rename on a 4-level
ladder via a per-domain `rename_table`:

| Level rung   | Example (`release_date`)        |
|--------------|---------------------------------|
| `descriptive`| `release_date`                  |
| `abbreviated`| `rel_dt`                        |
| `cryptic`    | `rdate`, `c_code`               |
| `anonymized` | `Attribute_3`, `col_07`         |

Bidirectional in S1: easy may rename *up* (anonymized → descriptive) when
baseline starts low; hard renames *down*. Regenerates
`input/schemamatching/sm_mapping.csv` so the SM stage sees the new headers.
Affects almost exclusively the SM stage.

| Music dial — `level_assignments` (rung per source) | easy | medium | hard |
|---|---|---|---|
| musicbrainz | descriptive | abbreviated | cryptic     |
| discogs     | descriptive | descriptive | abbreviated |
| lastfm      | descriptive | cryptic     | anonymized  |

All three sources rename in lockstep at easy (no SM challenge); at hard
the rungs disperse so a single SM matcher faces three different naming
distances from the target schema at once.

**Audit:** `knob_08_naming_intensity` (rung-weighted: descriptive=0 /
abbreviated=1 / cryptic=2 / anonymized=3).

### Knobs not used in the released variants

- **K7 — Value ambiguity / collision rate** ([knob_07_value_ambiguity.md](../knobs/knob_07_value_ambiguity.md)).
  Specified, but not used in the released variants. K7 was specced with
  three sub-parameters:
  - `referential_ambiguity_rate` — the active sub-parameter (e.g.
    `"Republic of Korea"` → `"Korea"`, `"John A. Smith Jr."` → `"John
    Smith"`). **Evaluator-compatible**: by design the K7 doc bounds its
    reach to "value pairs the lenient fusion evaluator already tolerates,"
    which is the same `lexical_extended_jaccard` close-enough mechanism the
    rest of the fusion stage uses. The blocker here is **substrate**:
    companies has the strongest case (`country`, partial `founders`
    shortenings) but `founders` only covers ~10% of rows → effective cell
    budget ≈ 190; games' developer/publisher names are usually full studio
    names; music's natural ambiguity is in cross-entity homonyms (e.g.
    `John Williams`), which belong to K2. Building K7's three new
    mechanisms — a per-attribute ambiguity map curated from gold values, a
    pre-injection lenient-evaluator probe, and per-cell rollback on
    committee collapse (no other knob needs that) — does not pay off
    against ≤200 cells in the strongest domain.
  - `multi_sense_conflict_rate` + `polysemy_rate_categorical` — **dropped**.
    These two would need fusion gold extended to *accepted sets* (e.g.
    `genre = {Rock, Alternative}` where any member counts), which the
    close-enough evaluator can't model (`Rock` and `Alternative` aren't
    lexically close — the song genuinely is both).
  - The cross-entity label-collision slice of K7's scope is part of K2 as
    a fourth niche-metric signal.

  The dimensions K7 targets (Value Ambiguity in Norm; Conflict Rate and
  Conflict Subtlety in Fusion) are under-stressed in the released variants.
- **K9 — Schema completeness / distractors** ([knob_09_schema_completeness.md](../knobs/knob_09_schema_completeness.md)).
  Specified for **Scenario 2** (fully synthetic tasks) only — S1 inherits
  the original use case's column set as-is, so K9 has nothing to inject
  against.

---

## Training, validation, and test sets in the variants

Each base task ships labeled files for four steps: a schema-matching gold
mapping, normalization validation / test sets, train / val / test splits
for entity matching per source pair, and fusion validation / test gold.
The variants copy some of these through, add regenerated companions, or
rewrite them to track the changes of the knobs.

| Stage | In `use cases/<domain>/base/input/` (base task) | In `use cases/<domain>/<level>/input/` (variants) | What changed and why |
|---|---|---|---|
| **Schema matching** | `sm_mapping_gold.json`, `target_schema.json`, taxonomy CSVs | `sm_mapping.csv`, `target_schema.json`, taxonomy CSVs | K8 (naming) renames source column headers, so the gold mapping is rewritten to reference the new headers (otherwise the SM matcher would predict against columns that don't appear in the gold). The target schema and the taxonomies are those of the base task; the target schema of the Music variants carries fewer value constraints (for example, no range for `duration`). |
| **Entity matching** | Per source pair: train / val / test splits, plus further labeled files in some domains (see [use cases/README.md](../use%20cases/README.md#entity-matching-files)) | Copies of the base splits **plus** `<a>_2_<b>_{train,val,test}_baseline_pruned.csv` and `<a>_2_<b>_{train,val,test}_corner_filled.csv` (Games: no `val`) | K2 removes entities and adds new ones (refills or interpolated near-twins), so the base splits would (a) reference records that the variant dropped and (b) miss the new K2-injected corner cases. The `corner_filled` splits are the closed-set version — every pair references only records present in the variant. |
| **Fusion** | `validation_set.xml`, `test_set.xml` (Products: `fusion_validation_set.csv`, `fusion_test_set.csv`; Papers: `fusion_val.jsonl`, `fusion_test.jsonl`) | `validation_set.xml`, `test_set.xml` (Papers: JSON Lines despite the extension) | No knob mutates the fusion gold: K10 verifies this with a SHA sentinel before and after the reshuffle, and K3 and K4 keep a surviving source carrier for every fusion-gold cell and entity. The variants keep the base task's 100 validation and 100 test entities and their graded values; where a variant removed the record that keys a gold entity, the gold record is keyed on a surviving member. Variants therefore measure fusion against the same entities and values as the base task, isolating the difficulty signal to source data quality. |
| **Normalization** | `validation.csv`, `test.csv` | `validation.csv`, `test.csv`, built on the variant's source values | The knobs change the source values, so each variant has its own normalization sets. The normalization committee runs against the variant's source data and is scored on its test set. |

### EM split regeneration mechanics

[lib/corner_case_miner.py:regenerate_em_splits](lib/corner_case_miner.py#L698)
runs per `(source_pair, split)` and writes two versions of each split,
`baseline_pruned` (step 1 only) and `corner_filled` (steps 1 and 2):

1. **Carry over surviving originals.** Every original `(id1, id2, label)`
   whose IDs both still exist in the post-K2 / post-K4 source frames is
   kept verbatim. Pairs whose IDs were K2-removed are dropped. K2 is the
   only knob that can invalidate an original pair — no other knob
   renames, merges, or splits entity clusters, so labels never flip for
   surviving IDs.
2. **Backfill to the original `(size, positive_ratio)`.** Each split
   inherits the size and positive ratio of the corresponding original
   split. The free slots are filled from the corner-mined pools only:
   - **Corner positives** = K2-interpolated cross-source near-twin pairs.
   - **Corner negatives** = hard-negative-gated cross-cluster pairs
     (close enough to confuse a blocker but in different gold clusters).
3. **Enforce disjointness across train / val / test.** A consumed-pairs
   tracker prevents the same record pair from appearing in two splits.
   If a backfill pool runs dry — common at hard, where K2 has stripped a
   lot of entities — the split undersizes rather than overlaps; a warning
   is logged if the realised positive ratio drifts more than 2pp from
   target.
4. **Re-run post-K4.** K4 may demote additional rows at hard, so the
   regen runs a second time with a refreshed `ids_present` filter
   against the post-K4 sources. Same pools, fresh survivors filter — no
   regenerated pair references a record that K4 removed.

### EM file families in the variants

- **Closed-set (`*_corner_filled.csv`)** — every pair references records
  present in the variant. This is the **primary** difficulty surface:
  matchers are measured on a test consistent with the post-variant
  population, and the corner-case ratio tracks K2's dial. The test split
  is the variant's evaluation set for blocking and entity matching.
- **Survivors (`*_baseline_pruned.csv`)** — the base task's pairs whose
  records survive in the variant, the starting point of `corner_filled`.
- **Open-set (copies of the base splits)** — every pair references the
  base task's records, some of which the variant dropped, so these files
  penalise matchers for records that are not in the variant. They are not
  evaluation sets.

---

## Phase 3 — Validation per level

```
use cases/<domain>/<level>/input/
                    │
                    ▼
   ┌────────┬────────────┬─────────────┬──────────────┬─────────┐
   ▼        ▼            ▼             ▼              ▼
  SM      Norm       EM-blocking    EM-matching     Fusion       (committees, frozen YAML SHAs)
                    │
                    ▼
   validation/<domain>/<level>/metrics.json
                    │
                    ▼
  analyze_monotonicity.py   →   monotonicity_report.csv
                                ├── committee macro_f1     (baseline → easy → medium → hard)
                                ├── best-member ceiling
                                ├── ceiling_responsiveness (per-knob Pearson on best-member F1)
                                └── per-knob intensity verdict (K5 distinct families, K8 naming rank,
                                                                K10 realised swap rate)
                    │
                    ▼
   validation/<domain>/monotonicity_report.md
```

[scripts/validate_variant.py](scripts/validate_variant.py) runs each
committee against a packaged variant and writes a `metrics.json` whose every
leaf carries `_baseline` and `_delta` twins (so a reader can see both
absolute performance and the drop versus baseline at a glance). It refuses to
start if the committee YAML SHAs diverge from those recorded in
`baseline_metrics.json` — drift would silently invalidate every delta.

[scripts/analyze_monotonicity.py](scripts/analyze_monotonicity.py) consumes
the per-level metrics across `baseline → easy → medium → hard` and decides
whether each stage's committee macro_f1 and best-member F1 (the
strongest individual matcher's F1, which is what a downstream user
would actually deploy) drop monotonically with difficulty. It also writes
per-knob audit verdicts driven by the realised CSVs above. Its
`monotonicity_report.md` per domain rolls those up into a human-readable verdict.

A second analyzer, [analyze_ablation.py](scripts/analyze_ablation.py),
compares per-knob ablation variants (one knob at hard, all others at easy;
generated and validated by
[run_ablation_validation.py](scripts/run_ablation_validation.py)) with the
baseline and the full hard variant, to attribute the cross-level drop to
individual knobs.

---

## Reading order

1. This file — pipeline mental model.
2. [PIPELINE.md](PIPELINE.md) — runbook with commands per phase.
3. [knobs/README.md](../knobs/README.md) — canonical knob order rationale +
   cross-cutting rules.
4. Individual [knobs/knob_NN_*.md](../knobs/) — full per-knob spec.
