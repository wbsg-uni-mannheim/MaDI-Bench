# Knob 7 — Value ambiguity / collision rate

**Status:** specified, not used in the released variants. **Scenario:** S1 + S2.

> **Note.** Everything below is reference design.

## Definition

**Within-entity referential ambiguity only.** Cross-entity label collisions ("different things look the same") are explicitly *not* in Knob 7 — they belong to Knob 2 as a label-collision distractor-selection signal. Knob 7 targets its declared dimensions (Value Ambiguity (Norm), Conflict Rate (Fusion), Conflict Subtlety (Fusion)); the released variants leave those dimensions under-stressed.

## Status

Companies, games and music have thin Knob 7 substrate (companies strongest, but founders coverage ~10%; games and music near zero). Combined with the cost of the per-attribute ambiguity map, lenient-evaluator probing, and rollback machinery — all for one sub-parameter — **the released variants do not use Knob 7**. The dimensions Knob 7 targets are **under-stressed in the released variants**.


## Sub-parameters

### Active

- `referential_ambiguity_rate` — fraction of (entity, attribute) cells where at least one source is mutated to carry a referentially underspecified value (e.g. `"Republic of Korea"` → `"Korea"`; `"John A. Smith Jr."` → `"John Smith"`). Individually defensible but referentially underspecified relative to the gold.

### Dropped

- `multi_sense_conflict_rate` — dropped: would require extending the fusion gold to accepted sets (e.g. `{Rock, Alternative}`), incompatible with the current fusion evaluator.
- `polysemy_rate_categorical` — dropped for the same reason.

## Easy / Medium / Hard

| Level | `referential_ambiguity_rate` | Target state | Generator action |
|---|---|---|---|
| **Easy** | 0% | Every value is referentially specific. Disambiguators present where natural language allows them. | Normalize-down: where a source carries a known-ambiguous form, replace with the more specific form propagated from a sibling source or the gold. |
| **Medium** | low | A meaningful tail of cells carry shortened/ambiguous forms; most values remain specific. | Identity / light injection. Most domain baselines likely sit here. |
| **Hard** | moderate | Pervasive shortened forms across key and secondary attributes. Naive fusion strategies that pick the modal value frequently land on the underspecified form. | Per-cell rewrite from specific → ambiguous form, drawing from a per-attribute ambiguity map. |

## Generator mechanism — per-attribute ambiguity map

Per-domain artifact mapping specific forms to their ambiguous shortened forms, constructed from actual gold values (not a generic gazetteer). Example for companies:

```yaml
country:
  "Republic of Korea": "Korea"
  "Democratic Republic of the Congo": "Congo"
founders:
  "John A. Smith Jr.": "John Smith"
  "Maria Garcia-Lopez": "Maria Garcia"
```

Small, manually curated per domain. Cells not covered by the map are not eligible for Knob 7 mutation.

## Composition

- **Knob 1 (paraphrase):** orthogonal — Knob 1 makes the same referent look different at the surface; Knob 7 makes the value referentially underspecified.
- **Knob 5 (formats):** orthogonal — Knob 5 reformats a fully-specified value; Knob 7 *removes information* from it.
- **Knob 6 (noise):** distinct — noise produces *wrong* values; Knob 7 produces *insufficient* values.
- **Knob 2:** handles cross-entity label collisions.
- **Per-source treatment:** Knob 7 is *not* per-source-shaped (unlike Knobs 3/6). Cell-level injection across the corpus.

## Fusion safety

**Bounded by lenient fusion (intended).** Knob 7's reach is limited to value pairs the lenient fusion evaluator already tolerates. On committee collapse: **rollback** the offending injections (no gold mutation). Effective range = "what lenient fusion already accepts."

### Monotonicity guards

1. **Per-cell clean-survivor floor** (mirrors Knobs 3 and 6).
2. **Committee check.** Sharpest drop on naive most-frequent / random-pick fusion strategies.
3. **Rollback on collapse.**

## Committee expectations

- **SM / Blocking:** flat / mostly flat.
- **EM:** small monotone drop.
- **Fusion:** primary target. `longest_string` strategy specifically resists this knob — that resistance is the intended discrimination signal.

## Per-domain notes

- **Companies:** **non-zero baseline** (DBpedia `England` vs `United Kingdom` co-occur for UK entities). Easy must *normalize-down*. Founders coverage ~10% → effective cell budget ~190 cells.
- **Games:** at easy. Developer/publisher names usually full studio names. Effective surface even thinner than companies.
- **Music:** at easy for the *narrow* sub-parameter. Cross-entity homonyms (`John Williams`, `Crash`) belong to Knob 2 label-collision, not here.

## Provenance

`transform_fn=referential_ambiguate`, `transform_params={ambiguity_map_entry, original_form, ambiguous_form}`. Rollback emits mirror `transform_fn=rollback_for_committee`.

## Algorithm selection

**Not used in the released variants.** Per the *Status* section above and [cross_cutting.md §Per-knob fix-strategy defaults](cross_cutting.md#per-knob-fix-strategy-defaults) (row 7), no algorithm selection is performed for Knob 7. The dimensions Knob 7 targets (Value Ambiguity, Conflict Rate, Conflict Subtlety) are under-stressed in the released variants. An implementation would follow the same tier framework as the other knobs: a deterministic per-attribute ambiguity map (Tier A, authored from gold values as the card already specifies) for the single active `referential_ambiguity_rate` sub-parameter, with no LLM involvement. The cross-entity label-collision signal is part of Knob 2 ([knob_02_niche_density.md](knob_02_niche_density.md#metric-set), `label_collision` metric), so that slice of the Knob 7 scope is in the released variants.
