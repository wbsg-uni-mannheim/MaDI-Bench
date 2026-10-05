# statistics/

Central XLSX reporting for the variant pipeline — one workbook per domain. Re-generate after every new baseline / variant / validate run.

## Regenerate

```
source .venv/bin/activate
PYTHONPATH=. python usecases_synthetic/scripts/build_statistics.py --domain companies --domain games --domain music --domain papers --domain products
```

Or for a specific subset of domains:

```
python usecases_synthetic/scripts/build_statistics.py --domain music --domain products
```

## File layout

Each workbook has sixteen sheets:

1. **evaluation_legend** — what each sheet measures: the four EM evaluation surfaces (baseline- or variant-trained model × baseline or variant test gold) and the surface behind each `committee_summary` row.
2. **sizes** — source row counts across `baseline` + `easy` + `medium` + `hard`.
3. **splits** — combined train / val / test breakdown: per-EM-pair `total` / `positive` / `negative` / `pos_rate` from each level's `<source>_2_<source>_<split>.csv` gold file, plus a fusion (validation / test) entity-count block at the bottom. A file missing at a level leaves that level's cells blank; a split with no file at any level is left out.
4. **examples** — 10 entity clusters per domain, selected by **value-set drift** between baseline and hard (Jaccard distance over record values, ignoring column names — so K8 column renames do not dominate selection) among the clusters whose records exist at every level. Each cluster expands to every configured source-pair edge × 4 levels, rendered as `k=v; k=v; ...` strings. Reading down a record column makes K1/K5/K6 value mutations + K10 corruption directly visible.
5. **transformations** — same 10 clusters as `examples`, but rendered per-record and per-field. For each cluster member, every field is one row with values across baseline / easy / medium / hard side-by-side. Field alignment is position-based against the baseline column order (K8 renames preserve position), so the canonical baseline column name labels each row regardless of how the level renames it.
6. **committee_summary** — per-stage (SM, Norm, EM-blocking, EM-matching, Fusion) committee macro headline + best-member value + best-member name, across all 4 levels.
7. **per_member** — every committee member's headline metric value across all 4 levels. Grouped by stage with a section header row.
8. **selection_map** — the method that each per-attribute member (`pydi_per_attribute_optimal` in fusion, `rule_per_attribute_optimal` in normalization) chose per attribute, across all 4 levels.
9. **EM match (train=… test=…)** — four sheets, one per EM matching surface: model trained on the baseline (`BL`) or variant (`Var`) training data, scored on the baseline test gold (`BL`) or the variant's regenerated test gold (`Var`). `train=Var test=Var` is the EM matching row of `committee_summary`.
10. **EM block (train=… test=…)** — the same four surfaces for EM blocking; the EM blocking row of `committee_summary` is `macro_pair_recall`.

## Where the data comes from

- Baselines: `usecases_synthetic/baselines/<domain>/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/<domain>/<level>/metrics.json`
- Source / EM / fusion files: `use cases/<domain>/base/input/` for the base task, `use cases/<domain>/<level>/input/` for variants. Products honors `data_root: usecases_synthetic/usecases` per [config/domains/products.yaml](../config/domains/products.yaml).

## Source code

[scripts/build_statistics.py](../scripts/build_statistics.py).
