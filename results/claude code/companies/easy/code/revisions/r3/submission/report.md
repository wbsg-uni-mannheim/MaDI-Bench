# Companies integration — method report (revision r3)

**No labeled evaluation and no supervised training were performed.** No labels, judge, teacher or matcher checkpoint
was used, and no model was fitted. All numbers below are structural, label-free diagnostics computed from the saved
artifacts. None of them is a precision, recall, F1, pair-completeness or fusion-accuracy figure.

Sources: `dbpedia` (10,089 rows), `forbes` (2,000), `fullcontact` (1,933). Native ids are kept verbatim everywhere.
Rebuild: `work/rebuild.sh` runs `s1_schema → s2_normalize → s3_blocking → s4_features → s4_bands → s4_match →
s5_cluster → s6_fusion → diagnostics`. It needs no network access. The name embeddings were computed once by
`work/embed_cache.py` (general-purpose `text-embedding-3-small`, inference only) and cached in `work/state/emb_name.npy`.
My pair decisions live in `work/manual_decisions.csv`. I verified the rebuild in a fresh copy with `submission/` removed:
all five data files came out identical.

## 1. Schema matching (`s1_schema.py`, `work/state/schema_mapping_inventory.csv`)
Columns were mapped by meaning after inspecting values:
- `id`, `name`, `country`, `city`, `founded`, `industry`, `keypeople`, `assets` and `revenue` map to the attribute of the same name.
- `forbes.website` duplicates the id URL, so it is unmapped. I only use its slug as an extra name key for matching.
- Forbes money is in **US$ billions** (`3,124.90`).
- dbpedia money has mixed units: raw dollars, millions, and Forbes-style billion strings. Its mapping confidence is 0.7.
- fullcontact has no industry, assets or revenue. Forbes has no city, founded or keypeople.

## 2. Normalization (`s2_normalize.py`, `work/aliases.py`, `work/taxonomy_plan.json`, `work/taxonomy_coverage.csv`)
- **Country:** mapped to CLDR `Country Name` via the short/variant columns plus explicit aliases (USA, UK variants,
  England/Scotland/Wales → United Kingdom, Korea (Republic of), Russian Federation, …). Historical states such as
  "British Raj" are left null. Canonical rates: dbpedia 0.998, forbes 1.0, fullcontact 0.999.
- **Industry:** mapped to GICS `Industry Name`, which is an exhaustive list. All 82 Forbes categories are mapped by hand.
  About 370 dbpedia free-text values are mapped where the meaning is clear. Ambiguous values ("Manufacturing",
  "Technology") stay null. Canonical rates: forbes 1.0, dbpedia 0.79.
- **Founded:** parsed to `YYYY-MM-DD`. 66 values outside 1700–2016 (e.g. year `0019`) are nulled.
- **Money:** Forbes ×1e9. For dbpedia:
  - strings in the Forbes billion format (`d,ddd.dd`) and values below 100 are treated as billions;
  - values from 100 to 1e5 are treated as millions;
  - anything larger is treated as raw dollars.

  This is a heuristic without independent confirmation. It only matters for entities that have no Forbes record.
- **Names:**
  - The display name drops legal-form suffixes (Inc, Corp, Ltd, plc, AG, SA, …) and, for dbpedia, the trailing
    Wikipedia disambiguation in parentheses.
  - Mojibake or missing dbpedia names are repaired from the URI.
  - The matching key also drops generic tokens (group, holdings, the), folds accents and maps "&" to "and".
- **Keypeople:** parsed as lists. Placeholders such as `<N A>` are dropped. Disambiguations like "(businessman)" are removed.

## 3. Blocking (`s3_blocking.py`)
The candidate set is a union of four methods, run for each source pair:
1. exact name-key match (including Forbes slug and fullcontact tagline splits);
2. char-3-gram TF-IDF top-5 in each direction, cosine ≥ 0.5;
3. cached name-embedding top-5 in each direction, cosine ≥ 0.6;
4. dbpedia↔forbes pairs whose revenue **and** assets strings are numerically identical.

Results:
- 17,661 candidates out of 43.5M cross-source pairs (reduction ratio 0.99959).
- By source pair: dbpedia–forbes 8,116; dbpedia–fullcontact 6,784; forbes–fullcontact 2,761.
- The largest per-record candidate count is 119.
- Records with no candidate: dbpedia 5,719 (most dbpedia companies have no counterpart in the smaller sources), forbes 179, fullcontact 186.
- The exported `blocking/candidates.csv` adds 2 pairs that only arise through transitive closure of the final clusters.
  They are flagged in `work/state/closure_candidates.csv`.

This is candidate coverage only. It is not recall.

## 4. Entity matching (`s4_features.py`, `s4_bands.py`, `s4_match.py`)
Evidence per pair:
- name-key equality, edit ratio, token-set ratio, IDF of shared and extra tokens, geographic extra tokens, embedding cosine;
- agreement or conflict on country (Hong Kong/Macao treated as compatible with China), city and founding year (±1);
- financial ratio;
- exact equality of the raw revenue/assets strings.

Missing values count as unknown.

**Key data observations that shaped the rules:**
- dbpedia records often carry Forbes numbers verbatim. 135 of 145 dbpedia↔forbes pairs with identical revenue+assets
  end up in the same cluster through name evidence alone.
- Some of those carriers are divisions: CBS Radio, Boeing Commercial Airplanes, Renault Sport.
- fullcontact contains exact duplicate rows (HugeDomains ×10, Google ×3) and near-duplicates.
- Country and city fields are noisy or swapped across linked records. Example: Forbes lists Continental Resources in
  Romania while its linked dbpedia record says USA.

**Decision tiers:**
- **Exact name key**, with no conflicts or with at least one support and at most one conflict, and no non-company
  dbpedia disambiguation → accept.
- **Financial-string equality** plus a shared distinctive name token → accept.
- **Near-exact spelling** (ratio ≥ 0.92, no geographic extra token, no conflicts, country agrees or other support) → accept.
- **Extra tokens only generic** ("Foods", "Industries", "International", …), country agrees, embedding ≥ 0.75 → accept.
- **Acronym**, country agrees → accept.
- **Containment** without geographic or division words, country agrees, plus city or year agreement → accept.
- **Everything else in the review band** (exact names with conflicts; similar names with ratio ≥ 0.85, containment,
  equal financials or embedding ≥ 0.85): I decided these pair by pair from the records, following the entity
  definition. Regional arms and divisions such as "Siemens Canada", "Kia Motors America" and "Walt Disney Animation
  Studios" were rejected unless source-value evidence tied them. Careers/press/help pages of the same organisation
  were accepted. Result: 328 accepted and 625 rejected. They are stored in `work/manual_decisions.csv`, with rejection
  as the default. These decisions are part of the pipeline, not a reference set, and they are not used to score anything.
- **One-to-one:** dbpedia–forbes edges must be mutual best by score. Each fullcontact duplicate group keeps only its
  best partner per source, while a dbpedia or Forbes record may absorb several fullcontact near-duplicate profiles.

**Accepted group edges:** exact key 1,443; manual 325; financial+name 45; containment 20; near-exact 18;
generic suffix 13; acronym 2.

## 5. Clustering (`s5_cluster.py`)
- Exact-duplicate fullcontact rows are merged first.
- Accepted edges are then applied greedily, strongest first, and a merge is refused if the cluster would hold more than
  one dbpedia or more than one Forbes record. No edge was refused in r3.
- `cluster_id` is a native id of a member, preferring forbes, then dbpedia, then fullcontact.
- Correspondences are all cross-source pairs within each final cluster: 1,891 direct and 20 transitive, 1,911 in total.

## 6. Fusion (`s6_fusion.py`, `work/state/fusion_provenance.csv`)
- Values come only from the cluster's own normalized members.
- Each source contributes one vote (its most frequent value), so duplicate fullcontact profiles cannot outvote.
- Ties are broken by source priority:

| Attribute | Priority on ties |
|---|---|
| name | forbes, dbpedia, fullcontact |
| country | forbes, dbpedia, fullcontact |
| industry | forbes, dbpedia |
| founded (voted by year) | dbpedia, fullcontact |
| city | dbpedia (cleaned), fullcontact |
| assets / revenue | forbes, dbpedia (values outside the schema range are discarded) |

- **City cleaning:** concatenated dbpedia strings like `GuangdongShenzhen` or `Newark, New JerseyPrudential Headquarters`
  are split. The cleaner prefers the segment that equals the fullcontact city, then skips region names and address
  fragments, then keeps the part before the comma.
- **Keypeople:** the union of member lists, serialized as JSON.
- Missing values stay missing; nothing is imputed.
- The share of cells whose contributing sources disagree: assets 6.2%, revenue 7.3%, name 4.6%, industry 2.2%,
  city 1.4%, founded 0.8%, country 0.6%.

## Diagnostics (final artifacts, `work/diagnostics.jsonl`, revision r3)
- **Ids and coverage:** 14,022/14,022 source ids in membership, 0 unresolved, 0 source mismatches, coverage 1.0 per source.
- **Rows:** 12,290 fused rows; the `_id` set equals the membership `cluster_id` set.
- **Correspondences:** 100% inside candidates, 100% same-cluster, 100% cross-source.
- **Cluster sizes:** 10,777 of size 1, 1,311 of size 2, 192 of size 3, 8 of size 4, 1 of size 5, 1 of size 10 (HugeDomains duplicates).
- **Sources per cluster:** 1 source 10,783; 2 sources 1,321; 3 sources 186. Singleton share 0.877.
- **Same-source multiplicity:** no cluster holds 2+ dbpedia or 2+ Forbes records; 27 clusters hold fullcontact duplicates.
- **Validity:** country in CLDR 1.0; industry in GICS 1.0; founded format and range 1.0; assets and revenue in range 1.0;
  keypeople list length 1.0.
- **Density:** name 0.9998 (2 fullcontact rows have no name at all); founded 0.82; country 0.96; city 0.78;
  industry 0.53; assets 0.19; revenue 0.27; keypeople 0.01.
- **Traceability:** sampled 3- to 10-member clusters trace back to their own members (Bank of China, Qualcomm, Encana, Costco, …).

## Rejected approaches
- Plain connected components over similarity edges, and name-containment without attribute support: these merged
  parents with divisions (Gazprom/Gazprom Neft, Renault/Renault Sport).
- Letting country disagreement veto exact-name pairs: countries are too noisy (BBMG, Aurubis, Fujitsu).
- City from fullcontact first: it picked subsidiary locations (Costco → Enfield).
- Any supervised or pseudo-labeled matcher: excluded by the task rules.

## Known weaknesses and uncertainty
- I accepted exact Forbes-number carriers even when the dbpedia record is a division (e.g. CBS Radio ↔ CBS). This follows
  the evidence of value copying, but it conflicts with the literal entity definition. Some of these may be wrong.
- Pair-by-pair decisions reflect my judgement about parent, group and listed-subsidiary boundaries (e.g. COSCO vs China
  Cosco Holdings, ChinaCoal vs China Coal Energy). These cases are ambiguous.
- The dbpedia unit heuristic for money is unverified. Forbes and dbpedia names contain injected typos
  (e.g. "Costco Wholsale"). Name fusion prefers Forbes spelling.
- Country and city votes among noisy sources can pick a copied or swapped value (e.g. Siemens → Canada).
- About 30 unmatched fullcontact/dbpedia pairs share an exact city and year but have unrelated names. I left them
  unmatched; a few (Serpro, WEC Energy) are probably true matches that were never blocked.
