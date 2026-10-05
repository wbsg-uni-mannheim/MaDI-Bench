# Companies integration – method report

**No labeled evaluation or supervised training was performed.** There was no label service, reference mapping, gold clusters or fusion teacher. All numbers below are label-free structural diagnostics computed from the saved artifacts (`work/diagnostics.jsonl`), not accuracy/precision/recall/F1. Matching decisions were made with deterministic rules; my own inspection of example pairs guided the rules but was never turned into a reference set.

Pipeline: `work/s1_schema.py` → `s2_normalize.py` → `s3_blocking.py` → `s4_matching.py` → `s5_cluster.py` → `s6_fusion.py` → `export.py`. `work/rebuild.sh` runs all stages offline. The only external resource is a cached file of general-purpose text embeddings (`work/state/emb_cache.json`), created once by `work/s0_embed_cache.py` and used for industry-label lookup only. Rebuild does not call the network. Revisions are saved in `work/revisions/` (r1 = first complete pass, r3 = intermediate; the final revision is the current `submission/`).

## 1. Schema matching (`sm_mapping.csv`)
| target | dbpedia | forbes | fullcontact |
|---|---|---|---|
| id | id | id | id |
| name | org_nm | name | nm |
| founded | est_yr | – | fy (DD.MM.YYYY) |
| country | ctry | country | cn |
| city | hq_city | – | cy |
| industry | sect (0.9) | industry | – |
| assets | tot_ast | assets | – |
| revenue | ann_inc (0.8) | revenue | – |
| keypeople | kp_nm (0.8) | – | kp (0.8) |

Unmapped: `forbes.website` (the Forbes profile URL, a duplicate of the id). The mapping was chosen by comparing meaning and example values, not column names.

Uncertainty:
- dbpedia `ann_inc` could mean income rather than revenue. Values match Forbes revenue for shared companies, so I mapped it to revenue.
- The `kp*` fields hold key people in general (executives as well as founders), while the target wants founders.

## 2. Normalization (coverage: `work/taxonomy_coverage.csv`; plan: `work/taxonomy_plan.json`)

**Names**
- Removed legal-form suffixes (Inc, Corp, Ltd, AG, SA, S.p.A., OAO, Pty, …) and trailing dbpedia disambiguators such as "(company)".
- Recovered 17 mojibake dbpedia names ("�?guas de Portugal") from the percent-encoded dbpedia URI.
- Matching keys come in two forms:
  - **Strict:** legal forms removed.
  - **Loose:** additionally removes group, holding(s), international, and/of.
- A parenthetical that names a country ("Coop (Italy)", "Safeway (Canada)") is kept in the key as a regional-arm marker.

**Founded**
- Year taken from YYYY-MM-DD or DD.MM.YYYY (both formats occur in both sources).
- Character-noise repair (O/o→0, l/I/q→1, B→8, G→6 …).
- Years kept only within 1700–2016. Output is YYYY-01-01 (the evaluator compares year only).
- Canonical rate: 0.99 dbpedia, 1.0 fullcontact.

**Country (CLDR, exhaustive)**
- Matching order: canonical names and aliases → token permutation ("States United") → manual aliases (England/Scotland/Wales→United Kingdom, Korea (Republic of)→South Korea, PRC, Burma, …) → fuzzy match (ratio ≥ 80) for typos ("Unitcd States").
- Canonical rate is at least 0.99 in every source.
- 57 values stay unmapped, e.g. bare "United" (US or UK) and "Soviet"-era entities.

**City**
- dbpedia concatenates several values ("MassachusettsFramingham, Massachusetts", "SeoulIncheon"). I split them at case boundaries, but only where one side is a known city or region, so noise such as "TriPoli" is not split.
- Segments that are US states, provinces or countries are dropped. City-states such as "Singapore" are kept when nothing else remains.
- The fused city is voted over the members' city segments.

**Industry (GICS "Industry Name", exhaustive)**
- 3,029 values mapped with a manual dictionary of frequent Forbes and dbpedia categories (`work/industry_overrides.json`).
- 435 exact matches.
- 2,973 mapped to the nearest GICS industry or sub-industry label by embedding cosine similarity (≥ 0.40, inference only).
- 320 unmapped values (e.g. "Final good", "Law") left null.
- All output values are in the taxonomy.

**Assets / revenue**
- Parses $/€/£/ISO-code prefixes, million/billion words (including typos like "millikn"), EU thousands format "4.796.000.000,0" and US commas.
- **Currencies are not converted.** Face values are kept, because the same synthetic company carries identical "€14.8 billion" strings in several sources.
- Scale heuristics (documented, unverifiable):
  - Forbes bare revenue < 5000 is read as US$ billions (Forbes style "148.70").
  - A bare dbpedia value, or a bare Forbes asset value, below 100 is read as billions; 100–100,000 is read as millions.
  - 0 is treated as missing.
  - Descriptive text ("Mid-sized private holdings") and ranges become null.
- Parse rates: Forbes 0.98–0.99; dbpedia 0.87 (assets) and 0.98 (revenue).

**Key people**
- Parses list literals and ';' or ','-separated strings.
- Strips role labels ("CEO:", "(founder)") and drops generic bodies ("Executive Board", "leadership team").
- Serialized as a JSON list.

## 3. Blocking (`blocking/candidates.csv`, 51,103 cross-source pairs; all written, nothing sampled)
Union of four methods:
1. Exact strict-key match.
2. Acronym key (initials of a name with 3 or more tokens vs a short name).
3. Character-3-gram TF-IDF top-8 nearest neighbours in both directions for every source pair (cosine ≥ 0.35).
4. Near-duplicates within fullcontact.

Results:
- Full cross product is 47.4M pairs; reduction ratio is 0.9989.
- By source pair: dbpedia–forbes 22,340; fullcontact–dbpedia 21,415; fullcontact–forbes 7,348.
- Records with no cross-source candidate: dbpedia 3,060 (mostly niche companies with distinctive names), forbes 80, fullcontact 47.
- Every emitted correspondence is inside the candidate set (diagnostic `correspondences_in_candidates` = 1.0).

Without reference matches, pair completeness cannot be measured.

## 4. Entity matching (`s4_matching.py`)
Each candidate pair gets a **name class** and an **evidence score e** from non-name attributes.

Evidence weights:
- Country agree +1. Country conflict −0.7, or −0.3 when fullcontact is involved, because its country field is observably noisy (Google→Kuwait, GM→South Africa). China/Hong Kong/Taiwan pairs count as compatible.
- A parenthetical country that contradicts the other record's country: −1.
- City: token-subset agreement +1, conflict −0.3.
- Founding year: within 1 year +1, otherwise −1.5.
- Key-people surname overlap +1.5, disjoint −1.
- Revenue and assets: ratio ≤ 1.25 gives +1 each; ratio > 3 gives −1 each.
- Same GICS industry +0.5.
- dbpedia type descriptor on one side only ("(automobile)", "(retailer)"): −0.5.

Name classes and acceptance thresholds:

| class | name condition | accept if |
|---|---|---|
| N1 | strict-key token sets equal, or equal after removing spaces ("Commerz Bank"/"Commerzbank") | e ≥ −1.5, and not (descriptor mismatch with no positive evidence) |
| N2 | loose keys equal ("Allianz Group"/"Allianz") | e ≥ −0.3 |
| N3a | 1-edit typo, same first letter, length ≥ 8 | e ≥ 0.5 |
| N3b | 2 edits | e ≥ 1.5 |
| N4 | one name's tokens contain the other's, extra tokens only generic (industries, technologies, foods, products…) | e ≥ 1.0 |
| N5 | acronym | country agrees and is not the US, e ≥ 1 |

Rationale:
- The data contains synthetic **hard-negative families**, e.g. "Wasatch Meridian Network" vs "Wasatch Meridian Health Network", or two dbpedia "Wellington LedgerWorks" records founded in 1997 vs 2003 with different people. So an extra distinctive token blocks a match, and year or key-people conflicts weigh heavily.
- Extra geographic tokens ("DONG Energy UK", "Dentsu Indonesia", "Toyota (GB)") mark subsidiaries or regional arms. Under the entity definition these are not the parent, so they are never harmless.

I inspected pairs near each threshold. Examples of what the rules decided:
- **Rejected:** GE Energy/OGE Energy, Deere (automobile)/Deere & Co, Northern Christian Radio/NCR, Fujitsu Technology Solutions/Fujitsu.
- **Accepted:** Exon Mobil/Exxon Mobil, ICBC acronym, Hormel Foods/Hormel.

Accepted pairs by class: N1 2,007; N2 119; N3 11; N4 15; N5 4.

## 5. Clustering (`s5_cluster.py`)
- Greedy constrained union–find over accepted edges. Edges are sorted by name class first (N1 > N2 > N3/N4 > N5), then by evidence, then by id for determinism.
- Constraint: at most one dbpedia and one Forbes record per cluster, because each is a unique-entity listing. Fullcontact may contribute several rows, since it contains exact duplicates (e.g. 10 "HugeDomains.com" rows); within-fullcontact edges are allowed only for N1.
- 23 edges were rejected by the constraint (`work/state/s5_rejected_edges.csv`), e.g. Metro AG vs Metro Inc, Merck KGaA vs Merck & Co, and the second twin of each synthetic family.
- Correspondences are all cross-source pairs inside the final clusters.

Diagnostics:
- 14,332 records → 12,574 entities.
- Cluster sizes: 11,145 singletons; 1,112 of size 2; 311 of size 3; 5 of size 4; 1 of size 10 (fullcontact duplicate rows).
- Sources per cluster: 2 sources 1,122; 3 sources 300.
- Records in multi-source clusters: dbpedia 957, Forbes 995, fullcontact 1,212.
- No cluster contains two dbpedia or two Forbes records.
- 34 of 1,428 multi-record clusters have founding years more than 1 year apart. I inspected these: they are real companies whose sources disagree (e.g. BP 1935 vs 1870, Nokia) or format-swapped noise, not merges of different companies.

## 6. Fusion (`s6_fusion.py`, provenance in `work/state/s6_provenance.json`)
Resolved attribute by attribute, only from the cluster's own members:

| attribute | resolver |
|---|---|
| name | cleaned name from Forbes, else dbpedia, else fullcontact |
| founded | year vote; ties go to dbpedia |
| country | weighted vote (fullcontact 0.5) |
| city | segment vote across members |
| industry | Forbes, else dbpedia |
| assets / revenue | Forbes, else dbpedia |
| keypeople | vote over normalized name sets; ties go to dbpedia |

`id` is the cluster id, which equals `_id` (the id of the highest-priority member). No value is invented; missing stays missing.

Output density: founded 0.76, country 0.88, city 0.74, industry 0.49, assets 0.16, revenue 0.20, keypeople 0.03. Validity checks: country and industry are 100% in taxonomy; dates are 100% well-formed and within 1700–2016; money values are integers within schema bounds; keypeople are JSON lists. Membership and fused ids match 1:1 in both directions, and all ids resolve to source rows.

## Known weaknesses / unresolved uncertainty
- **Money scale and currency are ambiguous.**
  - Synthetic records use bare numbers ("121.75", "742.5") whose unit cannot be determined.
  - dbpedia values are sometimes in local currency (e.g. yen for Mabuchi).
  - Forbes is preferred when present, but dbpedia-only entities rely on the documented scale heuristics.
- **Key people:** the target wants founders, but the sources mix founders and executives, and some names are token-shuffled noise ("Ellison Mara"). I keep all parsed person names.
- **Industry:** the mapping to GICS is a judgment. The manual dictionary and embedding fallback may differ from the curator's mapping, especially for broad dbpedia categories like "Manufacturing" or "Technology".
- **Missed matches** (conservative by design):
  - renamed or longer legal names: "The Goodyear Tire & Rubber Company" vs "Goodyear", "DaVita" vs "DaVita HealthCare Partners";
  - a name with an extra country token: "Stockland Australia" vs "Stockland";
  - generic single-word names are not matched.
- **Possible wrong merges:** exact-name pairs with little supporting evidence (e.g. dbpedia "Macy's" (1858 store) was chosen over "Macy's, Inc." for the Forbes listing). The founded-year vote can also pick a noisy dbpedia value when only two sources disagree.
- **Output conventions:** city keeps the first plausible segment (e.g. district-level names such as "Suan Luang District" can win over "Bangkok"). Names keep "Company"/"Group" when not preceded by a legal form.
- **Rejected approaches:**
  - trusting fullcontact country (too noisy);
  - converting currencies (would contradict identical cross-source face values);
  - loose token-set or containment matching without an extra-token whitelist (it merged subsidiaries and synthetic twins);
  - supervised matchers (not permitted).
