# Companies integration: method report (label-free)

**No labelled evaluation was done and no supervised training was used.** No matcher was trained or fine-tuned, I made no pseudo-label split, and I had no teacher or judge service. Every number below is a structural, label-free diagnostic computed from the saved artifacts. None of them is precision, recall, F1, pair completeness or fusion accuracy.

The whole pipeline can be rebuilt with `work/rebuild.sh`. It runs `s1_schema.py` → `s2_normalize.py` → `s3_blocking.py` → `s4_match.py` (features, then `decide`) → `s5_cluster.py` → `s6_fusion.py`. `s7_diagnostics.py` computes the diagnostic panel. The rebuild uses only the task inputs plus a cache of general-purpose embeddings (`work/state/emb_cache.pkl`, text-embedding-3-small, inference only). `NO_NETWORK=1` is set, so it makes no network calls. I re-ran it and the sorted outputs were byte-identical to the submission. Earlier coherent revisions are kept in `work/revisions/r1..r7`. r7 is the submitted one.

## Source profile

| source | rows | id | contents |
|---|---|---|---|
| dbpedia | 10,069 | resource URI | name (`nm`), establishment date (`ey`), country (`cn`), headquarters (`hq`), segment (`sg`), key people (`kpn`), assets (`ta`), income/revenue (`ai`) |
| forbes | 2,107 | company URL | name, `url`, region, business segment, assets, sales |
| fullcontact | 2,004 | `fullcontact_N` | name, country, city, people, founding date |

What I observed in the data:

* **Injected copies.**
  * 604 DBpedia and 425 Forbes ids are numeric (`.../resource/9417415`, `.../companies/2714627/`). These are injected records.
  * 28 of the numeric Forbes rows carry a `url` that points to their canonical Forbes page. I treat this as a certain within-source link.
  * The other numeric rows are either noisy duplicates or members of **synthetic entity families**. These families appear in all three sources, e.g. *HarbourLink Energy / Ferries / Rail / Aotearoa*, which are near-name hard negatives.
* **Record noise.** Names, countries and dates carry token shuffles, token deletion, OCR and keyboard substitutions, truncation and abbreviations.
* **Name-swap noise.** The displayed name is sometimes replaced by a related company's name, while the URI or slug keeps the true identity. Examples:
  * Forbes `juniper-networks` is shown as "Unisphere Networks".
  * DBpedia `Brunswick_Boat_Group` is shown as "Brunswick Corporation".
* **Money values.** The same underlying value is written in different ways:
  * units, millions or billions;
  * US, European or space-separated locale formats;
  * free text such as `€1.8 billion` or `USD 280 millio`.
* **Perturbed money values.** About 45% of the high-precision Forbes values are off the 0.1-bn grid that the unperturbed values sit on. Dividing them by **0.7842 or 0.9215** puts them back on that grid.
  * I found these factors by comparing Forbes url-linked duplicate pairs, e.g. Itochu 60.23 vs 76.8 bn and AmerisourceBergen 15,527.16 vs 19,800 m.
  * DBpedia's unperturbed values sit on a 1-m grid.

## 1. Schema matching (`s1_schema.py` → `sm_mapping.csv`)

I mapped the columns by hand, checking meaning, types and examples.

| target | dbpedia | forbes | fullcontact |
|---|---|---|---|
| name | nm | co_nm | Attribute_2 |
| founded | ey | – | Attribute_6 |
| country | cn | region | Attribute_3 |
| city | hq | – | Attribute_4 |
| industry | sg | bus_seg | – |
| assets | ta | ast_val | – |
| revenue | ai | sls_fig | – |
| keypeople | kpn | – | Attribute_5 |

* The DBpedia `ai` → revenue mapping is the least certain (score 0.75). Its magnitudes match Forbes sales, e.g. Brunswick 3.748 bn.
* The Forbes `url` column is used only as a within-source duplicate link.
* The DBpedia URI title and the Forbes URL slug are used as clean identity-name evidence.

## 2. Normalization (`s2_normalize.py`, `country.py`, `industry.py`, `numparse.py`, `textnorm.py`)

* **Country (CLDR, exhaustive).** Values are matched in this order:
  * the taxonomy's own name, short-name and variant columns;
  * about 150 hand-curated aliases (official, native and historical names; England/Scotland → United Kingdom; `中华人民共和国`; Nippon);
  * a sorted-token key;
  * an OCR-confusable key (o/a, c/e, h/b, l/i);
  * anagram, unique-prefix and guarded fuzzy matching.
  * Ambiguous fragments are left null: `United`, `Republic of`, `South`.
  * Canonical rate by source: dbpedia 97.7%, forbes 98.8%, fullcontact 98.2%.
* **Industry (GICS Industry Name, exhaustive).** Values are matched in this order:
  * a manual override dictionary of about 330 semantic judgements, e.g. Regional Banks → Banks, Video game industry → Entertainment, Oil & Gas Operations → Oil Gas & Consumable Fuels;
  * fuzzy and OCR-confusable lookup in that dictionary;
  * embedding nearest neighbour over GICS sub-industry names and the override keys.
  * Every non-empty value gets a GICS name. The long-tail embedding choices are the least reliable part.
* **Founded.** I extract the year: letters are mapped back to digits (l, i, q → 1; o, p → 0; w → 2; …) and a single keyboard-neighbour 9→0 fix is allowed.
  * Two-digit years ≤ 16 become 20yy, otherwise 19yy. These years are down-weighted in fusion.
  * Output is `YYYY-01-01`. All sources only carry `01-01`.
* **City.** I split camel-case concatenations (`IllinoisLake Forest, Illinois`), drop regions, states, countries and street fragments, and keep the first city token.
  * `New York` is kept as written.
  * A corpus-frequency spelling repair then fixes rare city strings (frequency ≤ 1), mapping them to a frequent one (frequency ≥ 5) in three cases:
    * spacing or case variants (`Minneap olis`, `SHENZHEN`);
    * a unique non-initial OCR or keyboard single-character variant (`Modrid` → Madrid, `Plono` → Plano);
    * a unique truncation (`Lake Fores` → Lake Forest).
  * The first version used plain edit distance and rewrote real towns (Hamden → Camden, Tustin → Austin), so I restricted it to the cases above.
* **Money.**
  * Parsing is locale-aware. Text with a currency or unit is taken at nominal value and **not converted** (e.g. `€1.8 billion` → 1.8e9). I have no rates and no evidence of the gold convention.
  * Bare numbers ≥ 1e7 are read as units, < 1000 as billions and anything in between as millions. A DBpedia bare value < 1000 that would exceed 300 bn revenue (1.5 tn assets) is read as millions.
  * Off-grid values are restored when dividing by 0.7842 or 0.9215 lands on the grid within rendering precision. The grid is 0.1 bn for Forbes and 1 m for DBpedia.
  * All alternatives are kept as weighted candidates for fusion.
* **Key people.** Lists are split on list syntax, `;`, `and` and commas. I strip roles (fuzzy, OCR-tolerant: `Cbief Executive Officer`), fix digit-in-word OCR and drop generic phrases such as "Executive leadership team" or "Boord of Dircctors".
  * Founder organisations are kept, e.g. `Government of Victoria`.
* **Names.**
  * Display names: legal-form suffixes and type disambiguators such as `(company)` or `(automobile)` are removed. The schema asks for names without legal suffixes.
  * Matching keys: ASCII-folded, OCR-repaired, abbreviations expanded (Int'l, Mgmt, Sys., Hldgs…), with a "core" variant that drops generic words.

## 3. Blocking (`s3_blocking.py` → `blocking/candidates.csv`)

Three complementary candidate sources are combined:

* rare-token blocks on name keys (tokens in more than 40 records are skipped; 47 such tokens);
* embedding kNN (k = 12, cosine ≥ 0.5) over the primary and raw names;
* exact core-key blocks, plus the Forbes url links.

Within-source pairs are included because the sources contain injected duplicates.

| measure | value |
|---|---|
| candidate pairs | 120,658 |
| reduction ratio | 0.9988 |
| dbpedia–forbes | 22.9k |
| dbpedia–fullcontact | 21.1k |
| forbes–fullcontact | 6.4k |
| records with no candidate | 689 (inspected: unique names) |
| max candidates per record | 154 |
| correspondences inside the candidate set | 100% |

The export is the complete set. Any cluster-closure pair not already a candidate would be added; there were 0.

## 4. Entity matching (`features.py`, `s4_match.py`)

**Name evidence.** An IDF-weighted soft token coverage, requiring *both* names to be covered. Tokens are compared with an OCR/typo tolerance of 0 / 1 / 2 OSA edits by length. A joined-string rule handles concatenations.

* Authoritative names are the DBpedia URI title and the Forbes slug.
* A raw displayed name that is inconsistent with its own title or slug gets weight 0.7. This neutralises name-swap noise such as McCafé→"McDonalds" and Unisphere↔Juniper.
* If both full names carry *different* descriptor words ("Crown International" vs "Crown Holdings"), an exact core match is demoted.

**Attribute evidence.** Country, founding year (±1), city, industry, assets, revenue (within 2%, modulo the perturbation factors and ×1000 scale), key-people overlap and embedding cosine. A missing value counts as unknown, never as agreement.

**Decision rules (deterministic).**

| rule | accepts when | notes |
|---|---|---|
| distinct native ids | never, for native dbpedia–dbpedia or forbes–forbes pairs | distinct URIs are distinct entities |
| url link | always | |
| exact name (≥ 0.97) | informative name, or a Forbes pair with country not conflicting | rejected on country conflict plus attribute conflict, on native–native country conflict with no support, and for generic-word-only names |
| generic / disambiguated exact | country agrees and at least 1 supporting attribute | applies to generic names and to DBpedia titles with a non-type disambiguator such as `(Russia)` or `(original)`. This is how `Deere (automobile)` is kept apart from Deere & Co |
| close name 0.85–0.97 | country compatible, at least 1 support, no conflicts | |
| contained name | one name fully covered, at most 1 non-generic extra token, embedding ≥ 0.75, no conflicts | |
| truncated name | the token-subset side is an injected copy or fullcontact; country and at least 2 supports | native short titles denote the parent, so they are excluded |
| money match | assets **and** revenue agree, country agrees, name ≥ 0.5, embedding ≥ 0.75 | |

I set the thresholds by inspecting the pairs around each band; the listings were printed with `work/show_pairs.py`. I did not fit anything.

Accepted edges by rule:

| rule | edges |
|---|---|
| exact name | 2039 |
| contained | 83 |
| url link | 28 |
| truncated | 28 |
| generic + support | 13 |
| money | 8 |
| close / partial | 10 |

## 5. Clustering and refinement (`s5_cluster.py`)

I take connected components of the accepted edges and apply two repairs:

* **Identity constraint.** A cluster may not contain two native DBpedia or two native Forbes records. When it does, the two are separated by a **minimum-weight edge cut**, with edge score as capacity. 17 edges were removed; all are logged in `work/state/rejected_edges.csv`, e.g. Macy's vs Macy's Inc., Metro Inc. vs Metro Group, Edison (company) vs Edison International.
* **Conflict detach.** In clusters of 3 or more, a member that contradicts at least 2 other members on at least 2 of {year, city, people} is detached. This happened twice, e.g. "Vector Loom Interactive, San Jose 1994" vs the "Vector Loom, Palo Alto 2008" family.

Resulting structure:

| measure | value |
|---|---|
| clusters | 12,430 |
| singleton share | 0.894 |
| max cluster size | 9 (nine fullcontact `HugeDomains.com` rows) |
| clusters by number of sources (1 / 2 / 3) | 11,300 / 747 / 383 |
| clusters with several records from one source | 228 |
| share of records in multi-member clusters | native Forbes 27.2%, native DBpedia 8.1%, injected DBpedia copies 76.0%, injected Forbes copies 77.6%, fullcontact 52.6% |

## 6. Fusion (`s6_fusion.py` → `fused.csv`)

Values are fused attribute by attribute from each cluster's own members.

* **name** — a consensus medoid over candidate names, with priorities: DBpedia title > Forbes name > others. OCR-noisy candidates are penalised and the most complete variant gets a bonus. The Forbes slug is title-cased and used only when the raw Forbes name is missing, truncated, an acronym of the slug or unrelated to it.
* **country, city, industry** — majority vote with source priority as tie-break (country and industry: Forbes > DBpedia > fullcontact; city: DBpedia > fullcontact). Native records rank ahead of injected ones.
* **founded** — year vote; four-digit years rank above two-digit ones.
* **assets, revenue** — weighted candidate groups within 2%. Each member counts at most once per group. Weights are tier × source: grid 1.0, restored 0.95, text 0.8, raw off-grid 0.4–0.6; Forbes 1.0, DBpedia 0.7. Values are written as integers.
* **keypeople** — the member list that best agrees with the other lists, then fewer OCR tokens, then more complete names. Written as a JSON list.

`_id` equals the membership `cluster_id`. `id` is the representative native source id.

Density (non-empty share):

| attribute | density |
|---|---|
| name | 0.997 (39 clusters have no name in any member) |
| founded | 0.755 |
| country | 0.865 |
| city | 0.686 |
| industry | 0.455 |
| assets | 0.161 |
| revenue | 0.193 |
| keypeople | 0.048 |

Validity: country and industry are 100% in their taxonomies; founded matches the pattern 100% and is inside 1700–2016 for 99.75%; assets and revenue are integers in range 100%. Missing values were never filled by guessing.

## Label-free checks performed

* Every submitted id resolves to a source id.
* Every source record appears exactly once in `membership.csv`.
* The `fused._id` set equals the `membership.cluster_id` set.
* Every correspondence is cross-source, lies inside one final cluster and is in the candidate set.
* Sampled multi-member clusters were traced from fused values back to the members' raw values.
* `scripts/self_check.py` passes.

## Known weaknesses and open uncertainty

* **Parent vs subsidiary vs brand.** This depends on names and on the few shared attributes. Forbes–fullcontact pairs share only name and country, so exact brand names are accepted. Examples: `Woolworths` vs `Woolworths Limited`, and `Northern Rock (Asset Management)`.
* **fullcontact countries.** These are often regional offices (Ericsson India, Accenture San Diego). I still matched exact-name Forbes records, treating the company as one entity; this may disagree with a stricter reading.
* **Money.**
  * The million-or-billion reading of bare DBpedia numbers is uncertain.
  * Perturbation restoration can pick the wrong factor when both fit (United Continental revenue: 45.0 vs 38.3 bn).
  * Currency-labelled values are not converted to USD.
* **Industry.** The GICS mapping is a judgement call: e.g. Retail → Broadline Retail, Manufacturing → Machinery, and the long-tail embedding choices.
* **City.** Granularity is uncertain: district vs city (Minato vs Tokyo), and "New York" vs "New York City".
* **Key people.** The target asks for founders. Sources mostly give founders (DBpedia) or executives (the synthetic records); I output what the members state.
* **Missed matches.** Heavily corrupted names (token deletion down to a single generic word, keyboard garbling) stay singletons when no other evidence links them.

## Approaches considered and rejected

* **Plain token Jaro–Winkler.** It inflated similarity between unrelated names (RAE/BAE Systems, Maxi/Maxis).
* **Weakest-edge removal for constraint violations.** Replaced by the minimum cut.
* **Conflict detach on a single conflicting pair.** Real-world discrepancies (Coloplast 1954 vs 1957) triggered it.
* **Trusting raw names of native records.** Name-swap noise caused false merges.
* **Any supervised or pseudo-labelled model.** Not permitted in this condition.
