# Companies integration — method report (revision r2)

**No labeled evaluation and no supervised training were done.** There is no reference mapping, match set or fusion gold,
so this report gives no precision, recall, F1, pair completeness or fusion accuracy. All numbers below are structural,
label-free diagnostics computed from the saved artifacts by `work/diagnostics.py`, with the log in `work/diagnostics.jsonl`.
Pair decisions come from deterministic rules that I wrote after looking at examples. Those examples were not used as labels.

Pipeline: `work/s1_schema.py` → `s2_normalize.py` → `s3_blocking.py` → `s4_features.py` → `s4_match.py` → `s5_cluster.py`
→ `s6_fusion.py` → `s7_export.py` → `diagnostics.py`. `work/rebuild.sh` runs all of them. I ran it in a fresh copy of the
workspace with no `submission/` directory. All five data files matched the submitted ones exactly (row sets compared).
Revision snapshots are in `work/revisions/r1` and `work/revisions/r2` (final).

## 1. Schema matching (`sm_mapping.csv`)
Columns were mapped by meaning, using the metadata and example values:
- dbpedia: entity_uri→id, org_name→name, established→founded, nation→country, headquarters→city (city parsed out of
  a concatenated location string, score 0.8), sector→industry (free text mapped by keyword rules, 0.8),
  keypeople_name→keypeople (the metadata calls them founders), total_assets_val→assets and annual_income→revenue
  (no unit given; score 0.7–0.8).
- forbes: forbes_url→id, company→name, region→country, business_segment→industry, asset_value→assets, sales_figure→revenue.
  `url` is identical to `forbes_url` in 100% of rows, so it is left unmapped.
- fullcontact: the anonymised Attribute_1..6 columns were identified from their values as id, name, country, city,
  keypeople (list strings) and founded (ISO dates).
Every target attribute has at least one contributing source. Forbes has no city, founded or keypeople. FullContact has no
industry, assets or revenue.

## 2. Normalization (`work/state/norm.pkl`; raw values are kept next to the canonical ones)
- **Country:** mapped to the CLDR `Country Name` using the canonical, short and variant columns, plus documented extra aliases
  (England/Scotland/Wales→United Kingdom, "United States of America", "Korea (Republic of)", "Taiwan, Province of China[a]",
  historical states, etc.; see `work/taxonomy_plan.json`). Values that cannot be mapped become null. Canonical rate:
  dbpedia 0.999, forbes 1.0, fullcontact 0.999 (`work/taxonomy_coverage.csv`).
- **Industry:** the taxonomy is exhaustive, so values are GICS `Industry Name`. Forbes segments (84 values) use a manual
  semantic table. DBpedia free-text sectors (1,125 distinct values) use ordered keyword rules, which cover 0.87 of
  non-null values. The rest become null rather than being forced into a category.
- **City:** DBpedia `headquarters` values are concatenated with no separator (e.g. "MilwaukeeWisconsin",
  "GuangdongShenzhen"). The parser splits on lowercase→uppercase boundaries and commas, then drops region/country segments.
  A comma suffix counts as a region only if it appears more often as a suffix than on its own. Parsed rate is 0.98.
  FullContact city is used as given.
- **Founded:** year only (every source date is YYYY-01-01). Years outside the schema's consistency range 1700–2016 are
  nulled. Some parse as 1 or 4555. Real pre-1700 years are also lost this way; this is a known limitation.
- **Money:** Forbes is already in US$. DBpedia values carry no unit. On 200 exact-name DBpedia–Forbes pairs, values below
  1e3 sat about 9 orders of magnitude below Forbes (billions), and values between 1e3 and 1e6 sat about 6 orders below
  (millions). DBpedia values are therefore scaled ×1e9 below 1e3 and ×1e6 between 1e3 and 1e6. Some DBpedia values are
  in local currency (e.g. JPY). Values above the schema maxima are treated as unknown at fusion.
- **Keypeople:** Python-list strings are parsed into JSON lists.
- **Names:** comparison keys strip accents, punctuation, parentheses and trailing legal forms (Inc, Corp, Ltd, plc, AG,
  SA, SAB de CV, …). For mojibake DBpedia names (�) the percent-decoded URI slug is used. Alias keys come from
  parenthetical content and " - " segments in FullContact and Forbes names (e.g. "Pacific Gas and Electric (PG&E)",
  "Telenor - Internet for all"). A segment is not used as an alias if the dropped part is a region or it reads
  "a X company" or "division". DBpedia parentheses are Wikipedia disambiguators, not aliases.

## 3. Blocking (`blocking/candidates.csv`, complete, 65,339 pairs)
The candidate set is the union of:
- exact keys (name key, core key without generic words, URI/URL-slug key);
- TF-IDF char 2–4-gram cosine kNN (k=10, cos ≥ 0.3), both directions for each source pair;
- acronym keys (e.g. DSME ↔ Daewoo Shipbuilding & Marine Engineering);
- alias keys.

Reduction ratio: dbpedia/forbes 0.9986, dbpedia/fullcontact 0.9986, forbes/fullcontact 0.9978. Records with no
candidate: dbpedia 2,665, forbes 62, fullcontact 46. For dbpedia this mostly means no other source has a similar name.
Candidate coverage is not pair completeness, because no true matches are known.

## 4. Entity matching (`work/s4_match.py`, `work/state/scored_pairs.csv`)
Edit similarity was rejected after inspection: it pairs "X Technologies" with "Y Technologies", Bank of China with
Bank of India, and so on. Instead, each pair gets a **name tier** from a strict, IDF-aware token comparison, and
**contextual evidence** is added to it:
- Name tiers:
  - exact or differing only by generic words (group, holdings, inc, …): 1.0 / 0.95;
  - alias exact: 0.85;
  - one side has an extra word that restates the company's own industry, e.g. "Comerica Bank", "Allstate Insurance"
    (industry words come from GICS/Forbes/DBpedia labels plus a synonym list): 0.8;
    lowered to 0.7 for DBpedia–Forbes, because Wikipedia has separate articles for subsidiaries, and when the shorter
    name is a "Group/Corporation/Holdings" (parent vs listed affiliate, e.g. Mitsubishi Corporation vs Mitsubishi Motors);
  - common descriptor word: 0.7; plural or one-edit typo in a long token: 0.75 / 0.65;
  - other distinctive extra word: 0.6;
  - extra geographic word (regional arm, e.g. "E.ON Russia", "TDK - Americas"): 0.4, unless it just names the shared country.
- Evidence added to the tier:
  - country: +0.15 if equal, −0.25 if different (−0.1 for exact, rare names; China vs Hong Kong is neutral);
  - city: +0.1 / −0.05;
  - founding year: equal +0.15, within 2 +0.05, more than 10 apart −0.1;
  - revenue or assets: log-ratio < 0.05 gives +0.2, < 0.15 gives +0.1, > 1 gives −0.1;
  - a DBpedia disambiguator naming a different industry or country than the other record ("Deere (automobile)" vs
    Deere & Co, "Safeway (Canada)" vs Safeway US): −0.3.
- Missing values count as unknown and add nothing.
- Threshold: 0.9. An exact name therefore needs no contradiction. A descriptor or industry variant needs country plus one
  more agreeing signal. A distinctive extra token needs two or three signals.
- Pairs just above and below the threshold were reviewed by tier (listed in `work/step_log.jsonl`). The rules were then
  revised for these documented failure patterns: conglomerate families, disambiguators, loose typos, and DBpedia
  parentheses being treated as aliases.

## 5. Clustering (`work/s5_cluster.py`)
- Greedy 1:1 assignment per source pair, taking the highest score first, with deterministic tie-breaks
  (47 edges were rejected by one-to-one).
- FullContact contains exact duplicate profiles, e.g. Google Inc ×3 and HugeDomains.com ×10 (parked-domain profiles).
  These are merged only when the names are identical and country, city and founded do not conflict.
  No within-DBpedia or within-Forbes merges are made.
- Union-find then runs with a constraint of at most one DBpedia and one Forbes record per cluster. The weakest edge is
  dropped until the constraint holds (4 edges dropped).
- Rejected edges are in `work/state/rejected_edges.csv`. Correspondences are all cross-source pairs inside the final
  clusters.

## 6. Fusion (`work/s6_fusion.py`, provenance in `work/state/fusion_provenance.csv`)
A value wins by vote only if at least two distinct sources agree after normalization. Otherwise a fixed source priority
decides:
- name: Forbes > DBpedia (disambiguator removed) > FullContact, with legal suffix removed;
- country: Forbes > DBpedia > FullContact;
- city: FullContact (clean field) > DBpedia (parsed);
- industry: Forbes table > DBpedia rules;
- assets and revenue: Forbes (US$) > DBpedia (scaled, range-checked);
- founded: DBpedia > FullContact;
- keypeople: DBpedia founders > FullContact key persons.

No value is invented. Missing stays missing.

## 7. Label-free diagnostics (final r2, from saved submission files)
- Every record of every source appears exactly once in membership. There are 0 unresolved ids. 100% of correspondences
  are in the candidate set and in the same cluster. Fused `_id` equals the membership cluster ids.
  100% of provenance record ids belong to their own cluster.
- 12,472 clusters. Sizes: 11,114 of size 1, 1,184 of size 2, 168 of size 3, 5 of size 4, and one of size 10
  (the HugeDomains duplicate profiles). Sources per cluster: 1 → 11,122; 2 → 1,191; 3 → 159. Singleton share 0.891.
- Share of records in multi-source clusters: dbpedia 0.088, forbes 0.434, fullcontact 0.583. 1,688 correspondences.
- Accepted edges by tier: exact/generic 1,626; extra_own_industry 32; alias 19; descriptor 20; extra_token 16; typo 4;
  own-country geo word 4; acronym 3; partial 1.
- Density: name 1.0, country 0.975, founded 0.857, city 0.810, industry 0.609, revenue 0.306, assets 0.201,
  keypeople 0.088.
- Validity: country ∈ CLDR 1.0, industry ∈ GICS 1.0, founded pattern/range 1.0, money in schema range 1.0,
  keypeople valid JSON list 1.0.
- Share of fused cells where members held more than one distinct value: assets 5%, revenue 6%, everything else ≤ 2%.

## 8. Known weaknesses and uncertainty
- Forbes–FullContact pairs carry only name and country evidence, so correct matches with a subsidiary-sounding extra
  word are rejected. Examples are "Boeing Store" and "SoftBank Capital". These may be FullContact profiles returned for
  the listed company, but under the entity definition they are business units, so I did not merge them.
- Families of separately listed affiliates (Mitsubishi *, China Resources *, Ping An *) and short or ambiguous names
  remain error-prone. When two Forbes affiliates tie against one FullContact record, the 1:1 tie-break is arbitrary
  but deterministic.
- DBpedia money units and currencies are unreliable. The scaling rule is inferred from the data, and values of 1e6 or
  more are kept raw even when they are really in millions. Revenue and assets for DBpedia-only entities are therefore
  uncertain. The 2016 target period is not represented by any source (Forbes covers 2013/14).
- City choice between FullContact (sometimes a ward or office location, e.g. "Minato-ku") and DBpedia (parsed) is a
  priority heuristic. The DBpedia city parser can pick a district or county.
- DBpedia industry comes from keyword rules on a free-text sector, so it is approximate. Founders from FullContact may
  include non-founder key persons.
- Rejected approaches: edit-distance or Jaro-Winkler thresholds (too many X/Y Technologies merges); DBpedia
  parentheses as aliases; the first typo rule (Broadjam/Broadcom, Bungie/Bunge); a blanket bonus for Forbes–FullContact
  pairs (it would merge subsidiaries).
