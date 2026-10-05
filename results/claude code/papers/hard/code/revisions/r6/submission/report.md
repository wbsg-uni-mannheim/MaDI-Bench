# Papers integration: method report (no-label condition)

**No labeled evaluation or supervised training was done.** No matcher was trained or fine-tuned, and no pseudo-labels were created. No embedding service or external lookup was used. Every threshold below comes from reading the source records directly and choosing a rule. The numbers in this report are label-free structural diagnostics, not precision, recall, F1, pair completeness or fusion accuracy. The pipeline is deterministic and `work/rebuild.sh` regenerates every submission file. A fresh-copy run was verified to reproduce the files exactly (row order and float scores aside).

Pipeline: `work/s1_schema.py`, then `s2_normalize.py`, `s3_blocking.py`, `s4_features.py`, `s4_match.py`, `s5_cluster.py`, `s6_fusion.py` and `diagnose.py`. Intermediate files are in `work/state/`, revisions in `work/revisions/`, and logs in `work/step_log.jsonl` and `work/diagnostics.jsonl`.

## 1. Schema matching (`sm_mapping.csv`)
All three sources carry the 11 target attributes plus `id`. The mapping was chosen by meaning and checked against examples, not by column name:

- **crossref:** typ, ttl, auth_list, pub_year, venue, vol, iss_no, pg_first, pg_last, ref_count, cited_count.
- **dblp:** ty, t, au, py, j, v, i, fp, lp, rc, cc.
- **open_alex:** Attribute_3 (type), 4 (title), 5 (authors), 6 (year), 7 (journal), 10 (volume), 11 (issue), 12 (first page), 13 (last page), 14 (referenced works), 15 (cited by).

Unmapped columns are kept in the translated table for audit:
- publisher: crossref `publ`, dblp `pb`, open_alex `Attribute_8`
- abstract: crossref only
- keywords: crossref `kw_list`, dblp `kw`, open_alex `Attribute_9`

In dblp, pages and counts are about 99% empty.

## 2. Normalization (`work/state/s2_normalized.pkl`; raw values kept as `raw_*`)
The data is heavily and deliberately corrupted:
- OCR substitutions (O→0, l→1, B→8, S→5, G→6)
- keyboard-neighbour substitutions (`wp28`, `qrtifpe`)
- truncations (`201`, `arti`)
- thousand separators and stray spaces in numbers (`1 864`, `3,907`, `0,0`, `49.O`)
- HTML entities and markup
- author lists in at least six formats, including broken quoting, `A, B, and C`, localized conjunctions (`e`, `y`, `und`, `i`), semicolons, crossref affiliations mixed into author lists, and dblp homonym suffixes (`0001`, sometimes noised)
- abbreviated or typo'd venues
- title word drops, shuffles, synonym swaps and `Syst.`-style abbreviations
- generator placeholders (`UNCHANGED`, `<AUNCHANGED>`, `<Same as input>`)

The same functions are applied to every source:

| Attribute | Normalization |
|---|---|
| type | Closed enum from the target schema, plus aliases, prefix truncation, fuzzy ratio ≥ 60, and a length-preserving keyboard-adjacency match (≥ 70% of positions) for keyboard-noised values. 100% of non-empty values map to the enum (`work/taxonomy_coverage.csv`, `work/taxonomy_plan.json`). |
| year | OCR repair, then minimum-substitution decoding over {2018..2021} using the keyboard/OCR confusion model. Ambiguous results stay missing. Canonical rate is about 98.5% of non-empty values per source; truncations (`20`, `201`) are left unresolved. |
| volume, issue, pages | Digit-group joining, OCR repair and `.0` removal. Genuine locators (`e1007004`, `D442`, `W1`, `Suppl 1`, `CSCW`, `1-2`) are kept. Values with keyboard-noise letters (`qe`, `w2`) are flagged noisy and not used. About 95.5–98% canonical. |
| counts | Separator, decimal and OCR repair. Keyboard-noised values are left missing. |
| authors | Parsed to lists. Affiliations and countries are dropped; dblp suffixes are removed only for comparison keys. Name keys are accent-folded and lowercased. |
| venue | Keys are folded and lowercased with `&`→`and`. Corpus-level spellings are unified within the abbreviated and full forms separately, e.g. `Meural computing anr applicatione` → `Neural Computing and Applications` (see below). |

Venue canonicalization rule: casing variants are mapped to the most frequent spelling. A rare key (≤ 3 records) is mapped to a key at least 10× more frequent only if the two are character near-duplicates (normalized Levenshtein ≥ 0.8), every token aligns, and they differ by at most one token. 4.5k records changed; a sample was inspected and a few errors are possible (e.g. `journal of information` → `journal of informetrics`).

## 3. Blocking (`blocking/candidates.csv`, complete)
The full cross product is about 6.2e9 cross-source pairs. Blocking is the union of three kNN methods over sparse TF-IDF cosine, including same-source neighbours (needed for duplicate detection):

1. **Title words:** max_df 0.02, k = 12, cosine ≥ 0.25.
2. **Title character 4-grams:** robust to typos and merged words; k = 8, cosine ≥ 0.30.
3. **Author surnames and initial+surname tokens:** catches truncated or garbled titles; k = 10, cosine ≥ 0.30.

Results:
- 1,369,255 cross-source candidate pairs (crossref–dblp 493k, crossref–open_alex 378k, dblp–open_alex 498k); reduction ratio 0.99978.
- Pairs unique to each method: title 717k, char 453k, author 503k. The methods are complementary.
- 548 records (0.4%) have no cross-source candidate, mostly records with neither a title nor authors.
- Maximum cross degree is 452 (generic titles).
- The in-cluster pairs implied by clustering (below) are added to the exported file, so all correspondences are candidates (checked: 100%).

This is candidate coverage, not pair completeness.

## 4. Entity matching (`s4_features.py`, `s4_match.py`): direct rules, no training
Evidence computed per pair:

| Feature | Definition |
|---|---|
| title similarity | Soft 1:1 token overlap. Tokens match if exact, by prefix (≥ 3 chars, for abbreviations and truncation) or at Levenshtein ≥ 0.75. Measured as a share of the longer title (`t_max`) and of the shorter title with stopwords removed (`t_min`, detects dropped words and truncation). |
| exact title | Title keys are identical. |
| colon prefix | One title equals the other's part before `:`, `?` or ` - ` (crossref keeps only the main title). |
| author agreement | Strict name alignment (Jaro-Winkler surname plus initial, with order swaps), combined with a bag of name tokens that is robust to first and last names shuffled between authors. |
| venue | Abbreviation-aware venue similarity. |
| year and locators | Year difference and equality of clean volume, issue, first and last page. |

Missing values count as unknown, not as agreement.

Hard contradictions block a match:
- two differing locators among volume, issue and pages (first and last page, volume and first page, or volume and issue)
- differing volume while the venues also disagree
- year difference ≥ 3, or exactly 2 unless title and authors agree strongly (OpenAlex often reports the online-first year)
- differing numbers or roman numerals in the titles (`WMT 2018/2019`, `type II/III`, `Part I/II`)
- one title is a comment, reply, erratum, retraction or demo title and the other is not
- dblp/crossref type article vs inproceedings together with a year difference (conference vs journal version, which the entity definition treats as separate)

Accept rules:

| Rule | Condition |
|---|---|
| r1 | Long title with t_max ≥ 0.85 and no hard conflict. Authors agree ≥ 0.34, are missing, or (if lower) year matches and ≥ 2 corroborating locators. |
| r2 | 0.6 ≤ t_max < 0.85 and authors ≥ 0.75, plus either full containment of one title in the other, or t_max ≥ 0.75 with ≥ 1 corroboration. |
| r3 | Exact or colon-prefix title for short titles, with authors ≥ 0.75 or ≥ 2 corroborations. |
| r4 | Title < 0.6 but authors ≥ 0.9 and first page plus last page or volume agree. |
| r5 | Long title fully contained in the other, authors ≥ 0.9 and ≥ 1 corroboration. |
| r6 | Single-author papers with identical long titles plus corroboration. |

Same-source pairs (injected noisy duplicates, mostly ids ≥ 60000) additionally need:
- authors ≥ 0.85 (or missing with an identical title and year)
- near-identical or containing titles
- same year and same type

How the thresholds were chosen: the tables and samples of pairs around each boundary were inspected. Tightening r2 removed "same authors, different paper" pairs, e.g. "Autonomic resource provisioning…" vs "An autonomic risk- and penalty-aware…". The number and meta-title conflicts removed series and comment false merges.

Accepted pairs: 62,896 in total, of which 61.5k are cross-source (r1 51.9k, r3 8.4k, r2 2.0k, r5 320, r4 227, r6 9).

## 5. Clustering (`s5_cluster.py`)
Accepted edges are merged greedily in descending score order. A merge is refused if it would put two records of the same source together without a direct accepted edge between them. This stops weak bridges from joining a conference paper and its journal version: 342 bridging edges were refused. Inspected examples include "Progressive disclosure" (CHI vs TiiS) and "SHEILA policy framework" (LAK vs JLA). Unmatched records remain singletons.

Final structure:
- 82,547 entities
- cluster sizes: 36,741 of size 1, 37,449 of size 2, 8,195 of size 3, 158 of size 4, 4 of size 5
- sources per cluster: 37,246 with one source, 37,627 with two, 7,674 with three
- singleton share 0.445
- 1,346 clusters containing same-source duplicates

`correspondences.csv` contains every cross-source pair within a final cluster: 61,667 pairs, 468 of them implied transitively. There are no pairs across clusters.

## 6. Fusion (`s6_fusion.py`)
Each attribute is resolved from the cluster's own normalized member values. Ties are broken by a fixed source order.

| Attribute | Resolution |
|---|---|
| type | Majority vote; tie order dblp > crossref > open_alex (open_alex labels conference papers as article). |
| title | Majority on the normalized key, then more non-abbreviated tokens (defeats dropped words and truncation), then corpus bigram plausibility (defeats word shuffles), then open_alex > crossref > dblp. dblp's trailing period is removed only when another source is in the cluster; dblp-only entities keep it. |
| authors | The member list agreeing most with the others, then corpus plausibility of full names, then source order. Each name is then corrected by per-person consensus: strict alignment, majority, then corpus frequency. dblp homonym suffixes are stripped in multi-source clusters and kept for dblp-only entities. |
| publication_year | Vote over decoded years in 2018–2020; tie order crossref > dblp > open_alex. crossref and dblp agree 99%; open_alex differs in about 16% of matched pairs. |
| journal | The corpus-canonical full name supported by the most members, where a dblp abbreviation counts as support for the full name it abbreviates. Otherwise the abbreviation is used. |
| volume, issue, pages | Vote over clean values; noisy values are never output. A last page smaller than the first page is dropped. |
| counts | open_alex > crossref > dblp. These are snapshot values: crossref and open_alex reference counts are exactly equal in only 14% of matched pairs, so voting would be meaningless. |

## Diagnostics (final revision; label-free)
- Every one of the 136,876 source records appears exactly once in membership, with no unknown ids.
- fused `_id` equals the membership cluster ids.
- Schema violations: 0 for every attribute. Rows with last page < first page: 0.
- Traceability: in a random sample of 3,000 fused rows, every title, volume and first page equals a member's normalized value. 119 journal values differ from member values only because of the documented corpus canonicalization.
- Fused density:

  | Attribute | Density |
  |---|---|
  | type | 0.95 |
  | title | 0.95 |
  | authors | 0.93 |
  | publication_year | 0.94 |
  | journal | 0.67 |
  | volume | 0.59 |
  | issue | 0.29 |
  | first_page | 0.64 |
  | last_page | 0.64 |
  | referenced_works_count | 0.77 |
  | cited_by_count | 0.77 |

- Share of multi-member clusters whose members disagree:

  | Attribute | Share |
  |---|---|
  | year | 8.9% |
  | volume | 2.3% |
  | first page | 0.13% |
  | type | 14.5% (mostly open_alex "article" vs "inproceedings") |

## Known weaknesses and uncertainty
- The source whose values the gold fused record uses is unknown. The following choices are reasoned guesses, not measured:
  - source order for type, year and counts
  - keeping dblp's trailing period and homonym suffixes for dblp-only entities
  - full venue names over abbreviations
- Noised values with no clean alternative are left empty rather than guessed. This covers keyboard-noised pages, volumes and counts, and truncated years.
- Hard cases remain:
  - records with neither title nor authors
  - same-author, similar-title papers that differ only by a synonym swap
  - injected duplicates whose authors are heavily shuffled
  - generic short titles ("Poster", "Experience") with missing authors
- A few same-title, different-paper pairs (e.g. several "Relation Understanding in Videos" challenge papers) are separated only by author disagreement.
- Venue canonicalization and per-name author consensus can occasionally replace a correct rare spelling with a frequent near-duplicate.
- Rejected approaches:
  - unconstrained connected components, which chained conference and journal versions
  - containment at t_min ≥ 0.9, which merged "linear" vs "nonlinear" titles
  - a 'poster' meta-prefix rule, which would split legitimate "Poster: X" / "X" pairs
  - a supervised matcher (not permitted)
