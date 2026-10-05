# Papers integration — method report (revision r3)

**No labeled evaluation and no supervised training were done.** I had no reference matches, no fusion gold and no labeling service, and I did not create pseudo-labels. Every number below is a structural, label-free diagnostic. None of them is an accuracy, precision, recall, pair-completeness or F1 figure.

Pipeline: `work/s1_schema.py` … `work/s7_diagnostics.py`, run by `work/rebuild.sh`. It is deterministic and uses no network or models. I rebuilt it in a fresh workspace copy without `submission/`, and all five data files came out identical.

## 1. Schema matching (`sm_mapping.csv`)
Each source column maps one-to-one onto a target attribute by meaning. I checked each mapping against the metadata descriptions and example values.
- **crossref:** work_type→type, title_text→title, contributor_names→authors, issued_year→publication_year, container_title→journal, volume_id/issue_id/page_first/page_last, reference_total→referenced_works_count, cited_total→cited_by_count.
- **dblp:** entry_type, publication_title, author_list, pub_year, venue_name (abbreviated venue names, so score 0.8), volume_no, issue_no, page_start, page_finish. DBLP has no reference or citation counts.
- **open_alex:** work_kind, display_title, authors_list, year_published, source_name, volume_tag, issue_tag, start_page, end_page, refs_count, citations_count.
- **Unmapped:** crossref publisher_name and abstract_text, open_alex topic_terms. None of these has a target attribute.

## 2. Normalization (`s2_normalize.py`)
Raw values are kept as `raw_*` columns. The following was applied the same way to every source:
- The string "None" is treated as null.
- HTML entities are unescaped (for example `&amp;`) and markup tags are stripped.
- DBLP's terminal period on titles is removed.
- DBLP homonym suffixes are removed ("Bing Li 0005" → "Bing Li").
- Stray whitespace in Crossref author names is trimmed.
- DBLP float pages are converted to integers ("1.0" → "1").
- The type vocabulary is closed (the target enum). Coverage was 100% with no unmapped values (`work/taxonomy_coverage.csv`, `work/taxonomy_plan.json`).

Comparison-only copies:
- a title key: NFKD-folded, lowercased, alphanumeric characters only;
- a title token list;
- author name token sets;
- a venue token list.

For title comparison I strip "Retracted on <date>:" and "RETRACTED ARTICLE:" prefixes (341 Crossref and 254 OpenAlex titles). The stored title is unchanged.

Observed source defects:
- About 200 DBLP titles are empty or truncated where the original title contained markup (for example "R", "The").
- Crossref often drops the subtitle after a colon.
- DBLP page ranges are article-local: the first page is always 1.

## 3. Blocking (`blocking/candidates.csv`, complete set)
The candidate set is the union of five methods, each run across all three source pairs:
1. Exact title key (keys of at least 12 characters).
2. TF-IDF word-cosine top-5 in both directions, with similarity ≥ 0.3.
3. First author surname + last author surname + year.
4. Full sorted author-surname set (records with ≥ 3 authors).
5. First author surname + first two title tokens.

For the key-based methods (1, 3, 4, 5), a block is skipped if any source contributes more than 30 records to it.

Result:
- 1,182,375 candidate pairs, roughly 390k per source pair.
- Reduction ratio 0.99989 against the 11.05 billion possible cross-source pairs.
- Records with no candidate at all: Crossref 168, DBLP 54, OpenAlex 24.
- Maximum candidates for one record: 441.
- 99.99% of submitted correspondences are in the candidate set. The rest are pairs implied transitively inside a cluster.

## 4. Entity matching (`s4_features.py`, `s4_match.py`)
Evidence computed for each candidate pair:
- title: edit ratio, Jaccard and containment on the title key, and whether one key is a prefix of the other;
- authors: order-free overlap, where two names match if they share a name token (handles "Miao Yang" vs "Yang Miao");
- year difference;
- equality of volume, issue, first page and last page;
- venue abbreviation similarity.

Missing values count as unknown, never as agreement.

A pair is accepted if it passes at least one rule and triggers no hard contradiction.

Accept rules:
- **A:** title ratio ≥ 0.95, ≥ 3 title tokens, and the authors support it. Support means author overlap ≥ 0.5, or one author list contains the other, or authors are missing, or some author overlap with the same year, or volume/page agreement.
- **B:** short titles (1–2 tokens): title ratio ≥ 0.95, author overlap ≥ 0.5 and year difference ≤ 1.
- **C:** title ratio 0.8–0.95, author overlap ≥ 0.8, year difference ≤ 1, and either title containment ≥ 0.9 or volume/page agreement.
- **D:** one title key is a prefix of the other (a truncated subtitle), author overlap ≥ 0.8 and year difference ≤ 1.
- **E:** a damaged or empty title that is a prefix of the other, with author overlap ≥ 0.9 and at least 2 authors (or one exactly matching author plus a matching first page).

Hard contradictions (rejected even when a rule passes):
- Volume numbers differ.
- Both first and last page differ, compared only between non-DBLP numeric pages.
- The numbers in the titles differ ("Part 1" vs "Part 2", "WMT 2018" vs "WMT 2019").
- DBLP says "article" while Crossref says "inproceedings". Inspection showed these are conference-vs-journal versions, which count as separate entities under the brief.
- Year difference ≥ 3, or a year difference of 2 without volume or page agreement.

I inspected the thresholds on stratified samples. I saw many online-first year gaps of 1–2 years with identical volume and pages, which is why year differences are tolerated when the volume or pages agree. I also saw that Crossref "article" vs DBLP "inproceedings" is systematic for AAAI and Procedia papers, so that direction is **not** treated as a conflict.

Result: 156,721 accepted edges:

| Rule | Edges |
|---|---|
| A | 130,814 |
| D | 18,310 |
| B | 4,527 |
| C | 2,190 |
| E | 880 |

Another 603 rule hits were vetoed by contradictions: tconf 41, vconf 257, pconf 215, dconf 154.

## 5. Clustering (`s5_cluster.py`)
Accepted edges are merged greedily in order of descending score, where score = 0.6·title + 0.4·authors + small volume/page bonuses − 0.05·year difference. A merge is refused if the merged cluster would contain two records from the same source, or would contain a pair the matcher hard-rejected. This blocked 501 and 10 merges respectively. Unmatched records stay as singletons.

Result: 76,130 entities.

| Source signature | Clusters |
|---|---|
| crossref + dblp + open_alex | 50,344 |
| dblp + open_alex | 5,075 |
| crossref + open_alex | 145 |
| crossref + dblp | 21 |
| crossref only | 10,239 |
| open_alex only | 5,155 |
| dblp only | 5,151 |

- The largest cluster has 3 records, and no cluster has two records from one source.
- Singleton share is 27%.
- Every record is assigned exactly once, and every membership id is a source id.

I inspected the best-scoring candidates of singletons. Most are clearly different papers by the same authors. Some residual misses remain:
- conference papers whose OpenAlex twin has a different year;
- DBLP records with damaged titles where author evidence is weak.

`correspondences.csv` lists every cross-source pair inside the final clusters.

## 6. Fusion (`s6_fusion.py`, per-cell provenance in `work/state/s6_provenance.csv`)
- **type:** majority vote. Ties go to dblp, then crossref, then open_alex (DBLP and Crossref share a vocabulary).
- **title:** vote on the comparison key. Ties go to the longest key, because truncation and damage shorten titles. Within the winning group, a rendering without a RETRACTED prefix is preferred, then open_alex > crossref > dblp.
- **authors:** one member's whole list, chosen by vote on order-free name-token sets. Ties go to dblp, then crossref, then open_alex. Lists are never unioned.
- **publication_year:** majority over values within the schema range 2018–2020. Ties go to dblp, then crossref, then open_alex. DBLP and Crossref agree on 99% of years; OpenAlex often reports the earlier online year.
- **journal:** vote on the venue token set. Ties go to crossref, then open_alex, then dblp, so full names beat DBLP abbreviations.
- **volume and issue:** vote over schema-valid values. Ties go to crossref, then open_alex, then dblp.
- **first/last page:** taken as a pair from one source, crossref > open_alex > dblp. The last page is dropped if it is invalid or smaller than the first page.
- **referenced_works_count and cited_by_count:** open_alex, falling back to crossref. The two sources rarely agree: reference counts are equal in only 11% of shared clusters, and citation counts are within the tolerance in 49%. I chose OpenAlex because the target attribute names are OpenAlex's own field names. This is an assumption, not something I could verify.

No value is guessed or taken from outside the member records.

## Diagnostics (from `work/diagnostics.jsonl`, revision r3)
- **Coverage and ids:** 100% of records in each source are covered. 0 unresolved ids, 0 duplicate memberships. The fused `_id` set equals the membership `cluster_id` set.
- **Schema violations in fused.csv:** 0 (enum, patterns, integer ranges, last page ≥ first page).
- **Density per attribute:**

| Attribute | Density |
|---|---|
| type, publication_year | 1.00 |
| title, authors | 0.999 |
| referenced_works_count, cited_by_count | 0.932 |
| first_page, last_page | 0.869 |
| journal | 0.809 |
| volume | 0.788 |
| issue | 0.487 |

- **Member disagreement rate** (share of clusters where members hold more than one distinct normalized value):

| Attribute | Disagreement |
|---|---|
| journal | 0.57 (mostly abbreviation vs full name) |
| cited_by_count | 0.65 |
| referenced_works_count | 0.63 |
| authors | 0.22 (name-form variants) |
| type | 0.21 |
| title | 0.14 |
| publication_year | 0.12 |
| issue | 0.017 |
| volume | 0.002 |

- **Traceability:** I checked 3,000 random clusters. Every fused title, journal, volume, issue, page, type and author list equals a value held by one of that cluster's own members.

## Known weaknesses and open questions
- The match thresholds are hand-set from inspected samples. Their error rates are unknown.
- Conference-vs-journal versions with missing pages and volumes can still be merged if their titles and authors agree. The OpenAlex type is always "article", so it gives no signal here.
- DBLP records whose titles were destroyed by markup are matched only on author and year evidence. Some remain singletons, and their fused titles are the damaged DBLP strings.
- The source preferences for counts, type ties and year are assumptions.
- I rejected, and did not use: embeddings, any supervised or Ditto model, and unioning author lists.
