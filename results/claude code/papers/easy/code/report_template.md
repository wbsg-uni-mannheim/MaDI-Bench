# Papers integration — method report

**No labeled evaluation and no supervised training were performed.** No reference matches, labels,
judge or teacher were used. None of the numbers below are accuracy, precision, recall, pair completeness or F1.
They are structural, label-free diagnostics taken from the saved artifacts.

Pipeline (`work/rebuild.sh`): `s1_schema` → `s2_normalize` → `s3_blocking` → `s4_features` + `s4_score` →
`s5_cluster` → `s6_fusion` → `s7_diagnostics`. The stages are deterministic, with no random steps except a
seeded diagnostic sample (seed 0). Intermediates are in `work/state/`, revision snapshots are in `work/revisions/`,
the step log is `work/step_log.jsonl`, and diagnostics are in `work/diagnostics.jsonl`.

## 1. Schema matching
All three sources (crossref 60,705 rows, dblp 60,690, open_alex 60,818) use the target attribute names. I checked
each mapping against example values rather than trusting the names alone, and every mapping is identity (score 1.0,
36 rows). Where a representation differs, the mapping stays the same and the value is transformed in
normalization: dblp `journal` holds ISO-style abbreviations, and dblp authors carry disambiguation suffixes
(`Bing Li 0005`). Unmapped source columns are `publisher`, `abstract_text` and `keywords`, which have no target
attribute. dblp has `referenced_works_count`/`cited_by_count` for only 1.3% of rows, and those rows are synthetic
records (see below). The target `id` is filled with the cluster id.

## 2. Normalization (`work/state/norm_*.pkl`; raw values are kept as `raw_*` columns)
- **type**: lowercased and mapped through aliases (`journal-article`, `journal article`, `journal_article` → `article`;
  `book_chapter` → `book-chapter`). Values outside the exhaustive enum (`book`, `album`: 11 per source) → null.
  Canonical rate is 0.9998 (`work/taxonomy_coverage.csv`, `work/taxonomy_plan.json`).
- **title**: double HTML-unescape, HTML/MathML tags stripped (inner text kept), zero-width characters removed,
  dblp's trailing period removed. The matching key is accent-folded, lowercased and alphanumeric only.
- **authors**: Python-list parse (0 failures). dblp 4-digit homonym suffixes are removed. Crossref lists sometimes
  contain comma-split affiliation fragments; these are removed at fusion by an affiliation-keyword filter.
- **numbers** (year, volume, issue, pages, counts): the data contains injected whitespace such as `2 018`, `12. 0`
  and `5 0.0`, plus thousands separators. Digit strings are de-spaced and `.0` is dropped. Alphanumeric locators
  (`e1005925`, `D607`, `S4`, `5-6`) are kept verbatim. There are 0 remaining numeric parse failures apart from
  2 values per source that were fixed by the thousands-separator rule.

## 3. Blocking (`submission/blocking/candidates.csv`, 4,523,110 pairs, the complete set)
This is the union of four candidate generators for each of the three source pairs:
- exact normalized-title key;
- char-3-gram TF-IDF cosine top-10 neighbours in both directions (cosine ≥ 0.3);
- for titles with fewer than 4 tokens (Crossref/OpenAlex often drop subtitles; some titles are truncated to 1–2 words):
  first title token + year, filtered to prefix relations;
- for the same short titles: pairs of author tokens + year.

Reduction ratio is ≥ 0.9995 per source pair. Records with at least one candidate: crossref 0.9999, dblp 0.9997,
open_alex 1.0. The records with no candidate have empty or garbage titles and no authors.
Without reference matches, this coverage is *not* pair completeness.

## 4. Entity matching (`s4_features.py`, `s4_score.py`)
Evidence is computed for each candidate pair:
- title: char ratio, token-set ratio, space-insensitive ratio (the data has injected spaces, e.g. `IR2V EC`), and
  prefix relations (dropped subtitles);
- author-token overlap / min;
- year difference;
- journal compatibility (equal tokens, or an in-order abbreviation-prefix test so that `J. Optim. Theory Appl.` ~
  the full name);
- equality of volume, issue, first page and last page;
- type equality (crossref–dblp only, because OpenAlex labels conference papers `article`);
- relative difference in reference count.

Score = title similarity T + small bounded adjustments. A missing value contributes 0 (it is treated as unknown,
not as agreement). Author disjointness gives −0.3. Year difference ≥ 2 gives only −0.08, because in 3-source
clusters OpenAlex disagrees with the other two on year 7.2k times (online-first versus issue year).
Accept if score ≥ 0.88, or if T ≥ 0.6, at least 2 hard identifiers agree with none contradicting, and author overlap ≥ 0.5.

I set the threshold by inspecting the score histogram (a valley at about 0.85–0.95) and by reading about 60
mutual-best pairs in the 0.7–0.95 band: below 0.85 they were almost all different papers by the same group.

Two guard rules came from inspected failure patterns:
1. **Partial-title rule.** If the titles are neither near-equal nor in a prefix relation, the pair is capped below
   threshold unless the token-subset holds and authors overlap ≥ 0.5, or ≥ 2 hard identifiers agree.
   This removed pairs such as "Probabilistic CBR for Open-World KG Completion" ≠ "Probabilistic CBR in Knowledge Bases".
2. **Generic-short-title rule.** For one-word non-acronym (truncated) titles such as `A`, `Global` or `Distributed`,
   the pair needs authors overlapping ≥ 0.5, or ≥ 2 hard identifiers when authors are unknown, or a unique prefix
   match within the same venue and year.

## 5. Clustering (`s5_cluster.py`)
Constrained greedy agglomeration. Accepted edges are processed in descending score order (ties broken by id), and
a merge happens only if:
- the result keeps ≤ 1 record per source (within a source, same-title records are different versions, e.g. a
  conference paper and its journal version; the entity definition keeps these separate); and
- no implied cross pair is a weak bridge: every implied pair must have T ≥ 0.8 (unless a title has ≤ 2 tokens) and
  score ≥ 0.7.

Edges rejected: 2,310 by the source constraint and 11 as weak bridges (for example, a WMT18 and a WMT19 system paper
chained through a dblp record). Both lists are in `work/state/rejected_*.csv`. Unmatched records remain singletons.
Correspondences are all cross-source pairs inside the final clusters: 156,194 pairs. Of these, 99.96% are blocking
candidates. The remaining 59 are transitively implied pairs involving truncated titles.

## 6. Fusion (`s6_fusion.py`, per-cell provenance in `work/state/fusion_provenance.csv`)
Each attribute is decided by a vote over comparison-normalized keys: title with dashes, quotes, accents and case
folded; journal as a token set; exact values for the rest. Ties go to a per-attribute source priority. I chose the
priorities from a label-free *odd-one-out* count over the 50k 3-source clusters, which counts how often a source
disagrees with the two others:
- title: dblp is the odd one out 15k times (mostly because it keeps subtitles). Priority open_alex > crossref > dblp.
- type: open_alex is the odd one out 12.6k times (it labels proceedings papers `article`). Priority dblp > crossref > open_alex.
- year: open_alex is the odd one out 7.2k times. Priority crossref > dblp > open_alex.
- volume, issue and pages: crossref > open_alex > dblp.

Authors use a medoid rule: the list with the highest summed name-level agreement with the other members wins.
Names match by token set, or by subset when at least 2 tokens are shared. Among equal lists the one keeping
diacritics wins, then priority crossref > dblp > open_alex.

The two counts exist only in crossref and open_alex, and their values disagree in about 57% of clusters.
open_alex is preferred there (the field names and the "as of" snapshot date suggest OpenAlex), with crossref as
fallback. **This is an assumption that I could not verify.**

Journal: full names win over dblp abbreviations through the vote and priority; an abbreviation is used only when no
full name exists. Schema validity: years outside 2018–2020, pattern-invalid volumes and pages, pages outside 1–100000,
last page < first page, and counts above the schema maxima are all set to null (never guessed). Authors are output as
a JSON list.

## Diagnostics (latest entry in `work/diagnostics.jsonl`)
DIAG_PLACEHOLDER

## Known limitations and uncertainty
- **Title fusion.** When Crossref and OpenAlex agree on the short title (subtitle dropped) and dblp has the full
  title, the short majority form is output. If the reference uses full titles, this is systematically wrong for about 9k clusters.
- **Count priority.** The preference for open_alex over crossref on referenced and cited-by counts is an assumption
  I could not verify.
- **Unresolvable truncated records.** Records whose titles are truncated to one word *and* have no authors mostly stay
  singletons, which is a conservative choice.
- **Genuine ambiguity.** Some same-title versions share authors, year and pages (e.g. two "Engineering Software for
  the Cloud" papers). Their assignment follows score order and may be swapped.
- **Unmatched records are singletons.** About 10k crossref records and about 5k dblp+OpenAlex pairs have no plausible
  partner in the other source(s); I inspected their best candidates and found different papers.
- **Synthetic records.** Records with ids above about 60,690 in every source are fictional (e.g. "Lantern Drift …")
  and are near-duplicated across sources. They go through the same rules; values outside the enum or year range are
  nulled in the fused output.

Rejected approaches:
- An author-token + year fallback for all records: it produced 27M candidates.
- Accepting via the author/journal bonus alone: it caused same-group different-paper merges.
- A strict "every implied pair must be an accepted candidate" bridge rule: it split correct clusters around truncated
  dblp titles, so I replaced it with the T/score bridge rule.
