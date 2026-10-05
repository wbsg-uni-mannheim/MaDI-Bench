# Papers integration (DBLP, Crossref, OpenAlex): method report

**No labeled evaluation, supervised training, or external judging was performed.** No labels, no teacher, no Ditto or other trained matcher, no pseudo-label sets. All thresholds are hand-set from inspecting the data. The numbers below are structural, label-free diagnostics. They are **not** precision, recall, F1, blocking pair completeness, or fusion accuracy.

Pipeline (`work/rebuild.sh` runs `work/s1_schema.py` … `s8_diagnostics.py`, about 35 min; intermediates are in `work/state/`):

## 1. Schema matching (`s1_schema.py`, `submission/sm_mapping.csv`)
- Each source's columns were mapped by comparing example values:
  - DBLP: `typ, ttl, auth, pub_yr, jrnl, vol, iss, fpage, lpage, ref_cnt, cite_cnt`
  - OpenAlex: `wt, tt, aus, pyr, jrn, vol, isu, fpa, lpa, rwn, cby`
  - Crossref: already uses the target names.
- Source-native `id` is kept. Every rule-based mapping has score 1.0.
- Unmapped (not in the target): `publisher/pub/pbl`, `abstract_text`, `keywords/kwds/kwd`.
- Coverage limits: DBLP has essentially no pages or counts (<1% filled), and Crossref `journal` is only 59% filled.

## 2. Normalization (`s2_normalize.py`, `work/taxonomy_coverage.csv`, `work/taxonomy_plan.json`)
The same rules apply to all sources. Raw values are kept next to the `_n` columns.

- **type** (closed enum): lowercase, then repair typos (`articlc`, `inproceeedings`, `book chapter`) to the nearest enum value within edit distance 2. 100% of non-empty values mapped.
- **year**: repair look-alike and keyboard-neighbour characters (`202 0`, `20l9`, `2O19`, `201B`, `w020`, `2920`), constrained to 2018–2020. Real out-of-range years (2021, 2006…) are kept at normalization but never fused. Parse rate ≥ 99.99%.
- **counts**: `12.O`, `0,0` → integers. Parse rate ≥ 99.86%; ambiguous values such as `1e.0` are left empty.
- **pages**: `1,345` → `1345`; stray spaces and `.0` removed; `None` → empty.
- **volume / issue**: digit spaces and `l`/`O` look-alikes fixed.
- **title / venue**: HTML entities unescaped, the trailing `.` of DBLP titles stripped. The match key additionally removes JATS/MathML tags and `RETRACTED:` prefixes.
- **authors**: three formats are parsed: Python list literal, `A, B, and C`, and `;`-separated. DBLP homonym suffixes (`Bing Li 0005`) are removed. About 4% of lists per source are scrambled by quote noise; these are flagged `authors_dirty` and are only used for matching through a token bag.

## 3. Blocking (`s3_blocking.py`, `submission/blocking/candidates.csv`)
For each source pair, the candidate set is the union of:
- exact normalized title key;
- title char-3/4-gram TF-IDF, top-10 nearest neighbours in both directions, cosine ≥ 0.35;
- author-surname TF-IDF, top-5 in both directions, cosine ≥ 0.5. This catches truncated or empty titles.

Results:
- About 715–735k candidates per source pair (2.17M total).
- Reduction ratio 0.9998 against 3.7×10⁹ possible pairs.
- 18–120 records per source pair have zero candidates.
- 132 extra pairs were added to the candidate file. These are transitive pairs inside final clusters (A–B and B–C matched, A–C never blocked), so every correspondence is contained in the candidate file.

## 4. Entity matching (`s4_features.py`, `score.py`, `s4_match.py`)
The score is additive, with hand-set log-odds-like weights (see `work/score.py`).

**Title**
- Base: max(char ratio, token-sort ratio) → +4 / +3 / +1.5 / −1 / −2.5 / −4.
- Exact key match → +5.
- Prefix/suffix truncation → up to +3.
- Main title before `:` equal to the other title (Crossref/OpenAlex often store only "KAPow"; DBLP stores "KAPow: …") → +3.5.
- Generic short titles shared by more than 4 records are capped at +2.
- Different numbers on both sides ("Part 1" vs "Part 3", "WMT18" vs "WMT19") → −3.
- A genuinely different word on both sides (not within edit distance 2 of any word on the other side) → −1.5.
- An empty or 1–2-character title is treated as unknown; the pair then needs author overlap ≥ 0.8.

**Other evidence**
- **Authors** (surname overlap, or token-bag overlap for scrambled lists): +2 / +1 / 0 / −2.5.
- **Year**: same +0.5, one apart −1, two or more apart −2.
- **Venue** (abbreviation-aware compatibility): +1 / −1.5.
- **Volume**: +1.5 / −2.5. **First page**: +1.5 / −2. Page `1` and volume = year earn only +0.3.
- **Crossref vs DBLP type**: article vs inproceedings → −2 (conference vs journal version).
- Missing values contribute 0.

**Decision rule**
- Accept at score ≥ 3.0. The per-record best-score histogram is bimodal, with its valley at about 0–3; I inspected pairs in every bin from 1 to 4.5.
- Then a greedy one-to-one assignment per source pair, by descending score. The data supports one row per entity per source: same-title records within one source were inspected and are different versions or editions.
- Result: 155,574 accepted pairs.

Examples inspected and rejected on purpose:
- conference vs journal versions with the same title (the brief defines these as different entities);
- "Part I / Part II";
- different papers by the same authors in the same volume.

## 5. Clustering (`s5_cluster.py`)
- Accepted edges are processed by descending score (ties broken by id). Two clusters merge only if the result still has at most one record per source.
- 49 edges violated this and were rejected (`work/state/s5_rejected_edges.csv`). Inspected examples were mostly near-duplicate pairs (SciMAT vs Citespace, NAR database issue 26th vs 27th).
- An earlier "remove the weakest edge" splitter was dropped: it over-split and depended on hash order.
- Correspondences are all cross-source pairs inside the final clusters: 155,892 pairs, of which 155,525 are direct edges.

## 6. Fusion (`s6_fusion.py`, provenance in `work/state/s6_provenance.csv`)
Each attribute is resolved from the cluster's own members. Support is the sum of similarity to all members' normalized values (a medoid vote). Deterministic tie-breaks were chosen from the agreement rates inside three-source clusters: Crossref–DBLP agree on year in 99% of clusters and on type in 98%; Crossref–OpenAlex agree on title in 97% and venue in 80%.

- **type / year**: majority vote; ties go to Crossref, then DBLP, then OpenAlex. Only years 2018–2020 are allowed.
- **title / authors**: medoid by string / name-token similarity; ties go to the longer value (truncation noise), then Crossref, then OpenAlex, then DBLP. Scrambled author lists are excluded when a clean list exists.
- **journal**: full names from Crossref/OpenAlex vote by exact key.
  - DBLP abbreviations add a vote through a dictionary learned from the clusters themselves: 1,295 abbreviation → full-name consensus entries, each needing ≥ 2 clusters and ≥ 50% share.
  - This repairs corrupted names such as "Senzors" and "Pattern 4ecognition".
  - It is not applied to DBLP-only entities, which keep the source's own value.
  - The expanded full name is the one transformed value that is not verbatim in a member. Diagnostics found 8 such cases in a 2,000-row sample.
- **volume / issue / pages**: majority vote, then priority Crossref, OpenAlex, DBLP. Values must match the schema pattern. `last_page < first_page` is blanked (this never happened).
- **referenced_works_count / cited_by_count**: Crossref and OpenAlex disagree in about 48% of clusters (OpenAlex is typically higher). There is no way to know which one the target uses, so **OpenAlex is preferred and Crossref is the fallback**. This is the least certain fusion decision.

## Label-free diagnostics (final revision, `work/diagnostics.jsonl`)
**Coverage and consistency**
- 182,067 of 182,067 source records are in the membership; 0 unresolved ids; 0 duplicates.
- Membership cluster ids and fused `_id` values match exactly in both directions.
- 100% of correspondences are in the candidate file and inside the same cluster. 0 same-source pairs; at most one record per source per cluster.

**Cluster structure**: 76,332 entities.
- all three sources: 50,157
- DBLP + OpenAlex: 5,079
- Crossref + OpenAlex: 320
- Crossref + DBLP: 22
- singletons: Crossref 10,250, DBLP 5,337, OpenAlex 5,167 (27% singleton share)

A sample of Crossref singletons checked by hand had no plausible counterpart (the best candidate was a different paper).

**Fused output**
- Density: type 0.97, title 0.998, authors 0.97, year 0.97, journal 0.74, volume 0.72, issue 0.36, first/last page 0.81, referenced 0.90, cited 0.90.
- Schema-invalid values: 0 for every attribute.
- Traceability (2,000 sampled rows): every fused value is present verbatim among the member values, except the 8 venue-dictionary expansions.

## Known weaknesses / uncertainty
- Which values the target uses for citation and reference counts is a guess (OpenAlex preferred).
- Where DBLP has the full title with subtitle but Crossref/OpenAlex have only the main title, the majority short form is emitted.
- Titles keep source markup (`<i>`, `<scp>`, MathML) when the majority of members carry it.
- Conference vs journal versions with identical titles are separated only by year, type, page, volume and venue evidence. Cases missing those fields may be merged or split wrongly.
- Empty-title records (DBLP titles cut at a leading symbol) are matched on authors, venue and volume only.
- Thresholds are hand-set; there is no measured error rate.
- Rejected alternatives:
  - weakest-edge component splitting (over-split, not deterministic);
  - a blanket short-title cap (lost distinctive short names such as "SketchStudio");
  - expanding DBLP abbreviations for DBLP-only entities.
