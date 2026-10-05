# Music release integration — method report (label-free)

**No labeled data was used, and nothing was trained.** No reference matches, labeling service, fusion teacher or
supervised/fine-tuned matcher was used. No embeddings were used either. All thresholds are hand-set rules. They
were checked by inspecting individual cases and by the label-free structural diagnostics below. None of the
numbers in this report is an accuracy, precision, recall, pair-completeness or F1 figure.

Entity = one specific issue of an album/EP/single. Discogs often lists many issues of the same title (for
example, 15 issues of *Nevermind*). The pipeline therefore treats these as distinct entities and never merges two
records from the same source.

Pipeline (`work/rebuild.sh`): `s1_schema.py → s2_normalize.py → s3_blocking.py → s4_features.py →
s5_match_cluster.py → s6_fusion.py`. The diagnostics come from `s7_diagnostics.py` and are logged to
`work/diagnostics.jsonl`. Revisions r1–r4 are in `work/revisions/`, and the final revision is r4.

## 1. Schema matching (`sm_mapping.csv`)
| source | columns → target |
|---|---|
| discogs | id, rel_nm→name, art_nm→artist, rel_dt→release-date, rel_ctry→release-country, dur→duration, lbl→label, gnr→genre, trks→tracks |
| lastfm | id, Attribute_2→name, Attribute_3→artist, Attribute_6→duration, Attribute_9→tracks |
| musicbrainz | id, t→name, ar→artist, rd→release-date, rc→release-country, du→duration, tk→tracks |

The lastfm columns are anonymous and were mapped from their values:
- Attribute_6 holds durations such as `12m 14s`, `15:03` or bare seconds (confidence 0.9).
- Attribute_2 is a title, but it sometimes carries an `Artist -  ` prefix or promo tags.

Coverage gaps:
- lastfm has no date, country, label or genre.
- musicbrainz has no label or genre.
- label and genre therefore come from discogs only.

## 2. Normalization (`normlib.py`, `s2_normalize.py`)
All raw values are kept as `raw_*` columns, and the input files are never modified.

- **Titles:** promo/noise tags are removed. These are `*popular*`, `[explicit]`, `- new -` and `Album:` on lastfm, and `(orig.)`, `(release)` and `(album)` on musicbrainz. The lastfm `Artist -  Title` prefix (dash plus double space) is also removed and kept as artist evidence. Mojibake is repaired with a cp1252/latin-1 round trip.
- **Artists:**
  - For comparison only: discogs `(n)` suffixes and `feat.` / `(and others)` are dropped, and musicbrainz `Last, First` is inverted.
  - In fused output, the discogs `(n)` suffix is kept, because lastfm copies of the same names also carry it (`John Williams (4)`).
  - Initial-abbreviated names (`T. Monochrome Set`, `B. Dylan`) are expanded only when exactly one observed full name fits the pattern. Names that discogs or musicbrainz list verbatim (`R. Kelly`) are never expanded. This produced 1,110 expansions.
- **Dates:** the formats handled are `YYYY-MM-DD`, `DD.MM.YYYY`, `MM/DD/YYYY`, `YYYY-MM` and `YYYY`, with OCR digit repair. Precision is recorded. A bare year becomes `YYYY-01-01` because the target requires a full date, and 80% of discogs dates are already on 01-01. This is a documented representational choice, not a known day. Parse rates are 97.9% (discogs) and 97.7% (musicbrainz).
- **Duration:** `1h 2m 3s`, `H:MM:SS`, `MM:SS`, seconds and thousands-separated numbers are all converted to integer seconds. Zero-duration placeholders (`0:00`, `0s`) are treated as missing.
- **Country:** values are mapped to canonical names through aliases, word permutations, OCR/skeleton matching, fuzzy matching and unique prefixes (see `work/taxonomy_plan.json`). Coverage is 98.1% for discogs and 99.1% for musicbrainz. Unresolved values (mostly 2-letter garbage) become missing.
- **Genre:** the Discogs genre vocabulary is kept rather than the taxonomy's `Genre Name` column (rationale in `taxonomy_plan.json`). Noisy atoms are repaired, and 98.9% of non-empty values map to known atoms. **This is the main uncertain modelling choice.** If the evaluator expects taxonomy parents such as `Electronic / Dance`, genre values will not match.
- **Labels:** each discogs label is kept, plus a spelling-repaired variant when a label at least 3× more frequent is within OCR-normalised ratio 88. There are 456 such repairs. Label is scored as a set that allows extra members, so the repairs are purely additive, but some of them are wrong.
- **Coverage table:** `work/taxonomy_coverage.csv`.

## 3. Blocking (`blocking/candidates.csv`, the complete set)
Four candidate generators are unioned across the three source pairs:
- (A) title character 3-gram TF-IDF, top-10 cosine ≥ 0.35, in both directions;
- (B) track-list word TF-IDF (max_df 0.05), top-10 ≥ 0.25;
- (C) artist+title word TF-IDF, top-10 ≥ 0.30;
- (D) exact normalized-title blocks, so that every issue of a title stays reachable. Blocks with more than 60 records on one side must also share an artist token.

Results:
- **Size:** 572,734 pairs out of 320.8M cross-source pairs, a reduction ratio of 0.9982.
- **Per source pair:** discogs–lastfm 299,943, discogs–musicbrainz 184,192, lastfm–musicbrainz 88,599.
- **Records with no candidates:** 417 discogs, 67 lastfm, 7 musicbrainz. These are mostly records without a title.
- **Per record:** median 23 candidates, max 1,728.

This is candidate coverage, not pair completeness. Every output correspondence is in the exported candidate file.

## 4. Entity matching (`s4_features.py`, `s5_match_cluster.py`)
The evidence score is an additive, hand-set score in log-odds-like points. Missing evidence contributes 0.

- **Title:**
  - +2 at token-sort ≥ .95; +1.5 at ≥ .85.
  - +1 when one title contains the other and token-sort ≥ .7, which covers dropped words.
  - +1 for a word-prefix truncation with the same artist; 0 for mere containment.
  - −2 below .5 and −0.5 otherwise.
  - An OCR-normalised second view of the strings is also compared.
- **Artist:** +2 at ≥ .9, +1 at ≥ .75, −2 below .6, −0.5 if unknown. The similarity is aware of initials and word order, and an initial alone cannot explain a full name.
- **Numeric title tokens:** −2 when both titles have numeric tokens and they disagree (`1969-1974` vs `1974-1979`).
- **Tracks:** fuzzy one-to-one track matching; a match found only after stripping `(... mix)` counts 0.5.
  - +4 with ≥ 5 matches, +3 with ≥ 2, +1.5 with 1.
  - These require ≥ 80% of the shorter list and ≥ 50–60% of the longer list. The longer-list condition is waived for lastfm pairs with ≥ 2 matches, because lastfm lists are often partial or extended.
  - A small overlap with a list of ≥ 4 tracks gives −1.5, and almost no overlap gives −2.
- **Duration:** +1.5 within 10 s or 2%, +0.5 within 10%, −1 beyond 30%.
- **Date and country:** same year +0.5, year gap ≥ 2 −1.5, identical exact date +1, country agree +0.5, country differs −1.

The threshold is 3.0: exact title plus near-exact artist, or strong track-list evidence. Accepted pairs at 3.0
and rejected pairs at 1.5–2.5 were inspected. Near the boundary, rejected pairs are mostly generic titles with a
missing artist (`Sunrise`, `Singles`), and accepted pairs are mostly truncated or abbreviated forms of the same
release. Several rule changes (r2–r4) came from wrong cases found this way:
- single-track lastfm records linked to albums;
- remix singles linked through version-stripped tracks;
- a link justified only by a title that names one of the release's tracks;
- *Best of Bowie* 1969–74 vs 1974–79.

Threshold sensitivity (number of clusters of size 2/3):
- threshold 2.5 → 3,251 / 801
- threshold 3.0 → 3,126 / 784
- threshold 3.5 → 2,853 / 752
- threshold 4.0 → 2,675 / 724

## 5. Clustering / refinement
Accepted edges are merged greedily, strongest first. The ranking uses the coarse score, plus a small continuous
tie-break (string similarities and duration closeness), plus a triangle-support bonus when both endpoints link to
the same third-source record. The bonus is capped below one evidence step.

An edge is rejected in three cases:
- it would put two records of one source into a cluster (1,548 edges, mostly alternative issues);
- any other cross pair of the two clusters has evidence ≤ 0 (the weak-bridge guard, 17 edges);
- for issue ambiguity: a lastfm or musicbrainz record has ≥ 3 discogs issues tied at the maximal score and no third-source support. Picking one would be a coin flip, so those discogs links are withheld (102 records).

Final result:
- 30,475 clusters in total: 26,565 singletons, 3,126 pairs and 784 three-source clusters.
- Maximum of one record per source per cluster.
- Share of records in multi-source clusters: discogs 13.7%, lastfm 32.8%, musicbrainz 70.0%.
- `correspondences.csv` lists all 5,478 cross-source pairs inside the final clusters, and it agrees with membership (checked).

## 6. Fusion (`fused.csv`, provenance in `work/state/s6_provenance.csv`)
Each attribute is resolved only from the cluster's own members:

- **name / artist:** members are grouped by order-insensitive token set. The largest group wins. On ties the pipeline prefers, in order:
  1. a token superset (the noise observed here deletes or truncates words);
  2. fewer noise flags (double space from a deleted word, mojibake, OCR tokens like `E1len`, initials, `(.` truncation, `Last, First`);
  3. source order discogs > musicbrainz > lastfm.
- **release-date:** majority vote; then day precision that is not 01-01, before a 01-01 day, month or year; then discogs.
- **release-country:** majority vote, then discogs.
- **duration:** a value confirmed by another member within 5 s wins. Otherwise the source order is musicbrainz > discogs > lastfm. That order comes from a label-free agreement count: in 3-source clusters where the other two agree, musicbrainz agreed 171/178, discogs 170/200 and lastfm 171/340.
- **tracks:** majority by case-insensitive set, then fewer mojibake tracks, then discogs > musicbrainz > lastfm. Output is a JSON list.
- **label, genre:** taken from the discogs member.

Resolution counts over the 3,910 multi-member clusters:

| attribute | outcomes |
|---|---|
| name | unanimous 2,754, superset 767, majority 232, tie-break 157 |
| artist | unanimous 2,964, tie-break 481 |
| date | agree 2,917, conflict 506 |
| country | agree 3,230, conflict 143 |
| duration | agree 1,833, tie-break 1,149 |
| tracks | agree 2,047, tie-break 1,481 |

Fused density:

| attribute | density |
|---|---|
| name | .974 |
| artist | .917 |
| date | .586 |
| country | .663 |
| label | .661 |
| genre | .653 |
| tracks | .769 |
| duration | .361 |

## Structural checks (final revision r4, `work/diagnostics.jsonl`)
- **Coverage:** all 35,169 source ids appear exactly once in membership; 0 ids fail to resolve; source labels are consistent.
- **Consistency:** fused `_id` set equals the membership cluster set; 100% of correspondences lie inside the candidate set; 0 same-source pairs.
- **Formats:** 100% of dates are valid ISO `YYYY-MM-DD`, durations are integers, and tracks are JSON lists.
- **Genre vs taxonomy:** 39% of fused genre values occur anywhere in the supplied taxonomy (see the genre caveat above).
- **Fused rows traced to members:** a random sample of 3-member clusters was checked manually.
- **Reproducibility:** `rebuild.sh` was run in a fresh copy with no state, and all five files were identical to the submission.

## Known weaknesses and uncertainty
- **Choice of discogs issue:** lastfm records often carry only title and artist (60% have no track list; 67% have no duration). The pipeline cannot then tell which discogs issue they correspond to. Links with 2 tied issues are still decided by a weak continuous tie-break.
- **Heavy injected noise:** truncations, word drops, OCR errors, abbreviations and swapped countries. It limits both matching and fusion. The noise in singleton records (e.g. `Psychos`, truncated titles) cannot be repaired.
- **Gold-format guesses:** genre vocabulary, the country canonical form, and 01-01 for year-only dates are guesses about the expected format.
- **Weights and threshold:** the additive score weights and the 3.0 threshold are judgement calls, with no measured precision or recall.

## Rejected / not used
- Supervised matchers, pseudo-labels and Ditto (not permitted).
- Embeddings (not needed; TF-IDF plus rules).
- Record-id proximity across sources: the ids appear loosely co-ordered, but this was deliberately not used as evidence.
- Mapping genre to taxonomy parents (see the genre caveat).
