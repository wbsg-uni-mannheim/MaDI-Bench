# Music release integration — method report

**No labeled evaluation, no supervised training, no labeling/judging service, and no embeddings or LLM
calls were used.** All thresholds were hand-set from inspecting source values. The numbers below are
label-free structural diagnostics measured on the saved artifacts, not accuracy scores. No precision, recall, F1,
pair completeness or fusion accuracy is claimed.

Pipeline: `work/s1_schema.py` → `s2_normalize.py` → `s3_blocking.py` → `s4_features.py` → `s5_match.py`
→ `s6_cluster.py` (T=3.5) → `s7_fusion.py` → `s8_diagnostics.py`. Entrypoint: `work/rebuild.sh`. It is
deterministic, runs in about 5 minutes on 1 CPU and makes no network calls. Running it in a fresh copy of the
workspace reproduced every submission file exactly. Revisions are in `work/revisions/`, the step log is in
`work/step_log.jsonl`, and the diagnostics are in `work/diagnostics.jsonl`.

## 1. Schema matching (`sm_mapping.csv`)
The mapping was done by hand from column meanings and example values:
- **discogs** `rec_uid,title_str,performer,pub_dt,origin_loc,duration,imprint,category,tracks_track-name`
- **lastfm** `item_code,album_title,band,album_length,tracks_track-name`
- **musicbrainz** `Attribute_1..6,9`

Target attributes with no source in a given dataset:
- lastfm has no date, country, label or genre.
- musicbrainz has no label or genre.
- Label and genre therefore come only from discogs.

Mapping uncertainty: discogs `category` is a Discogs genre list, not the taxonomy, so it is mapped through an alias
table (score 0.8). The lastfm name and artist fields are heavily perturbed (score 0.9).

## 2. Normalization (`work/state/records.pkl`, raw values kept as `raw_*`)
- **Noise patterns found by inspection and removed symmetrically:**
  - lastfm title prefixes `*popular*`, `[explicit]`, `Album:`, `- new -`, and the `Artist -  Title` prefix (removed only when the prefix equals the record's own artist)
  - musicbrainz title suffixes `(orig.)`, `(album)`, `(release)`
  - musicbrainz artist noise `(and others)` and a trailing `feat.`, plus the `Last, First` / `Name, The` inversion
  - mojibake repair (cp1252/latin1 → utf8, repeated)
- **Comparison keys:** lowercase, accents stripped, `&`→`and`, non-alphanumerics removed. Only these keys are case-folded; fused output keeps the source spelling.
- **Dates:** precision is recorded (day / month / year).
- **Duration:** 0 is treated as missing (schema rule).
- **Country:** discogs short names are mapped to the full names musicbrainz uses (`UK` → `United Kingdom of Great Britain and Northern Ireland`, `Russia` → `Russian Federation`, …). The vocabulary is non-exhaustive, so market labels (`Europe`, `UK & US`) are kept.
- **Genre:** the taxonomy is exhaustive, so each Discogs genre is mapped to a `Genre Name`. The primary genre is the first Discogs genre that has a mapping:
  - Electronic → Electronic / Dance
  - Funk / Soul → R&B / Soul
  - Hip Hop → Hip Hop / Rap
  - Reggae → Reggae / Dub
  - Stage & Screen → Soundtrack / Score
  - Latin → World Music
  - Folk, World, & Country → Folk
  - identical names map to themselves

  Non-Music, Children's and Brass & Military have no defensible equivalent and are left missing. Coverage: 22,466 of 22,627 discogs rows are canonical (0.993, see `work/taxonomy_coverage.csv`). Plan in `work/taxonomy_plan.json`.
- Track lists: 0 parse failures. Lastfm has an empty track list for 53% of rows.

## 3. Blocking (`blocking/candidates.csv`, complete set)
The candidate set is the union of TF-IDF nearest neighbours, computed in both directions for every source pair:
- title character 3–5-grams, top 10
- title + artist word tokens, top 5
- track-list words, top 5

Results:
- 1,039,147 candidate pairs; reduction ratio 0.9973 versus the full cross product.
- 0 records without a candidate.
- Audit: every same-artist, similar-title (token-set ≥ 80) pair involving an unlinked musicbrainz or lastfm record is already a candidate.
- Candidate coverage is not pair completeness.

## 4. Entity matching (`s4`, `s5`)
**Features** (see `s4_features.py`):
- title fuzzy ratio and token-set ratio
- initial-aware artist token matching (lastfm `R. Trent` = `Ron Trent`, `T.` = `The`), with compact equality (`C/A/T` = `Cat`) and per-artist matching for discogs `A|B` credits
- fuzzy track-list overlap
- duration difference
- year and date equality
- country equality

**Lastfm title rule:** lastfm drops and truncates words, so containment only counts when the *lastfm* tokens ⊆ the other title and cover at least half of it. A differing number, `vol` or `part` caps title similarity at 0.7, because these usually mark a different volume.

**Gate:** title ≥ 0.75 and artist ≥ 0.75, **or** title ≥ 0.9, artist ≥ 0.5 and track overlap ≥ 0.6 (at least 3 tracks on each side).

**Score** = 2·title + 2·artist, plus:
- tracks: 2·overlap, and −1 for an overlap below 0.3
- duration: +1 within 5 s, +0.5 within 5%, −1 above 20%
- year: +0.5 same year, −1.5 if more than 1 year apart; +0.5 for the same full date
- country: +0.5 same; −0.5 for a different country (not applied to market labels)

For lastfm pairs, the track and duration penalties are switched off, because lastfm track lists and lengths are album-level and noisy (only 28% of lastfm/musicbrainz durations agree within 5 s). Accept threshold T = 3.5, which is roughly name + artist agreement with at most a mild contradiction.

**Threshold sensitivity** (cluster structure only):

| T | correspondences |
|---|---|
| 3.0 | 9,370 |
| 3.5 | 9,363 |
| 4.0 | 8,948 |
| 4.5 | 7,897 |

The pairs I inspected in the 3.0–3.5 band were mostly different issues of the same title (different year or track list, e.g. Showbiz 1999 vs 2003), so they are rejected. Track-only similarity was rejected as evidence because generic track names produced junk pairs.

## 5. Refinement / clustering (`s6_cluster.py`)
The task defines each release issue as its own entity, and discogs contains many reissues (and exact duplicate rows) of the same title. The clustering therefore works as follows:
- Accepted edges are processed greedily by descending score, then by the lower numeric native id. This is a deterministic tie-break: a lower Discogs id means the record was catalogued earlier.
- A merge is refused if it would put two records of the same source in one cluster (2,159 edges refused this way), or if any explicitly scored cross pair between the two clusters scores below 2.5 (4 edges).
- Ambiguous near-ties are flagged but not dropped.
- Unmatched records stay as singletons.

Decisions are logged in `work/state/edge_decisions.csv`. Correspondences are all cross-source pairs within each final cluster (9,363); every one of them is a blocking candidate and in the same cluster.

## 6. Fusion (`s7_fusion.py`, provenance in `work/state/fusion_provenance.csv`)
- **name:** discogs > musicbrainz > lastfm, using cleaned values (discogs is the least noisy source).
- **artist:** vote over normalized keys, counting sources; ties go to discogs > musicbrainz > lastfm; the spelling comes from the highest-priority supporter.
- **release-date:** the discogs date if it has day precision. Otherwise a same-year day-precision date from musicbrainz. Otherwise a year or year-month completed as `-01-01` / `-01`, the convention both sources already use. Dates must fall within 1900–2016.
- **release-country, label (discogs `|` list), genre, tracks:** the first non-empty value in priority order (tracks: discogs > musicbrainz > lastfm).
- **duration:** groups of values within ±5 s; the group supported by the most sources wins; ties go to discogs > lastfm > musicbrainz (the schema example 903 follows lastfm over musicbrainz).
- Nothing is imputed; missing stays missing.

## 7. Diagnostics (final run, from the saved files)
**Coverage and integrity:**
- 37,255 source records, all in `membership.csv` exactly once; 0 unresolved ids.
- Cluster ids and fused `_id` are the same set in both directions; `_id` is unique.
- 29,578 fused rows. Cluster sizes: 23,587 singletons, 4,305 pairs, 1,686 triples. Singleton share 0.80.
- Composition: 1,686 discogs+lastfm+musicbrainz, 1,479 lastfm+musicbrainz, 1,416 discogs+lastfm, 1,410 discogs+musicbrainz.
- Share of records linked to another source: musicbrainz 0.96, lastfm 0.46, discogs 0.20. Discogs is a 22.6k superset with many releases that appear in no other source.

**Validity:** 0 invalid dates, 0 genres outside the taxonomy, 0 invalid durations, 0 invalid track lists.

**Traceability:** every fused value equals, or is the documented transformation of, a value from its own cluster members (0 untraceable).

**Density:**

| attribute | all rows | multi-source rows |
|---|---|---|
| date | 0.74 | 0.95 |
| country | 0.79 | 0.92 |
| label | 0.77 | 0.75 |
| genre | 0.76 | 0.75 |
| tracks | 0.86 | 1.00 |
| duration | 0.57 | 0.91 |

**Disagreement within multi-source clusters** (normalized): artist 0.30 (mostly lastfm initials), name 0.21 (lastfm word loss), country 0.04.

## 8. Known weaknesses / unresolved uncertainty
- **Reissue choice:** discogs reissues with identical title, artist and tracks and no date, country or duration to separate them are resolved by the lowest-id tie-break, which may pick the wrong issue. 23% of discogs+lastfm clusters involve a discogs record that has identical-key siblings.
- **Lastfm singletons:** duplicate lastfm rows of the same album cannot all join (one record per source per cluster), so they remain singletons.
- **Artist disambiguators:** discogs `(2)`-style suffixes are kept unless the other sources outvote them.
- **Genre mapping:** mapping Discogs genres onto the taxonomy is a judgment call, especially `Folk, World, & Country` → Folk and `Latin` → World Music.
- **Date convention:** it is unknown whether the evaluator expects padded `YYYY-01-01` or the more specific musicbrainz date. I chose the more specific date whose year agrees.
- **Duration conflicts:** duration disagreements between sources are frequent, and the tie-break priority is a heuristic.
- **Rejected approaches:** track-only matching (generic track names), skipping ambiguous ties (this discarded well-supported duplicate matches), a symmetric token-set title similarity (it over-matched, e.g. "Casanova" ↔ "Terra Casanova"), and embeddings (not needed; TF-IDF blocking left no evident gaps).
