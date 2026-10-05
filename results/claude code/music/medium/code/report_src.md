# Music release integration — method report

**No labeled evaluation and no supervised training were performed.** No labels, judges, reference
mappings, pseudo-label sets or trained matchers were used or built. Every number below is a label-free
structural diagnostic computed from saved artifacts. None of them is a precision, recall, F1, PC or fusion-accuracy
estimate. Correctness of the output is unknown to me.

Pipeline: `work/s1_schema.py` → `s2_normalize.py` → `s3_blocking.py` → `s4_features.py` →
`s5_match_cluster.py` → `s6_fusion.py`. `work/rebuild.sh` reruns all six deterministically (about 4 minutes on 1 CPU, no network, no
randomness). Diagnostics are appended to `work/diagnostics.jsonl`; decisions are logged in `work/step_log.jsonl`.
Coherent snapshots are in `work/revisions/` (r1 = first complete pipeline, one record per source per cluster;
r2 = with within-source copy groups; the final submission is r2 plus fusion/normalization refinements).

## 1. Schema matching (`sm_mapping.csv`)
I mapped columns by meaning after inspecting values, not by name alone:

| source | columns → target |
|---|---|
| discogs | all 9 columns map 1:1 (name, artist, release-date, release-country, duration [s], label, genre, tracks) |
| lastfm | t_nm→name, ar_nm→artist, dur_s (h:mm:ss)→duration, tnms→tracks. It has no date, country, label or genre |
| musicbrainz | rel_nm→name, art→artist, rel_dt→release-date, rel_ctry→release-country, dur (m:ss, minutes may exceed 60)→duration, trks→tracks. It has no label or genre |

Lastfm names/artists get score 0.95 because they carry prefixes and abbreviations. Everything else is 1.0.

## 2. Normalization (`s2_normalize.py`; raw values kept in `*_raw` columns)
- **Mojibake**: repeated cp1252→utf-8 repair (musicbrainz is single-encoded, lastfm often double-encoded).
  Some strings are lossy (`?` replacement chars) and cannot be repaired.
- **Titles**: removed lastfm artifacts `[explicit]`, `- new -`, `Album:`, `*popular*` and the `Artist -  Title`
  prefix (marked by a double space). Removed `(album)`, `(orig.)` and `(release)` suffixes (mostly musicbrainz). A double space
  in a raw title is kept as a "word dropped" marker. The comparison key is ASCII-folded lowercase alphanumerics.
- **Artists**: removed discogs `(n)` disambiguators. Un-inverted musicbrainz `Last, First[, Band][, The]`, and removed
  `(and others)`, `and others` and a dangling `feat.`. Removed the lastfm dangling ` .` / ` (.`. Abbreviated initials
  (`F. McDonald`) are handled in matching, not rewritten.
- **Dates**: `dd.mm.yyyy`, ISO and `yyyymmdd` formats. OCR-style digit corruption (O→0, l/I/i→1, B→8, G→6, q→9) and stray spaces are fixed.
  Year-month values (`2010-09`) are kept only as year/month, never padded into a full date. Parse rate ≥99.9%
  (`work/taxonomy_coverage.csv`).
- **Country**: The canonical form is the full name used in the schema examples (UK/USA → "United Kingdom of Great Britain and
  Northern Ireland"/"United States of America"). The vocabulary is built from frequent clean values, plus aliases
  (Nippon, Deutschland, Holland, Éire, …), repair of word-shuffled/typo'd/space-split forms, and 2-character corruptions of `UK`.
  Multi-region values (`UK & Europe`, `Europe`) are kept as-is. Canonical rate: discogs 99.5%, musicbrainz 99.9%.
  For comparison only, Russia and Russian Federation share one key.
- **Genre (taxonomy decision)**: Only discogs has genre. Its values are the 15 Discogs top-level genres
  (`Electronic`, `Funk / Soul`, `Folk, World, & Country`, …). The supplied taxonomy's `Genre Name` column
  (`Electronic / Dance`, …) does not contain these values, and the schema's own examples (`Electronic`, `House`, `Techno`)
  are not taxonomy Genre Names either. So I treated the taxonomy as non-exhaustive and kept Discogs spelling.
  Heavy noise (typos, shuffled words across `|`, flattened lists like `Pop Rock Funk Soul`, synonyms like `Kids` or
  `Rock & Roll`) is parsed word by word into Discogs genres. Order comes from the most frequent clean ordering of the same set.
  Multi-genre values stay `|`-joined. Canonical rate is 99.99%. See `work/taxonomy_plan.json`.
- **Duration**: converted to integer seconds. Musicbrainz `0:00` is treated as unknown.
- **Tracks**: parsed from list literals; the lastfm `[]` is treated as unknown. I added a comparison-only "core" key that drops
  bracketed qualifiers and ` - …` suffixes (`(Album Version)`, `- X feat. Y`, BBC-session tags).

## 3. Blocking (`submission/blocking/candidates.csv`, 672,701 pairs, complete set)
This is the union, per source pair, of three generators:
- (A) top-10 nearest neighbours (both directions) on char-3-gram TF-IDF of the title key, cosine ≥0.35;
- (B) the same on title+artist;
- (C) shared rare track titles (≥2 shared, or 1 when a release has ≤2 tracks).

Per-pair reduction ratios: 0.9985 (discogs–lastfm), 0.9979 (discogs–musicbrainz), 0.9975 (lastfm–musicbrainz).
Pairs unique to each method: A 374,758; B 126,016; C 7,432. Records with zero candidates: discogs 313, lastfm 25, musicbrainz 1.
One asserted pair was added through copy-group expansion (a copy of the record was a candidate).
Every correspondence is contained in the exported set (checked by re-reading the files).
Within-source candidates for copy detection (discogs 41k, lastfm 12k pairs) are internal and not exported.
Candidate coverage here is NOT pair completeness.

## 4. Entity matching (`s4_features.py`, `s5_match_cluster.py`)
Each candidate pair gets an additive evidence score from hand-set, interpretable points (nothing is fitted):

| evidence | points |
|---|---|
| title similarity (token-sort; one missing word = 0.97 with drop marker, 0.9 otherwise) | ≥0.97: +3, ≥0.9: +2, ≥0.8: +0.5, ≥0.7: −1, else −3 |
| artist (token containment with initial matching; whitespace-free char ratio for typos; missing = 0) | ≥0.9: +2, ≥0.7: +1, ≥0.5: −0.5, else −3 |
| tracklist (fuzzy core-title overlap / larger list; containment of smaller list) | ≥0.9: +3, ≥0.7: +2, containment ≥0.8: +1, ≥0.5: 0, else −2 |
| same track count ≥3 (not lastfm) | +0.5 |
| duration, discogs–musicbrainz | ≤10 s: +2, ≤60 s: +1, ≤300 s or ≤10%: 0, else −1 |
| duration, pairs involving lastfm (partial lists) | ≤5 s: +1.5, ≤30 s: +0.5, else 0 |
| date (discogs–musicbrainz) | same full date +2 (+1 if a `-01-01` placeholder), same year +0.5, ±1 year −0.5, else −2 |
| country | equal +1, different −0.5 |

Why these rules:
- On pairs whose title and artist are unambiguous, discogs–musicbrainz durations differ by 1 s at the median, so duration is strong evidence there.
- Lastfm durations and tracklists are often partial (median difference 96 s), so for lastfm only agreement counts.
- Discogs durations are sometimes about 2× the musicbrainz value for otherwise identical releases, so a duration contradiction is only −1.

Acceptance: score ≥4.5, or score ≥4.0 when the artist positively agrees. The second case is the typical
"one title word dropped + same artist" pattern; inspecting the 4.0–4.5 band showed that the implausible edges there had a missing artist.
An edge must be (near-)best for both endpoints within its source pair (tie tolerance 0.5).
If an endpoint has >2 near-tied alternatives, the edge is "ambiguous". Ambiguous edges are used only if the ambiguity
disappears after unambiguous edges are placed; 174 ambiguous edges were rejected. Example: a lastfm album with only
title+artist, facing several discogs issues of that title. Choosing one at random would rarely hit the right issue.
Structural sensitivity (not accuracy; measured on the pre-copy-group revision): T=4.0/4.5/5.0 gave 8.8k/8.4k/8.3k correspondences, and MAX_TIES 1/2/3 gave 8.2k/8.4k/8.5k.
The structure is stable.

## 5. Refinement / clustering
- **Within-source copies.** Inspecting high-scoring edges rejected by a one-record-per-source constraint showed that
  discogs and lastfm contain noisy *copies of the same release*. Examples: two Kiss "MTV Unplugged" discogs rows
  identical in every field; seven lastfm "Doggystyle" rows sharing the same odd track order with random tracks dropped
  and artists abbreviated. Musicbrainz has no such copies (2 same title+artist pairs in 4.8k records).
- **Copy rules.**
  - Discogs: title ≥0.9, compatible artist, NO conflict on date/year/country/label/duration/tracklist when both
    sides have a value, and ≥2 positively agreeing fields. Merges are complete-link, so no conflicting pair can exist inside a group.
    Same-title discogs rows that differ in date, country or label are kept apart as different issues.
  - Lastfm: title ≥0.97 (or ≥0.9 with track containment ≥0.8), artist ≥0.9 (or missing artist with track containment
    ≥0.8 and ≥3 tracks), and track containment ≥0.8 whenever both lists exist.
  - Result: 575 discogs and 748 lastfm copy edges; copy-group sizes 2:727, 3:70, 4:19, 5+:15 (largest are 11 and 9).
    The largest groups were inspected and look like copies.
- **Cross-source clustering.** Record edges are lifted to copy-group edges. Greedy union runs in descending score order
  with the constraint of ≤1 copy-group per source per cluster, so there is no transitive chaining of two different issues.
  Unmatched records stay singletons. Rejected edges are saved in `work/state/rejected_edges.pkl` (231 source conflicts, 174 ambiguous).
- **Correspondences.** These are all cross-source record pairs inside the final clusters (9,159: discogs–mb 3,225,
  lastfm–mb 3,069, discogs–lastfm 2,865). Within-source copy pairs are not listed, because the spec asks for cross-source pairs.
  The score is the evidence score divided by 16, clipped to [0,1].

Diagnostics for the final revision:
- 37,335 records, all assigned once, all ids native. 29,212 clusters; singleton share 0.793.
- Size distribution: 1: 23,165; 2: 4,116; 3: 1,839; 4–12: 92.
- Share of records in multi-record clusters: musicbrainz 94.2%, lastfm 47.9%, discogs 21.7%.
- Composition includes 1,603 discogs+lastfm+mb, 1,412 discogs+mb, 1,409 lastfm+mb and 796 discogs+lastfm clusters.
- Every correspondence is inside its own cluster and inside the candidate set.

## 6. Fusion (`fused.csv`; `_id` = `cluster_id`; `id` = representative native id)
Values come only from the cluster's own normalized members (per-cell rule in `work/state/fusion_provenance.csv`):
- **name**: vote on the normalized key. Tie-breaks: no word-drop marker, fewer noise abbreviations (`Mus.`), more words (noise drops words), then discogs > musicbrainz > lastfm.
- **artist**: support = the number of members compatible with the candidate (abbreviations and typos support the full
  form). Tie-breaks: fewer initials, exact-spelling frequency, then source priority.
- **release-date**: full ISO dates only. Vote; tie-breaks: not a `-01-01` placeholder, then musicbrainz > discogs.
  Year-only or year-month values are never padded.
- **release-country**: vote on the comparison key; tie → musicbrainz > discogs.
- **duration**: a value confirmed by another member within 5 s; otherwise musicbrainz > discogs > lastfm.
- **tracks** (JSON list): the medoid tracklist. With ≥3 lists, each track's spelling is decided by an aligned-track majority (changed 90 lists).
- **label**: union over discogs copies (`|`-joined; set attribute).
- **genre**: vote over discogs copies.

Density (share of non-null fused values): name 1.0, artist 0.913, date 0.641, country 0.714, label 0.686, genre 0.687, tracks 0.786,
duration 0.392. All dates match `YYYY-MM-DD` and all durations are integers. These are density and validity checks, not accuracy.
Trace check (re-reading `fused.csv` and `membership.csv`): all 29,212 fused rows were checked. There were 0 values of name, artist,
date, country, duration, genre, tracks or label that do not occur among the normalized values of the row's own members.
Reproducibility: `work/rebuild.sh` was run in a fresh copy with `work/state` emptied. It reproduced all submission files
exactly (sorted-row comparison).

## Known weaknesses / unresolved uncertainty
- **Issue-level ambiguity.** Lastfm has no issue attributes (date, country, label). When several discogs issues share
  title and artist, a lastfm record is linked only if tracks or duration single out one issue; otherwise it stays unlinked.
- **Copy detection is heuristic.** Two genuinely different issues whose distinguishing fields are all blank may have been merged.
  Conversely, copies whose fields were corrupted into conflicts stay separate.
- **Unresolvable disagreements.** Where discogs and musicbrainz disagree on date or country (e.g., `Europe` vs a
  specific country) or on duration (discogs ≈2×), the tie-break is a documented guess (prefer musicbrainz).
  Two-member typo disagreements (`Suit`/`Suite`) cannot be resolved by voting.
- **Genre vocabulary.** Kept as the Discogs vocabulary with `|`-joined multi-genres. If the evaluator expects taxonomy names
  or a single primary genre, genre values will not match.
- **Label noise.** Label strings with injected corruption cannot be repaired from a single source. The union keeps every observed variant.
- **Blocking set size.** The candidate set is large (672k) because I preferred wide coverage; I cannot measure its true completeness.
- **Rejected approaches.**
  - One-record-per-source clustering (revision r1): contradicted by observed within-source copies.
  - Pure embedding or TF-IDF matching: it cannot separate issues.
  - Mapping genres into the taxonomy's `Electronic / Dance`-style names: not supported by the schema examples.
  - Any supervised or pseudo-labelled matcher: not permitted.
