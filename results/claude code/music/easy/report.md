# Music release integration — method report (label-free)

**No labeled evaluation and no supervised training were done.** I had no reference matches, labeling service or fusion teacher. I trained or fine-tuned no model, and no embedding model was used. All thresholds come from inspecting source records and from structural (label-free) diagnostics. The numbers below are structural observations. They are not precision, recall, pair completeness or fusion accuracy.

Pipeline: `work/s1_schema.py` → `s2_normalize.py` → `s3_blocking.py` → `s4a_features.py` → `s4b_match.py` → `s5_cluster.py` → `s6_fusion.py` → `s7_diagnostics.py`. Run all of it with `work/rebuild.sh`. I ran the rebuild in a fresh workspace copy, and all five data files came out identical to the submission.

## 1. Schema matching
Every source uses target attribute names, and I checked the meanings against the values:
- discogs has all 9 attributes.
- lastfm has id, name, artist, duration and tracks.
- musicbrainz has everything except label and genre.

Two things need transformation rather than direct copying:
- Duration comes as `mm:ss` or `h:mm:ss` in discogs and lastfm, and as integer seconds in musicbrainz and some discogs rows. Everything is converted to seconds.
- Country uses different vocabularies: discogs uses short names like "UK", musicbrainz uses ISO-style long names.

Target fields missing from a source are left missing. Genre and label exist only in discogs.

## 2. Normalization
Original values are kept alongside the normalized ones in `work/state/normalized.csv`.

- **Comparison keys** are built by fixing mojibake (cp1252/latin1 round-trip), folding accents, lowercasing, removing punctuation and changing `&` to `and`.
- **Injected noise found and handled:**
  - lastfm, plus about 40 discogs rows: `*popular*` / `[explicit]` title prefixes (stripped unless the tag is the whole title), dropped or shuffled title words, and abbreviated or truncated artists ("R. Trent", "Type O", "Hole (.").
  - musicbrainz: `(album)` / `(orig.)` / `(release)` suffixes (stripped), "Last, First" artist order (reordered for comparison), and mojibake in titles and tracks.
  - discogs: multi-artist `A|B` credits and `(n)` disambiguators (both handled for comparison only).
- **Dates:** spaces and missing hyphens are repaired (for example `199701-01` becomes `1997-01-01`). Every non-null date parses as `YYYY-MM-DD`.
- **Duration:** a value of `0` is treated as unknown. Apart from zeros, every value parsed.
- **Country aliases:** mapped to the musicbrainz/ISO form used in the target examples (UK / United Kingdom / Great Britain → "United Kingdom of Great Britain and Northern Ireland", CA → Canada, NO → Norway, NZ → New Zealand, Russia → Russian Federation, and so on). Regions such as "Europe" and "UK & Europe" are kept as they are.
- **Genre:** the target declares the taxonomy `Music_Genres_Taxonomy.csv`, whose top-level names are things like "Electronic / Dance". The discogs genres ("Electronic", "Rock", "Pop|Rock", …) mostly have no exact counterpart there. Mapping "Electronic" to "Electronic / Dance" would be an invention, and the schema examples ("Electronic", "House") use discogs-style values. So the discogs genre is output **verbatim**. This is an explicit uncertainty.

## 3. Blocking
The candidate set is the union of three nearest-neighbour methods, each run per source pair and in both directions:
1. TF-IDF on character 3-grams of the normalized title: top 15, cosine ≥ 0.35.
2. The same on title + artist: top 15.
3. Word TF-IDF on the normalized tracklist: top 5, cosine ≥ 0.4, only for records with at least 2 tracks.

Results:
- **913,925 candidate pairs** in total: discogs–lastfm 461,653; discogs–musicbrainz 292,038; lastfm–musicbrainz 160,234.
- **Reduction ratio 0.9979.**
- **Pairs found only by one method:** title 591k, title + artist 119k, tracks 225.
- **Records with no candidates:** discogs 320, lastfm 29, musicbrainz 0.
- **Most candidates for one record:** 1,809.
- **Every correspondence is in the candidate set** (checked).

## 4. Entity matching
The entity is a specific release issue. Discogs often has several issues of the same title and artist, so title and artist establish *identity*, and issue-level evidence picks *which issue*.

Pair features (`s4a`, `s4b`):
- **Title similarity:** average of token-sort and token-set ratios. Token-set tolerates the words lastfm drops.
- **Artist similarity:** the maximum of fuzzy ratio, a space-insensitive ratio, prefix match, and a token alignment that allows initials. The alignment is *directed* for lastfm: every lastfm token must align to the other artist, because lastfm drops or abbreviates tokens but never adds them. For example, "Spirit Gang" does not match "Spirit".
- **Fuzzy tracklist containment `ft`:** greedy track pairing with token-sort ≥ 80.
- **Duration difference**, plus date and country agreement for discogs–musicbrainz pairs.

Score = `id_score` (0.5·title + 0.5·artist) + 0.25·tracks term + 0.08·duration term + 0.12·date term + 0.06·country term. Each term is in [-1, 1], and a term is 0 when the value is missing, so missing counts as unknown rather than agreement.

A pair is accepted if either route passes:
- **Standard:** `id_score ≥ 0.80` and `score ≥ 0.85`. If a pair has no issue-level evidence at all, `id_score ≥ 0.9` is required.
- **Alternative:** title token-set ≥ 0.9, at least 3 tracks on both sides, `ft ≥ 0.8`, `id_score ≥ 0.65` and `score ≥ 0.85`. This covers artist variants like "Dub Inc" / "Dub Incorporation" and "PlantLife" / "Plant Life".

A pair is vetoed when both tracklists are known, `ft < 0.3`, and the durations differ by more than 5 s.

Then a **greedy global one-to-one assignment per source pair** runs in score order, with ties broken by id. In revision 1 I used mutual-best selection instead; it dropped a correct pair whenever a neighbouring reissue competed for the same record.

How I got to these rules:
- I inspected unmatched musicbrainz records' best candidates, accepted pairs in the 0.8–0.95 score band, and unmatched lastfm records' best candidates.
- Most of the unmatched lastfm best candidates are different artists that share a generic title, so I left them unmatched.

## 5. Clustering
Accepted edges are added highest score first, and an edge is refused if it would put two records from the same source in one cluster. I checked the data assumption behind this: there are no duplicate ids, and discogs rows with the same title and artist are different issues.

- **Edges refused by this constraint: 176**, listed in `work/state/rejected_edges.csv`. The ones I inspected are correctly refused. Typically a lastfm record is a copy of one discogs issue while the musicbrainz record matches a different issue (Gomez "Liquid Skin", Buckcherry "Confessions", Type O Negative "Bloody Kisses").
- **Clusters:** 29,675 in total — 23,457 singletons, 2,413 with two records and 3,805 with three.
- **Source combinations:** discogs+lastfm+musicbrainz 3,805; discogs+lastfm 1,631; discogs+musicbrainz 446; lastfm+musicbrainz 336.
- **Singleton share:** 0.79.
- **Records per source per cluster:** at most 1.
- `correspondences.csv` lists every cross-source pair inside each cluster: 13,828 pairs. The score is the edge score, capped at 1.

## 6. Fusion
Each attribute is resolved only from the cluster's own records (`s6_fusion.py`; per-cell rules are in `work/state/fusion_provenance.csv`). The source priority discogs > musicbrainz > lastfm is a documented tie-break, not an estimated accuracy. It reflects the observed injected noise in lastfm titles and artists and in musicbrainz suffixes and mojibake.

| attribute | rule |
|---|---|
| name | Majority vote on the normalized key; ties go to source priority. The cleaned value is output (prefix, suffix and mojibake removed). |
| artist | First available by priority. Discogs gives the full credit (`A\|B` kept verbatim). A musicbrainz "Last, First" is reordered. |
| release-date | Kept if the sources agree. A `YYYY-01-01` value (likely a year-only placeholder) is replaced by a more specific date from the same year. Otherwise discogs wins. |
| release-country | Normalized country; discogs first. |
| label, genre | Discogs only, verbatim. Label is `\|`-separated. |
| tracks | Tracklist of the highest-priority record that has one, as a JSON list. |
| duration | The value with the most other values within ±5 s; ties go to priority. |
| id | The representative source-native id of the cluster (first member by priority). `_id` is the cluster id. |

Nothing is imputed: missing stays missing.

Share of fused rows with a value, per attribute: name 1.0, artist 0.98, date 0.69, country 0.76, label 0.73, genre 0.75, tracks 0.83, duration 0.52.

How attributes were resolved:
- **Dates:** sources agreed in 19,615 cells; 568 used the specific date over a placeholder; 300 disagreements were settled by priority.
- **Country:** discogs and musicbrainz agree in 92.6% of clusters where both have a value. Most disagreements are region versus country, such as "Europe" versus UK.

## Diagnostics on the final files (`work/diagnostics.jsonl`)
All of these pass:
- 0 unresolved ids, 0 records under the wrong source, 0 duplicate memberships.
- Every source is fully covered (1.0).
- Membership cluster ids and fused `_id` values match in both directions.
- Every correspondence is inside the candidate set and inside a single cluster.
- 0 correspondences between records of the same source.
- Every date is valid ISO, every duration is an integer, every tracklist is valid JSON.
- Every fused name traces back to one of its cluster's members (0 violations).

`scripts/self_check.py` passes.

## Known weaknesses and open uncertainty
- **Undecidable issue choice:** about 574 accepted discogs edges (517 of them with lastfm) had a competing discogs issue scoring within 0.01, usually because the lastfm record has no tracks or duration. The issue picked is then a deterministic tie-break (lowest id), not a matter of evidence.
- **Lastfm's title-word dropping** makes token-set title similarity necessary, which also admits some sub- or supertitle confusions ("Harvest" / "Harvest Moon" by the same artist). Artist agreement and issue evidence limit this, but cannot exclude it.
- **Genre and artist output formats** (verbatim discogs genre and multi-artist credit) are guesses about the expected representation. So is preferring a specific date over a `-01-01` date.
- **Thresholds** were chosen by inspecting borderline cases, not by optimizing against any labels, and I cannot measure matching quality.
- **Rejected approaches:**
  - Mutual-best selection, which lost valid pairs.
  - A symmetric artist-abbreviation rule, which caused false merges such as "Spirit (8)" with "Spirit Gang" and "Red Baron" with "Reba Rambo".
  - Mapping genres onto the taxonomy's top level.
  - Using record-id numbering as evidence: I deliberately avoided it.
