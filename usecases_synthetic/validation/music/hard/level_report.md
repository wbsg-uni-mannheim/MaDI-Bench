<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - music / hard

_Generated at 2026-09-29T12:07:39.717241+00:00_

- domain: `music`
- level: `hard`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_music.yaml@f269c37ab587`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee_music.yaml@07cec0b9969f`, em_matching=`em_matching_committee_music.yaml@2c90f2517590`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7726 | 0.6696 | 0.1030 |
| norm | macro_f1 | 0.4053 | 0.4875 | -0.0822 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9322 | 0.9505 | -0.0183 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8560 | 0.8242 | 0.0318 |
| fusion | overall_accuracy | 0.5428 | 0.7511 | -0.2083 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.6000 | 0.2500 | 0.3500 |
| duplicate_majority | 0.8649 | 0.8649 | 0.0000 |
| embedding_sbert | 0.7660 | 0.7692 | -0.0033 |
| instance_tf_cosine | 0.7805 | 0.7619 | 0.0186 |
| label_jw | 0.4444 | 0.0909 | 0.3535 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.9524 | 0.9500 | 0.0024 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.5309 | 0.6125 | -0.0816 |
| passthrough | 0.2654 | 0.3000 | -0.0346 |
| rule_per_attribute_optimal | 0.4198 | 0.5500 | -0.1302 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9765 | 0.9985 | -0.0220 | 0.9962 | 0.9964 |
| embedding_blocker | 0.9726 | 0.9970 | -0.0244 | 0.9962 | 0.9964 |
| sc_block | 0.9965 | 0.9985 | -0.0020 | 0.9924 | 0.9927 |
| sorted_neighbourhood_blocker | 0.8349 | 0.8498 | -0.0150 | 0.9955 | 0.9959 |
| standard_blocker | 0.8422 | 0.8589 | -0.0166 | 0.9868 | 0.9858 |
| token_blocker | 0.9706 | 1.0000 | -0.0294 | 0.9118 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.7464 | 0.6914 | 0.0550 | 0.6754 | 0.6914 | 0.7464 | 0.6914 |
| ditto_plm | 0.9698 | 0.9521 | 0.0177 | 0.8765 | 0.9521 | 0.8993 | 0.9521 |
| llm_matcher | 0.7557 | 0.7002 | 0.0555 | 0.6858 | 0.7002 | 0.7557 | 0.7002 |
| magellan | 0.9520 | 0.9531 | -0.0010 | 0.9293 | 0.9531 | 0.9509 | 0.9531 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8000 | 0.8307 | -0.0307 |
| musicbrainz_discogs | ditto_plm | 0.9627 | 0.9194 | 0.0433 |
| musicbrainz_discogs | llm_matcher | 0.8039 | 0.8483 | -0.0444 |
| musicbrainz_discogs | magellan | 0.9365 | 0.9243 | 0.0122 |
| musicbrainz_lastfm | comem | 0.6928 | 0.5522 | 0.1407 |
| musicbrainz_lastfm | ditto_plm | 0.9770 | 0.9848 | -0.0079 |
| musicbrainz_lastfm | llm_matcher | 0.7075 | 0.5522 | 0.1553 |
| musicbrainz_lastfm | magellan | 0.9675 | 0.9818 | -0.0143 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.5427 | 0.7645 | -0.2218 |
| casefusion_only | 0.5386 | 0.7149 | -0.1763 |
| fusionquery_only | 0.5413 | 0.7066 | -0.1653 |
| llm_only | 0.5551 | 0.6970 | -0.1419 |
| ltm_only | 0.5275 | 0.7355 | -0.2080 |
| prefer_higher_trust_only | 0.5399 | 0.7617 | -0.2218 |
| pydi_per_attribute_optimal | 0.5441 | 0.7562 | -0.2121 |
| truthfinder_only | 0.5661 | 0.7576 | -0.1915 |
| voting_only | 0.5441 | 0.7576 | -0.2135 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.8800 | 0.9600 | -0.0800 | 0.1100 | 0.2200 | -0.1100 |
| duration | 0.1899 | 0.8608 | -0.6709 | 0.0253 | 0.1013 | -0.0759 |
| genre | 0.2800 | 0.4267 | -0.1467 | 0.0000 | 0.0000 | 0.0000 |
| label | 0.6136 | 0.9659 | -0.3523 | 0.0114 | 0.0227 | -0.0114 |
| name | 0.8900 | 0.9300 | -0.0400 | 0.0500 | 0.1300 | -0.0800 |
| release-country | 0.6122 | 0.9796 | -0.3673 | 0.2347 | 0.7041 | -0.4694 |
| release-date | 0.9688 | 0.9896 | -0.0208 | 0.0417 | 0.0417 | -0.0000 |
| tracks | 0.0889 | 0.2000 | -0.1111 | 0.0889 | 0.2000 | -0.1111 |
