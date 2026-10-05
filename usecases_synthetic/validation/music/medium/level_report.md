<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - music / medium

_Generated at 2026-09-29T12:07:39.716497+00:00_

- domain: `music`
- level: `medium`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_music.yaml@f269c37ab587`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee_music.yaml@07cec0b9969f`, em_matching=`em_matching_committee_music.yaml@2c90f2517590`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8537 | 0.6696 | 0.1842 |
| norm | macro_f1 | 0.4676 | 0.4875 | -0.0199 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9437 | 0.9505 | -0.0068 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8345 | 0.8242 | 0.0103 |
| fusion | overall_accuracy | 0.7202 | 0.7511 | -0.0309 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.8333 | 0.2500 | 0.5833 |
| duplicate_majority | 0.8333 | 0.8649 | -0.0315 |
| embedding_sbert | 0.8718 | 0.7692 | 0.1026 |
| instance_tf_cosine | 0.7179 | 0.7619 | -0.0440 |
| label_jw | 0.7429 | 0.0909 | 0.6519 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.9767 | 0.9500 | 0.0267 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.6157 | 0.6125 | 0.0032 |
| passthrough | 0.2222 | 0.3000 | -0.0778 |
| rule_per_attribute_optimal | 0.5648 | 0.5500 | 0.0148 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9955 | 0.9985 | -0.0030 | 0.9964 | 0.9964 |
| embedding_blocker | 0.9955 | 0.9970 | -0.0015 | 0.9964 | 0.9964 |
| sc_block | 1.0000 | 0.9985 | 0.0015 | 0.9927 | 0.9927 |
| sorted_neighbourhood_blocker | 0.8288 | 0.8498 | -0.0210 | 0.9960 | 0.9959 |
| standard_blocker | 0.8423 | 0.8589 | -0.0165 | 0.9859 | 0.9858 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9067 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.7296 | 0.6914 | 0.0381 | 0.7296 | 0.6914 | 0.7296 | 0.6914 |
| ditto_plm | 0.9417 | 0.9521 | -0.0104 | 0.9111 | 0.9521 | 0.9111 | 0.9521 |
| llm_matcher | 0.7189 | 0.7002 | 0.0186 | 0.7189 | 0.7002 | 0.7189 | 0.7002 |
| magellan | 0.9480 | 0.9531 | -0.0050 | 0.9480 | 0.9531 | 0.9480 | 0.9531 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8625 | 0.8307 | 0.0318 |
| musicbrainz_discogs | ditto_plm | 0.9031 | 0.9194 | -0.0162 |
| musicbrainz_discogs | llm_matcher | 0.8567 | 0.8483 | 0.0084 |
| musicbrainz_discogs | magellan | 0.9173 | 0.9243 | -0.0070 |
| musicbrainz_lastfm | comem | 0.5966 | 0.5522 | 0.0445 |
| musicbrainz_lastfm | ditto_plm | 0.9803 | 0.9848 | -0.0045 |
| musicbrainz_lastfm | llm_matcher | 0.5811 | 0.5522 | 0.0289 |
| musicbrainz_lastfm | magellan | 0.9787 | 0.9818 | -0.0031 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.6584 | 0.7645 | -0.1061 |
| casefusion_only | 0.7025 | 0.7149 | -0.0124 |
| fusionquery_only | 0.6667 | 0.7066 | -0.0399 |
| llm_only | 0.6226 | 0.6970 | -0.0744 |
| ltm_only | 0.6281 | 0.7355 | -0.1074 |
| prefer_higher_trust_only | 0.6433 | 0.7617 | -0.1185 |
| pydi_per_attribute_optimal | 0.7259 | 0.7562 | -0.0303 |
| truthfinder_only | 0.7424 | 0.7576 | -0.0152 |
| voting_only | 0.6419 | 0.7576 | -0.1157 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.9500 | 0.9600 | -0.0100 | 0.1800 | 0.2200 | -0.0400 |
| duration | 0.1772 | 0.8608 | -0.6835 | 0.0886 | 0.1013 | -0.0127 |
| genre | 0.4267 | 0.4267 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| label | 0.8409 | 0.9659 | -0.1250 | 0.0227 | 0.0227 | 0.0000 |
| name | 0.9300 | 0.9300 | 0.0000 | 0.1300 | 0.1300 | 0.0000 |
| release-country | 0.8878 | 0.9796 | -0.0918 | 0.5816 | 0.7041 | -0.1224 |
| release-date | 0.9896 | 0.9896 | 0.0000 | 0.0521 | 0.0417 | 0.0104 |
| tracks | 0.7667 | 0.2000 | 0.5667 | 0.6889 | 0.2000 | 0.4889 |
