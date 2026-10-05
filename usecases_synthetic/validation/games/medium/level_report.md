<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - games / medium

_Generated at 2026-09-29T12:07:39.713649+00:00_

- domain: `games`
- level: `medium`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_games.yaml@fd979439ba3d`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee_games.yaml@c07cf762a7d5`, em_matching=`em_matching_committee_games.yaml@ead92fa3a2a8`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7850 | 0.6960 | 0.0890 |
| norm | macro_f1 | 0.1988 | 0.0806 | 0.1182 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9540 | 0.9560 | -0.0020 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.6716 | 0.6093 | 0.0623 |
| fusion | overall_accuracy | 0.6534 | 0.6948 | -0.0414 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7907 | 0.5000 | 0.2907 |
| duplicate_majority | 0.7619 | 0.8444 | -0.0825 |
| embedding_sbert | 0.7200 | 0.6531 | 0.0669 |
| instance_tf_cosine | 0.6250 | 0.7059 | -0.0809 |
| label_jw | 0.6341 | 0.1875 | 0.4466 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.9630 | 0.9811 | -0.0182 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.1963 | 0.1209 | 0.0754 |
| passthrough | 0.2444 | 0.0604 | 0.1840 |
| rule_per_attribute_optimal | 0.1556 | 0.0604 | 0.0951 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9476 | 0.9528 | -0.0052 | 0.9964 | 0.9963 |
| embedding_blocker | 0.9429 | 0.9387 | 0.0042 | 0.9964 | 0.9963 |
| sc_block | 0.9667 | 0.9434 | 0.0233 | 0.9964 | 0.9963 |
| sorted_neighbourhood_blocker | 0.9004 | 0.9155 | -0.0151 | 0.9988 | 0.9988 |
| standard_blocker | 0.9667 | 0.9858 | -0.0192 | 0.9989 | 0.9988 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9545 | 0.9530 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.5818 | 0.6071 | -0.0253 | 0.5835 | 0.6071 | 0.5818 | 0.6071 |
| ditto_plm | 0.7683 | 0.7155 | 0.0528 | 0.6788 | 0.7155 | 0.6770 | 0.7155 |
| llm_matcher | 0.5728 | 0.6054 | -0.0326 | 0.5744 | 0.6054 | 0.5728 | 0.6054 |
| magellan | 0.7634 | 0.5092 | 0.2542 | 0.7554 | 0.5092 | 0.7554 | 0.5092 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.7021 | 0.7143 | -0.0122 |
| dbpedia_sales | ditto_plm | 0.8231 | 0.8122 | 0.0108 |
| dbpedia_sales | llm_matcher | 0.6872 | 0.7143 | -0.0271 |
| dbpedia_sales | magellan | 0.8103 | 0.5934 | 0.2169 |
| metacritic_dbpedia | comem | 0.4615 | 0.5000 | -0.0385 |
| metacritic_dbpedia | ditto_plm | 0.7136 | 0.6188 | 0.0948 |
| metacritic_dbpedia | llm_matcher | 0.4583 | 0.4965 | -0.0381 |
| metacritic_dbpedia | magellan | 0.7164 | 0.4250 | 0.2914 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.6482 | 0.6431 | 0.0050 |
| casefusion_only | 0.6469 | 0.6948 | -0.0479 |
| fusionquery_only | 0.6696 | 0.7201 | -0.0504 |
| llm_only | 0.6393 | 0.6696 | -0.0303 |
| ltm_only | 0.6494 | 0.6721 | -0.0227 |
| prefer_higher_trust_only | 0.6721 | 0.7125 | -0.0404 |
| pydi_per_attribute_optimal | 0.6469 | 0.6948 | -0.0479 |
| truthfinder_only | 0.6810 | 0.7175 | -0.0366 |
| voting_only | 0.6318 | 0.6368 | -0.0050 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| ESRB | 0.9400 | 0.9800 | -0.0400 | 0.1000 | 0.0000 | 0.1000 |
| criticScore | 0.3733 | 0.3733 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| developer | 0.8842 | 0.9263 | -0.0421 | 0.1789 | 0.1579 | 0.0211 |
| genres | 0.7000 | 0.7400 | -0.0400 | 0.2300 | 0.6100 | -0.3800 |
| name | 0.9600 | 0.9700 | -0.0100 | 0.0600 | 0.0800 | -0.0200 |
| platform | 0.4400 | 0.4400 | 0.0000 | 0.1800 | 0.1800 | 0.0000 |
| publisher | 0.5882 | 0.7500 | -0.1618 | 0.0000 | 0.0000 | 0.0000 |
| releaseYear | 0.9000 | 0.9300 | -0.0300 | 0.3700 | 0.4000 | -0.0300 |
| userScore | 0.4000 | 0.4000 | 0.0000 | 0.0545 | 0.0364 | 0.0182 |
