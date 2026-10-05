<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - games / easy

_Generated at 2026-09-29T12:07:39.712865+00:00_

- domain: `games`
- level: `easy`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_games.yaml@fd979439ba3d`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee_games.yaml@c07cf762a7d5`, em_matching=`em_matching_committee_games.yaml@ead92fa3a2a8`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8813 | 0.6960 | 0.1853 |
| norm | macro_f1 | 0.0698 | 0.0806 | -0.0108 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9588 | 0.9560 | 0.0027 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.6275 | 0.6093 | 0.0182 |
| fusion | overall_accuracy | 0.6816 | 0.6948 | -0.0132 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 1.0000 | 0.5000 | 0.5000 |
| duplicate_majority | 0.8182 | 0.8444 | -0.0263 |
| embedding_sbert | 0.7692 | 0.6531 | 0.1162 |
| instance_tf_cosine | 0.6531 | 0.7059 | -0.0528 |
| label_jw | 1.0000 | 0.1875 | 0.8125 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.9286 | 0.9811 | -0.0526 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.0791 | 0.1209 | -0.0418 |
| passthrough | 0.0977 | 0.0604 | 0.0372 |
| rule_per_attribute_optimal | 0.0326 | 0.0604 | -0.0279 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9429 | 0.9528 | -0.0100 | 0.9973 | 0.9963 |
| embedding_blocker | 0.9429 | 0.9387 | 0.0042 | 0.9973 | 0.9963 |
| sc_block | 0.9571 | 0.9434 | 0.0137 | 0.9973 | 0.9963 |
| sorted_neighbourhood_blocker | 0.9194 | 0.9155 | 0.0039 | 0.9990 | 0.9988 |
| standard_blocker | 0.9905 | 0.9858 | 0.0046 | 0.9987 | 0.9988 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9633 | 0.9530 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.5693 | 0.6071 | -0.0378 | 0.5693 | 0.6071 | 0.5693 | 0.6071 |
| ditto_plm | 0.7104 | 0.7155 | -0.0051 | 0.7184 | 0.7155 | 0.7184 | 0.7155 |
| llm_matcher | 0.5755 | 0.6054 | -0.0298 | 0.5755 | 0.6054 | 0.5755 | 0.6054 |
| magellan | 0.6547 | 0.5092 | 0.1455 | 0.6675 | 0.5092 | 0.6675 | 0.5092 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.6782 | 0.7143 | -0.0361 |
| dbpedia_sales | ditto_plm | 0.8402 | 0.8122 | 0.0280 |
| dbpedia_sales | llm_matcher | 0.6971 | 0.7143 | -0.0171 |
| dbpedia_sales | magellan | 0.8430 | 0.5934 | 0.2496 |
| metacritic_dbpedia | comem | 0.4604 | 0.5000 | -0.0396 |
| metacritic_dbpedia | ditto_plm | 0.5806 | 0.6188 | -0.0381 |
| metacritic_dbpedia | llm_matcher | 0.4539 | 0.4965 | -0.0426 |
| metacritic_dbpedia | magellan | 0.4663 | 0.4250 | 0.0413 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.6330 | 0.6431 | -0.0101 |
| casefusion_only | 0.6898 | 0.6948 | -0.0050 |
| fusionquery_only | 0.7024 | 0.7201 | -0.0177 |
| llm_only | 0.6633 | 0.6696 | -0.0063 |
| ltm_only | 0.6545 | 0.6721 | -0.0177 |
| prefer_higher_trust_only | 0.7049 | 0.7125 | -0.0076 |
| pydi_per_attribute_optimal | 0.6847 | 0.6948 | -0.0101 |
| truthfinder_only | 0.7024 | 0.7175 | -0.0151 |
| voting_only | 0.6255 | 0.6368 | -0.0113 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| ESRB | 0.9800 | 0.9800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| criticScore | 0.3733 | 0.3733 | 0.0000 | 0.0267 | 0.0000 | 0.0267 |
| developer | 0.9158 | 0.9263 | -0.0105 | 0.1368 | 0.1579 | -0.0211 |
| genres | 0.7400 | 0.7400 | 0.0000 | 0.5900 | 0.6100 | -0.0200 |
| name | 0.9700 | 0.9700 | 0.0000 | 0.0400 | 0.0800 | -0.0400 |
| platform | 0.4500 | 0.4400 | 0.0100 | 0.1900 | 0.1800 | 0.0100 |
| publisher | 0.7059 | 0.7500 | -0.0441 | 0.0000 | 0.0000 | 0.0000 |
| releaseYear | 0.9000 | 0.9300 | -0.0300 | 0.4100 | 0.4000 | 0.0100 |
| userScore | 0.4182 | 0.4000 | 0.0182 | 0.0727 | 0.0364 | 0.0364 |
