<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - games / hard

_Generated at 2026-09-29T12:07:39.714401+00:00_

- domain: `games`
- level: `hard`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_games.yaml@fd979439ba3d`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee_games.yaml@c07cf762a7d5`, em_matching=`em_matching_committee_games.yaml@ead92fa3a2a8`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6926 | 0.6960 | -0.0034 |
| norm | macro_f1 | 0.2237 | 0.0806 | 0.1431 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9474 | 0.9560 | -0.0087 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.7263 | 0.6093 | 0.1170 |
| fusion | overall_accuracy | 0.5611 | 0.6948 | -0.1337 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.4571 | 0.5000 | -0.0429 |
| duplicate_majority | 0.8444 | 0.8444 | 0.0000 |
| embedding_sbert | 0.6809 | 0.6531 | 0.0278 |
| instance_tf_cosine | 0.6275 | 0.7059 | -0.0784 |
| label_jw | 0.3125 | 0.1875 | 0.1250 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.9259 | 0.9811 | -0.0552 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.2500 | 0.1209 | 0.1291 |
| passthrough | 0.2237 | 0.0604 | 0.1632 |
| rule_per_attribute_optimal | 0.1974 | 0.0604 | 0.1369 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9567 | 0.9528 | 0.0039 | 0.9959 | 0.9963 |
| embedding_blocker | 0.9663 | 0.9387 | 0.0277 | 0.9959 | 0.9963 |
| sc_block | 0.9567 | 0.9434 | 0.0133 | 0.9959 | 0.9963 |
| sorted_neighbourhood_blocker | 0.8694 | 0.9155 | -0.0461 | 0.9988 | 0.9988 |
| standard_blocker | 0.9443 | 0.9858 | -0.0416 | 0.9981 | 0.9988 |
| token_blocker | 0.9909 | 1.0000 | -0.0091 | 0.9564 | 0.9530 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.6264 | 0.6071 | 0.0192 | 0.4680 | 0.6071 | 0.6264 | 0.6071 |
| ditto_plm | 0.8227 | 0.7155 | 0.1072 | 0.5256 | 0.7155 | 0.6711 | 0.7155 |
| llm_matcher | 0.6506 | 0.6054 | 0.0452 | 0.5024 | 0.6054 | 0.6506 | 0.6054 |
| magellan | 0.8056 | 0.5092 | 0.2964 | 0.7149 | 0.5092 | 0.7832 | 0.5092 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.6374 | 0.7143 | -0.0769 |
| dbpedia_sales | ditto_plm | 0.8000 | 0.8122 | -0.0122 |
| dbpedia_sales | llm_matcher | 0.6947 | 0.7143 | -0.0195 |
| dbpedia_sales | magellan | 0.7984 | 0.5934 | 0.2049 |
| metacritic_dbpedia | comem | 0.6154 | 0.5000 | 0.1154 |
| metacritic_dbpedia | ditto_plm | 0.8455 | 0.6188 | 0.2267 |
| metacritic_dbpedia | llm_matcher | 0.6065 | 0.4965 | 0.1100 |
| metacritic_dbpedia | magellan | 0.8128 | 0.4250 | 0.3878 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.5750 | 0.6431 | -0.0681 |
| casefusion_only | 0.5561 | 0.6948 | -0.1387 |
| fusionquery_only | 0.5826 | 0.7201 | -0.1375 |
| llm_only | 0.5801 | 0.6696 | -0.0895 |
| ltm_only | 0.5536 | 0.6721 | -0.1185 |
| prefer_higher_trust_only | 0.5864 | 0.7125 | -0.1261 |
| pydi_per_attribute_optimal | 0.5498 | 0.6948 | -0.1450 |
| truthfinder_only | 0.5738 | 0.7175 | -0.1438 |
| voting_only | 0.5649 | 0.6368 | -0.0719 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| ESRB | 0.8000 | 0.9800 | -0.1800 | 0.0400 | 0.0000 | 0.0400 |
| criticScore | 0.3600 | 0.3733 | -0.0133 | 0.0000 | 0.0000 | 0.0000 |
| developer | 0.7789 | 0.9263 | -0.1474 | 0.1053 | 0.1579 | -0.0526 |
| genres | 0.4800 | 0.7400 | -0.2600 | 0.2200 | 0.6100 | -0.3900 |
| name | 0.9400 | 0.9700 | -0.0300 | 0.0700 | 0.0800 | -0.0100 |
| platform | 0.5000 | 0.4400 | 0.0600 | 0.2200 | 0.1800 | 0.0400 |
| publisher | 0.4265 | 0.7500 | -0.3235 | 0.0000 | 0.0000 | 0.0000 |
| releaseYear | 0.8600 | 0.9300 | -0.0700 | 0.3900 | 0.4000 | -0.0100 |
| userScore | 0.3455 | 0.4000 | -0.0545 | 0.0545 | 0.0364 | 0.0182 |
