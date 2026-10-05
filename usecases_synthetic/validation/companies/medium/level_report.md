<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - companies / medium

_Generated at 2026-09-29T12:07:39.710793+00:00_

- domain: `companies`
- level: `medium`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee.yaml@6f2bb9461525`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee.yaml@042119f4d1bf`, em_matching=`em_matching_committee.yaml@f0bb40e2173f`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7949 | 0.6855 | 0.1094 |
| norm | macro_f1 | 0.2718 | 0.3587 | -0.0869 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9822 | 0.9876 | -0.0054 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8721 | 0.8842 | -0.0121 |
| fusion | overall_accuracy | 0.5231 | 0.5765 | -0.0534 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.6875 | 0.3846 | 0.3029 |
| duplicate_majority | 0.8000 | 0.8000 | 0.0000 |
| embedding_sbert | 0.8108 | 0.7619 | 0.0489 |
| instance_tf_cosine | 0.6667 | 0.7692 | -0.1026 |
| label_jw | 0.7059 | 0.1739 | 0.5320 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.8936 | 0.9091 | -0.0155 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.3121 | 0.4348 | -0.1227 |
| passthrough | 0.2994 | 0.2935 | 0.0059 |
| rule_per_attribute_optimal | 0.2038 | 0.3478 | -0.1440 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9921 | 1.0000 | -0.0079 | 0.9853 | 0.9846 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9853 | 0.9846 |
| sc_block | 1.0000 | 1.0000 | 0.0000 | 0.9707 | 0.9691 |
| sorted_neighbourhood_blocker | 0.9611 | 0.9735 | -0.0124 | 0.9875 | 0.9867 |
| standard_blocker | 0.9611 | 0.9735 | -0.0124 | 0.9983 | 0.9985 |
| token_blocker | 0.9788 | 0.9788 | 0.0000 | 0.9931 | 0.9930 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8711 | 0.8950 | -0.0240 | 0.8711 | 0.8950 | 0.8711 | 0.8950 |
| ditto_plm | 0.8725 | 0.8699 | 0.0026 | 0.8716 | 0.8699 | 0.8716 | 0.8699 |
| llm_matcher | 0.8640 | 0.8848 | -0.0208 | 0.8640 | 0.8848 | 0.8640 | 0.8848 |
| magellan | 0.8808 | 0.8872 | -0.0064 | 0.8823 | 0.8872 | 0.8823 | 0.8872 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | comem | 0.8440 | 0.8649 | -0.0208 |
| forbes_dbpedia | ditto_plm | 0.8824 | 0.8750 | 0.0074 |
| forbes_dbpedia | llm_matcher | 0.8113 | 0.8673 | -0.0559 |
| forbes_dbpedia | magellan | 0.9023 | 0.9104 | -0.0082 |
| forbes_fullcontact | comem | 0.8981 | 0.9252 | -0.0271 |
| forbes_fullcontact | ditto_plm | 0.8627 | 0.8649 | -0.0021 |
| forbes_fullcontact | llm_matcher | 0.9167 | 0.9023 | 0.0143 |
| forbes_fullcontact | magellan | 0.8594 | 0.8640 | -0.0046 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.4770 | 0.5361 | -0.0590 |
| casefusion_only | 0.4492 | 0.4918 | -0.0426 |
| fusionquery_only | 0.4475 | 0.4590 | -0.0115 |
| llm_only | 0.4820 | 0.5082 | -0.0262 |
| ltm_only | 0.4459 | 0.5115 | -0.0656 |
| prefer_higher_trust_only | 0.4803 | 0.4967 | -0.0164 |
| pydi_per_attribute_optimal | 0.5066 | 0.5525 | -0.0459 |
| truthfinder_only | 0.4639 | 0.4607 | 0.0033 |
| voting_only | 0.4525 | 0.5148 | -0.0623 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.2045 | 0.2159 | -0.0114 | 0.0455 | 0.0568 | -0.0114 |
| city | 0.7385 | 0.7538 | -0.0154 | 0.1692 | 0.1231 | 0.0462 |
| country | 0.8400 | 0.8500 | -0.0100 | 0.2900 | 0.2900 | 0.0000 |
| founded | 0.8955 | 0.9254 | -0.0299 | 0.0746 | 0.1194 | -0.0448 |
| keypeople | 0.6667 | 0.7222 | -0.0556 | 0.0556 | 0.0000 | 0.0556 |
| name | 0.8300 | 0.8600 | -0.0300 | 0.3400 | 0.4600 | -0.1200 |
| revenue | 0.0694 | 0.2500 | -0.1806 | 0.0556 | 0.0556 | 0.0000 |
