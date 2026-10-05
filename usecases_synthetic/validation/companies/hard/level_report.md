<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - companies / hard

_Generated at 2026-09-29T12:07:39.711539+00:00_

- domain: `companies`
- level: `hard`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee.yaml@6f2bb9461525`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee.yaml@042119f4d1bf`, em_matching=`em_matching_committee.yaml@f0bb40e2173f`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6640 | 0.6855 | -0.0215 |
| norm | macro_f1 | 0.2991 | 0.3587 | -0.0596 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9838 | 0.9876 | -0.0038 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8737 | 0.8842 | -0.0106 |
| fusion | overall_accuracy | 0.4079 | 0.5765 | -0.1686 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.2500 | 0.3846 | -0.1346 |
| duplicate_majority | 0.8000 | 0.8000 | 0.0000 |
| embedding_sbert | 0.7059 | 0.7619 | -0.0560 |
| instance_tf_cosine | 0.7027 | 0.7692 | -0.0665 |
| label_jw | 0.3200 | 0.1739 | 0.1461 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.8696 | 0.9091 | -0.0395 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.3462 | 0.4348 | -0.0886 |
| passthrough | 0.2692 | 0.2935 | -0.0242 |
| rule_per_attribute_optimal | 0.2821 | 0.3478 | -0.0658 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9876 | 1.0000 | -0.0124 | 0.9850 | 0.9846 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9850 | 0.9846 |
| sc_block | 0.9921 | 1.0000 | -0.0079 | 0.9701 | 0.9691 |
| sorted_neighbourhood_blocker | 0.9744 | 0.9735 | 0.0009 | 0.9872 | 0.9867 |
| standard_blocker | 0.9699 | 0.9735 | -0.0035 | 0.9977 | 0.9985 |
| token_blocker | 0.9788 | 0.9788 | 0.0000 | 0.9932 | 0.9930 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8601 | 0.8950 | -0.0350 | 0.8814 | 0.8950 | 0.8601 | 0.8950 |
| ditto_plm | 0.8784 | 0.8699 | 0.0084 | 0.8601 | 0.8699 | 0.9221 | 0.8699 |
| llm_matcher | 0.8517 | 0.8848 | -0.0331 | 0.8652 | 0.8848 | 0.8517 | 0.8848 |
| magellan | 0.9045 | 0.8872 | 0.0173 | 0.8650 | 0.8872 | 0.9133 | 0.8872 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | comem | 0.8000 | 0.8649 | -0.0649 |
| forbes_dbpedia | ditto_plm | 0.8571 | 0.8750 | -0.0179 |
| forbes_dbpedia | llm_matcher | 0.8113 | 0.8673 | -0.0559 |
| forbes_dbpedia | magellan | 0.9037 | 0.9104 | -0.0067 |
| forbes_fullcontact | comem | 0.9202 | 0.9252 | -0.0050 |
| forbes_fullcontact | ditto_plm | 0.8996 | 0.8649 | 0.0347 |
| forbes_fullcontact | llm_matcher | 0.8920 | 0.9023 | -0.0103 |
| forbes_fullcontact | magellan | 0.9053 | 0.8640 | 0.0413 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.3754 | 0.5361 | -0.1607 |
| casefusion_only | 0.3689 | 0.4918 | -0.1230 |
| fusionquery_only | 0.3820 | 0.4590 | -0.0770 |
| llm_only | 0.3934 | 0.5082 | -0.1148 |
| ltm_only | 0.3770 | 0.5115 | -0.1344 |
| prefer_higher_trust_only | 0.3803 | 0.4967 | -0.1164 |
| pydi_per_attribute_optimal | 0.3885 | 0.5525 | -0.1639 |
| truthfinder_only | 0.3787 | 0.4607 | -0.0820 |
| voting_only | 0.3836 | 0.5148 | -0.1311 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.0114 | 0.2159 | -0.2045 | 0.0000 | 0.0568 | -0.0568 |
| city | 0.5692 | 0.7538 | -0.1846 | 0.0923 | 0.1231 | -0.0308 |
| country | 0.7000 | 0.8500 | -0.1500 | 0.1400 | 0.2900 | -0.1500 |
| founded | 0.7463 | 0.9254 | -0.1791 | 0.0896 | 0.1194 | -0.0299 |
| keypeople | 0.5000 | 0.7222 | -0.2222 | 0.1111 | 0.0000 | 0.1111 |
| name | 0.6600 | 0.8600 | -0.2000 | 0.0800 | 0.4600 | -0.3800 |
| revenue | 0.0833 | 0.2500 | -0.1667 | 0.0417 | 0.0556 | -0.0139 |
