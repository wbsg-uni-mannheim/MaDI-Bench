<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - companies / easy

_Generated at 2026-09-29T12:07:39.703227+00:00_

- domain: `companies`
- level: `easy`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee.yaml@6f2bb9461525`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee.yaml@042119f4d1bf`, em_matching=`em_matching_committee.yaml@f0bb40e2173f`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8854 | 0.6855 | 0.1998 |
| norm | macro_f1 | 0.2030 | 0.3587 | -0.1557 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9829 | 0.9876 | -0.0047 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.9066 | 0.8842 | 0.0224 |
| fusion | overall_accuracy | 0.5113 | 0.5765 | -0.0652 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 1.0000 | 0.3846 | 0.6154 |
| duplicate_majority | 0.8000 | 0.8000 | 0.0000 |
| embedding_sbert | 0.8421 | 0.7619 | 0.0802 |
| instance_tf_cosine | 0.6667 | 0.7692 | -0.1026 |
| label_jw | 1.0000 | 0.1739 | 0.8261 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.8889 | 0.9091 | -0.0202 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.2099 | 0.4348 | -0.2249 |
| passthrough | 0.2222 | 0.2935 | -0.0713 |
| rule_per_attribute_optimal | 0.1770 | 0.3478 | -0.1709 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9921 | 1.0000 | -0.0079 | 0.9861 | 0.9846 |
| embedding_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9861 | 0.9846 |
| sc_block | 0.9956 | 1.0000 | -0.0044 | 0.9722 | 0.9691 |
| sorted_neighbourhood_blocker | 0.9655 | 0.9735 | -0.0079 | 0.9879 | 0.9867 |
| standard_blocker | 0.9655 | 0.9735 | -0.0079 | 0.9976 | 0.9985 |
| token_blocker | 0.9788 | 0.9788 | 0.0000 | 0.9924 | 0.9930 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8972 | 0.8950 | 0.0022 | 0.8972 | 0.8950 | 0.8972 | 0.8950 |
| ditto_plm | 0.9092 | 0.8699 | 0.0393 | 0.8747 | 0.8699 | 0.8747 | 0.8699 |
| llm_matcher | 0.8911 | 0.8848 | 0.0064 | 0.8911 | 0.8848 | 0.8911 | 0.8848 |
| magellan | 0.9288 | 0.8872 | 0.0416 | 0.9344 | 0.8872 | 0.9344 | 0.8872 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| forbes_dbpedia | comem | 0.8649 | 0.8649 | 0.0000 |
| forbes_dbpedia | ditto_plm | 0.9197 | 0.8750 | 0.0447 |
| forbes_dbpedia | llm_matcher | 0.8649 | 0.8673 | -0.0024 |
| forbes_dbpedia | magellan | 0.9767 | 0.9104 | 0.0663 |
| forbes_fullcontact | comem | 0.9296 | 0.9252 | 0.0043 |
| forbes_fullcontact | ditto_plm | 0.8988 | 0.8649 | 0.0339 |
| forbes_fullcontact | llm_matcher | 0.9174 | 0.9023 | 0.0151 |
| forbes_fullcontact | magellan | 0.8810 | 0.8640 | 0.0170 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.4738 | 0.5361 | -0.0623 |
| casefusion_only | 0.4262 | 0.4918 | -0.0656 |
| fusionquery_only | 0.4279 | 0.4590 | -0.0311 |
| llm_only | 0.4574 | 0.5082 | -0.0508 |
| ltm_only | 0.4393 | 0.5115 | -0.0721 |
| prefer_higher_trust_only | 0.4852 | 0.4967 | -0.0115 |
| pydi_per_attribute_optimal | 0.4967 | 0.5525 | -0.0557 |
| truthfinder_only | 0.4393 | 0.4607 | -0.0213 |
| voting_only | 0.4492 | 0.5148 | -0.0656 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| assets | 0.0114 | 0.2159 | -0.2045 | 0.0114 | 0.0568 | -0.0455 |
| city | 0.8308 | 0.7538 | 0.0769 | 0.3077 | 0.1231 | 0.1846 |
| country | 0.8800 | 0.8500 | 0.0300 | 0.1800 | 0.2900 | -0.1100 |
| founded | 0.9254 | 0.9254 | 0.0000 | 0.0746 | 0.1194 | -0.0448 |
| keypeople | 0.6667 | 0.7222 | -0.0556 | 0.0556 | 0.0000 | 0.0556 |
| name | 0.8600 | 0.8600 | 0.0000 | 0.4300 | 0.4600 | -0.0300 |
| revenue | 0.0139 | 0.2500 | -0.2361 | 0.0139 | 0.0556 | -0.0417 |
