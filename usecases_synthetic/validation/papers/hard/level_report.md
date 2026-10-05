<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - papers / hard

_Generated at 2026-09-29T12:07:39.720259+00:00_

- domain: `papers`
- level: `hard`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_papers.yaml@466f940cff21`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee_papers.yaml@4c879705ce30`, em_matching=`em_matching_committee_papers.yaml@3d4e6082ed0c`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6291 | 0.7503 | -0.1212 |
| norm | macro_f1 | 0.3681 | 0.3562 | 0.0119 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9144 | 0.9580 | -0.0436 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.9159 | 0.9664 | -0.0505 |
| fusion | overall_accuracy | 0.6833 | 0.8423 | -0.1590 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.5882 | 0.9091 | -0.3209 |
| duplicate_majority | 0.7586 | 0.9206 | -0.1620 |
| embedding_sbert | 0.4262 | 0.7059 | -0.2797 |
| instance_tf_cosine | 0.4677 | 0.4118 | 0.0560 |
| label_jw | 0.4400 | 0.5763 | -0.1363 |
| llm_openai | 0.9859 | 0.9851 | 0.0008 |
| magneto_slm_llm | 0.7368 | 0.7436 | -0.0067 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.7031 | 0.6694 | 0.0338 |
| passthrough | 0.3906 | 0.2984 | 0.0922 |
| rule_per_attribute_optimal | 0.0104 | 0.1008 | -0.0904 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9175 | 0.9583 | -0.0408 | 0.9995 | 0.9997 |
| embedding_blocker | 0.8761 | 0.9019 | -0.0257 | 0.9995 | 0.9997 |
| sc_block | 0.9513 | 0.9949 | -0.0436 | 0.9995 | 0.9997 |
| sorted_neighbourhood_blocker | 0.9128 | 0.9769 | -0.0641 | 0.9996 | 0.9997 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8453 | 0.9318 | -0.0865 | 0.8555 | 0.9318 | 0.8453 | 0.9318 |
| ditto_plm | 0.9874 | 0.9988 | -0.0114 | 0.9790 | 0.9988 | 0.9811 | 0.9988 |
| llm_matcher | 0.8500 | 0.9354 | -0.0854 | 0.8614 | 0.9354 | 0.8500 | 0.9354 |
| magellan | 0.9809 | 0.9995 | -0.0186 | 0.9940 | 0.9995 | 0.9805 | 0.9995 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dblp_crossref | comem | 0.8806 | 0.9773 | -0.0967 |
| dblp_crossref | ditto_plm | 0.9905 | 0.9979 | -0.0074 |
| dblp_crossref | llm_matcher | 0.8798 | 0.9810 | -0.1013 |
| dblp_crossref | magellan | 0.9809 | 0.9994 | -0.0185 |
| dblp_open_alex | comem | 0.8100 | 0.8864 | -0.0764 |
| dblp_open_alex | ditto_plm | 0.9844 | 0.9997 | -0.0153 |
| dblp_open_alex | llm_matcher | 0.8201 | 0.8897 | -0.0696 |
| dblp_open_alex | magellan | 0.9810 | 0.9997 | -0.0187 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.7039 | 0.8503 | -0.1464 |
| casefusion_only | 0.6464 | 0.7451 | -0.0987 |
| fusionquery_only | 0.6855 | 0.7451 | -0.0597 |
| llm_only | 0.7343 | 0.8427 | -0.1085 |
| ltm_only | 0.6486 | 0.7777 | -0.1291 |
| prefer_higher_trust_only | 0.7082 | 0.8926 | -0.1844 |
| pydi_per_attribute_optimal | 0.6898 | 0.8265 | -0.1367 |
| truthfinder_only | 0.6779 | 0.8308 | -0.1529 |
| voting_only | 0.6844 | 0.8341 | -0.1497 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| authors | 0.6186 | 0.8454 | -0.2268 | 0.2680 | 0.4948 | -0.2268 |
| cited_by_count | 0.2000 | 0.2667 | -0.0667 | 0.0000 | 0.1333 | -0.1333 |
| first_page | 0.7778 | 0.9889 | -0.2111 | 0.0889 | 0.0778 | 0.0111 |
| issue | 0.6795 | 1.0000 | -0.3205 | 0.0256 | 0.0897 | -0.0641 |
| journal | 0.8202 | 1.0000 | -0.1798 | 0.4045 | 0.9663 | -0.5618 |
| last_page | 0.8068 | 1.0000 | -0.1932 | 0.0795 | 0.0795 | -0.0000 |
| publication_year | 0.9200 | 0.9600 | -0.0400 | 0.1100 | 0.0300 | 0.0800 |
| referenced_works_count | 0.6000 | 0.8267 | -0.2267 | 0.3333 | 0.7600 | -0.4267 |
| title | 0.7071 | 0.7778 | -0.0707 | 0.1414 | 0.2121 | -0.0707 |
| type | 0.9500 | 0.9900 | -0.0400 | 0.0300 | 0.0900 | -0.0600 |
| volume | 0.7912 | 1.0000 | -0.2088 | 0.0330 | 0.0000 | 0.0330 |
