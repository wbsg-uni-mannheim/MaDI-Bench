<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - papers / medium

_Generated at 2026-09-29T12:07:39.719469+00:00_

- domain: `papers`
- level: `medium`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_papers.yaml@466f940cff21`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee_papers.yaml@4c879705ce30`, em_matching=`em_matching_committee_papers.yaml@3d4e6082ed0c`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7494 | 0.7503 | -0.0009 |
| norm | macro_f1 | 0.3566 | 0.3562 | 0.0004 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9539 | 0.9580 | -0.0041 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.9563 | 0.9664 | -0.0101 |
| fusion | overall_accuracy | 0.7774 | 0.8423 | -0.0650 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.8197 | 0.9091 | -0.0894 |
| duplicate_majority | 0.7586 | 0.9206 | -0.1620 |
| embedding_sbert | 0.5634 | 0.7059 | -0.1425 |
| instance_tf_cosine | 0.4815 | 0.4118 | 0.0697 |
| label_jw | 0.7937 | 0.5763 | 0.2174 |
| llm_openai | 1.0000 | 0.9851 | 0.0149 |
| magneto_slm_llm | 0.8293 | 0.7436 | 0.0857 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.6801 | 0.6694 | 0.0108 |
| passthrough | 0.3897 | 0.2984 | 0.0913 |
| rule_per_attribute_optimal | 0.0000 | 0.1008 | -0.1008 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9583 | 0.9583 | 0.0000 | 0.9997 | 0.9997 |
| embedding_blocker | 0.9016 | 0.9019 | -0.0003 | 0.9997 | 0.9997 |
| sc_block | 0.9895 | 0.9949 | -0.0054 | 0.9997 | 0.9997 |
| sorted_neighbourhood_blocker | 0.9661 | 0.9769 | -0.0108 | 0.9997 | 0.9997 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.9121 | 0.9318 | -0.0197 | 0.9121 | 0.9318 | 0.9121 | 0.9318 |
| ditto_plm | 0.9982 | 0.9988 | -0.0006 | 0.9868 | 0.9988 | 0.9868 | 0.9988 |
| llm_matcher | 0.9156 | 0.9354 | -0.0197 | 0.9156 | 0.9354 | 0.9156 | 0.9354 |
| magellan | 0.9992 | 0.9995 | -0.0003 | 0.9992 | 0.9995 | 0.9992 | 0.9995 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dblp_crossref | comem | 0.9542 | 0.9773 | -0.0231 |
| dblp_crossref | ditto_plm | 0.9979 | 0.9979 | -0.0000 |
| dblp_crossref | llm_matcher | 0.9558 | 0.9810 | -0.0252 |
| dblp_crossref | magellan | 0.9994 | 0.9994 | 0.0000 |
| dblp_open_alex | comem | 0.8701 | 0.8864 | -0.0162 |
| dblp_open_alex | ditto_plm | 0.9985 | 0.9997 | -0.0012 |
| dblp_open_alex | llm_matcher | 0.8755 | 0.8897 | -0.0142 |
| dblp_open_alex | magellan | 0.9991 | 0.9997 | -0.0006 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.7852 | 0.8503 | -0.0651 |
| casefusion_only | 0.7234 | 0.7451 | -0.0217 |
| fusionquery_only | 0.7299 | 0.7451 | -0.0152 |
| llm_only | 0.8200 | 0.8427 | -0.0228 |
| ltm_only | 0.7430 | 0.7777 | -0.0347 |
| prefer_higher_trust_only | 0.8351 | 0.8926 | -0.0575 |
| pydi_per_attribute_optimal | 0.7972 | 0.8265 | -0.0293 |
| truthfinder_only | 0.7863 | 0.8308 | -0.0445 |
| voting_only | 0.7809 | 0.8341 | -0.0531 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| authors | 0.7216 | 0.8454 | -0.1237 | 0.3711 | 0.4948 | -0.1237 |
| cited_by_count | 0.0667 | 0.2667 | -0.2000 | 0.0000 | 0.1333 | -0.1333 |
| first_page | 0.9444 | 0.9889 | -0.0444 | 0.0778 | 0.0778 | -0.0000 |
| issue | 0.8974 | 1.0000 | -0.1026 | 0.0128 | 0.0897 | -0.0769 |
| journal | 0.9213 | 1.0000 | -0.0787 | 0.7191 | 0.9663 | -0.2472 |
| last_page | 0.9545 | 1.0000 | -0.0455 | 0.0795 | 0.0795 | 0.0000 |
| publication_year | 0.9600 | 0.9600 | 0.0000 | 0.0300 | 0.0300 | 0.0000 |
| referenced_works_count | 0.7600 | 0.8267 | -0.0667 | 0.6267 | 0.7600 | -0.1333 |
| title | 0.7778 | 0.7778 | 0.0000 | 0.2121 | 0.2121 | 0.0000 |
| type | 0.9700 | 0.9900 | -0.0200 | 0.0700 | 0.0900 | -0.0200 |
| volume | 0.9890 | 1.0000 | -0.0110 | 0.0000 | 0.0000 | 0.0000 |
