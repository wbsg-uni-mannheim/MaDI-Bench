<!-- rendered with validate_variant._write_level_report_md from metrics.json: em_blocking / em_matching = June committee run, norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - papers / easy

_Generated at 2026-09-29T12:07:39.718652+00:00_

- domain: `papers`
- level: `easy`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_papers.yaml@466f940cff21`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_blocking=`em_blocking_committee_papers.yaml@4c879705ce30`, em_matching=`em_matching_committee_papers.yaml@3d4e6082ed0c`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8226 | 0.7503 | 0.0723 |
| norm | macro_f1 | 0.3497 | 0.3562 | -0.0064 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9575 | 0.9580 | -0.0005 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.9661 | 0.9664 | -0.0003 |
| fusion | overall_accuracy | 0.8222 | 0.8423 | -0.0201 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 1.0000 | 0.9091 | 0.0909 |
| duplicate_majority | 0.9254 | 0.9206 | 0.0047 |
| embedding_sbert | 0.6747 | 0.7059 | -0.0312 |
| instance_tf_cosine | 0.4310 | 0.4118 | 0.0193 |
| label_jw | 0.9231 | 0.5763 | 0.3468 |
| llm_openai | 0.9859 | 0.9851 | 0.0008 |
| magneto_slm_llm | 0.8182 | 0.7436 | 0.0746 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.6477 | 0.6694 | -0.0216 |
| passthrough | 0.4015 | 0.2984 | 0.1031 |
| rule_per_attribute_optimal | 0.0000 | 0.1008 | -0.1008 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9579 | 0.9583 | -0.0004 | 0.9997 | 0.9997 |
| embedding_blocker | 0.9018 | 0.9019 | -0.0000 | 0.9997 | 0.9997 |
| sc_block | 0.9933 | 0.9949 | -0.0016 | 0.9997 | 0.9997 |
| sorted_neighbourhood_blocker | 0.9770 | 0.9769 | 0.0001 | 0.9997 | 0.9997 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.9349 | 0.9318 | 0.0030 | 0.9349 | 0.9318 | 0.9349 | 0.9318 |
| ditto_plm | 0.9980 | 0.9988 | -0.0008 | 0.9934 | 0.9988 | 0.9925 | 0.9988 |
| llm_matcher | 0.9330 | 0.9354 | -0.0024 | 0.9330 | 0.9354 | 0.9330 | 0.9354 |
| magellan | 0.9986 | 0.9995 | -0.0009 | 0.9994 | 0.9995 | 0.9971 | 0.9995 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dblp_crossref | comem | 0.9793 | 0.9773 | 0.0020 |
| dblp_crossref | ditto_plm | 0.9976 | 0.9979 | -0.0003 |
| dblp_crossref | llm_matcher | 0.9771 | 0.9810 | -0.0040 |
| dblp_crossref | magellan | 0.9979 | 0.9994 | -0.0015 |
| dblp_open_alex | comem | 0.8905 | 0.8864 | 0.0041 |
| dblp_open_alex | ditto_plm | 0.9985 | 0.9997 | -0.0012 |
| dblp_open_alex | llm_matcher | 0.8890 | 0.8897 | -0.0007 |
| dblp_open_alex | magellan | 0.9994 | 0.9997 | -0.0003 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.7733 | 0.8503 | -0.0770 |
| casefusion_only | 0.7527 | 0.7451 | 0.0076 |
| fusionquery_only | 0.7527 | 0.7451 | 0.0076 |
| llm_only | 0.8297 | 0.8427 | -0.0130 |
| ltm_only | 0.7549 | 0.7777 | -0.0228 |
| prefer_higher_trust_only | 0.8677 | 0.8926 | -0.0249 |
| pydi_per_attribute_optimal | 0.8189 | 0.8265 | -0.0076 |
| truthfinder_only | 0.8080 | 0.8308 | -0.0228 |
| voting_only | 0.8037 | 0.8341 | -0.0304 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| authors | 0.8351 | 0.8454 | -0.0103 | 0.4845 | 0.4948 | -0.0103 |
| cited_by_count | 0.2667 | 0.2667 | 0.0000 | 0.0000 | 0.1333 | -0.1333 |
| first_page | 0.9889 | 0.9889 | 0.0000 | 0.0778 | 0.0778 | 0.0000 |
| issue | 0.8974 | 1.0000 | -0.1026 | 0.0128 | 0.0897 | -0.0769 |
| journal | 0.9438 | 1.0000 | -0.0562 | 0.7191 | 0.9663 | -0.2472 |
| last_page | 0.9886 | 1.0000 | -0.0114 | 0.0795 | 0.0795 | 0.0000 |
| publication_year | 0.9600 | 0.9600 | 0.0000 | 0.0300 | 0.0300 | 0.0000 |
| referenced_works_count | 0.7867 | 0.8267 | -0.0400 | 0.7333 | 0.7600 | -0.0267 |
| title | 0.7778 | 0.7778 | 0.0000 | 0.2121 | 0.2121 | 0.0000 |
| type | 0.9900 | 0.9900 | 0.0000 | 0.0900 | 0.0900 | 0.0000 |
| volume | 0.9890 | 1.0000 | -0.0110 | 0.0000 | 0.0000 | 0.0000 |
