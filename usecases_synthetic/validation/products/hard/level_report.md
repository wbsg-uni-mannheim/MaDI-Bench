<!-- rendered with validate_variant._write_level_report_md from metrics.json: norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - products / hard

_Generated at 2026-09-29T12:07:39.724020+00:00_

- domain: `products`
- level: `hard`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_products.yaml@38bda2792a80`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_matching=`em_matching_committee_products.yaml@466ca28df1c1`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6566 | 0.7174 | -0.0608 |
| norm | macro_f1 | 0.4661 | 0.4573 | 0.0087 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8123 | 0.8969 | -0.0846 |
| fusion | overall_accuracy | 0.5621 | 0.7411 | -0.1790 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7614 | 0.8696 | -0.1082 |
| duplicate_majority | 0.8235 | 0.8764 | -0.0529 |
| embedding_sbert | 0.4651 | 0.4343 | 0.0308 |
| instance_tf_cosine | 0.5820 | 0.5278 | 0.0542 |
| label_jw | 0.3175 | 0.5949 | -0.2775 |
| llm_openai | 0.9800 | 1.0000 | -0.0200 |
| magneto_slm_llm | 0.6667 | 0.7188 | -0.0521 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.9130 | 0.8520 | 0.0610 |
| passthrough | 0.2998 | 0.3000 | -0.0002 |
| rule_per_attribute_optimal | 0.1854 | 0.2200 | -0.0346 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8641 | 0.8794 | -0.0152 | 0.8641 | 0.8794 | 0.8641 | 0.8794 |
| ditto_plm | 0.9242 | 0.9552 | -0.0310 | 0.9743 | 0.9552 | 0.9148 | 0.9552 |
| llm_matcher | 0.8040 | 0.8883 | -0.0843 | 0.8040 | 0.8883 | 0.8040 | 0.8883 |
| magellan | 0.6568 | 0.8647 | -0.2079 | 0.7912 | 0.8647 | 0.5529 | 0.8647 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.9412 | 0.9184 | 0.0228 |
| products_1_products_2 | ditto_plm | 0.9643 | 0.9804 | -0.0161 |
| products_1_products_2 | llm_matcher | 0.9020 | 0.9200 | -0.0180 |
| products_1_products_2 | magellan | 0.6462 | 0.8807 | -0.2346 |
| products_1_products_3 | comem | 0.8148 | 0.8261 | -0.0113 |
| products_1_products_3 | ditto_plm | 0.8852 | 0.9143 | -0.0290 |
| products_1_products_3 | llm_matcher | 0.7692 | 0.8632 | -0.0939 |
| products_1_products_3 | magellan | 0.6575 | 0.8929 | -0.2353 |
| products_1_products_4 | comem | 0.8364 | 0.8936 | -0.0573 |
| products_1_products_4 | ditto_plm | 0.9231 | 0.9709 | -0.0478 |
| products_1_products_4 | llm_matcher | 0.7407 | 0.8817 | -0.1410 |
| products_1_products_4 | magellan | 0.6667 | 0.8205 | -0.1538 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.5538 | 0.7787 | -0.2249 |
| casefusion_only | 0.5885 | 0.7632 | -0.1746 |
| fusionquery_only | 0.6172 | 0.7620 | -0.1447 |
| llm_only | 0.6400 | 0.7751 | -0.1352 |
| ltm_only | 0.6280 | 0.7536 | -0.1256 |
| prefer_higher_trust_only | 0.5179 | 0.7536 | -0.2356 |
| pydi_per_attribute_optimal | 0.6112 | 0.8002 | -0.1890 |
| truthfinder_only | 0.6041 | 0.7644 | -0.1603 |
| voting_only | 0.5395 | 0.7679 | -0.2285 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.9485 | 0.9691 | -0.0206 | 0.2165 | 0.0825 | 0.1340 |
| bus_type | 0.4808 | 0.4615 | 0.0192 | 0.2308 | 0.2115 | 0.0192 |
| chipset_name | 0.4400 | 0.5200 | -0.0800 | 0.1600 | 0.1200 | 0.0400 |
| color | 0.6154 | 0.8846 | -0.2692 | 0.0769 | 0.1538 | -0.0769 |
| form_factor | 0.4250 | 0.4250 | 0.0000 | 0.2250 | 0.3250 | -0.1000 |
| height_mm | 0.3810 | 0.7143 | -0.3333 | 0.0476 | 0.0476 | -0.0000 |
| interface_type | 0.1250 | 0.1250 | 0.0000 | 0.0000 | 0.0833 | -0.0833 |
| length_mm | 0.5909 | 0.8182 | -0.2273 | 0.0455 | 0.0455 | 0.0000 |
| memory_type | 0.9200 | 1.0000 | -0.0800 | 0.1200 | 0.0800 | 0.0400 |
| model | 0.6486 | 0.8378 | -0.1892 | 0.2297 | 0.3919 | -0.1622 |
| model_number | 0.7952 | 0.9518 | -0.1566 | 0.0723 | 0.0602 | 0.0120 |
| product_type | 0.9900 | 1.0000 | -0.0100 | 0.1400 | 0.0000 | 0.1400 |
| read_speed_mb_s | 0.3953 | 0.8837 | -0.4884 | 0.2093 | 0.0698 | 0.1395 |
| storage_connection_type | 0.7750 | 0.8500 | -0.0750 | 0.2000 | 0.3000 | -0.1000 |
| storage_gb | 0.8133 | 0.9600 | -0.1467 | 0.5067 | 0.0933 | 0.4133 |
| vram_gb | 0.8000 | 1.0000 | -0.2000 | 0.3600 | 0.0000 | 0.3600 |
| weight_g | 0.4167 | 0.7500 | -0.3333 | 0.0833 | 0.0833 | 0.0000 |
| width_mm | 0.2500 | 0.6500 | -0.4000 | 0.0000 | 0.0000 | 0.0000 |
| write_speed_mb_s | 0.4688 | 0.9375 | -0.4688 | 0.1562 | 0.0938 | 0.0625 |
