<!-- rendered with validate_variant._write_level_report_md from metrics.json: norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - products / medium

_Generated at 2026-09-29T12:07:39.723085+00:00_

- domain: `products`
- level: `medium`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_products.yaml@38bda2792a80`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_matching=`em_matching_committee_products.yaml@466ca28df1c1`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7205 | 0.7174 | 0.0032 |
| norm | macro_f1 | 0.4698 | 0.4573 | 0.0125 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.7908 | 0.8969 | -0.1061 |
| fusion | overall_accuracy | 0.6976 | 0.7411 | -0.0435 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.8370 | 0.8696 | -0.0326 |
| duplicate_majority | 0.8166 | 0.8764 | -0.0598 |
| embedding_sbert | 0.4786 | 0.4343 | 0.0443 |
| instance_tf_cosine | 0.5366 | 0.5278 | 0.0088 |
| label_jw | 0.6292 | 0.5949 | 0.0343 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.7459 | 0.7188 | 0.0272 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.9072 | 0.8520 | 0.0552 |
| passthrough | 0.2986 | 0.3000 | -0.0014 |
| rule_per_attribute_optimal | 0.2036 | 0.2200 | -0.0164 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8455 | 0.8794 | -0.0338 | 0.8528 | 0.8794 | 0.8455 | 0.8794 |
| ditto_plm | 0.9073 | 0.9552 | -0.0479 | 0.9660 | 0.9552 | 0.8100 | 0.9552 |
| llm_matcher | 0.8649 | 0.8883 | -0.0233 | 0.8795 | 0.8883 | 0.8649 | 0.8883 |
| magellan | 0.5454 | 0.8647 | -0.3193 | 0.8565 | 0.8647 | 0.3576 | 0.8647 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.8500 | 0.9184 | -0.0684 |
| products_1_products_2 | ditto_plm | 0.9130 | 0.9804 | -0.0673 |
| products_1_products_2 | llm_matcher | 0.9048 | 0.9200 | -0.0152 |
| products_1_products_2 | magellan | 0.4918 | 0.8807 | -0.3889 |
| products_1_products_3 | comem | 0.8444 | 0.8261 | 0.0184 |
| products_1_products_3 | ditto_plm | 0.8727 | 0.9143 | -0.0416 |
| products_1_products_3 | llm_matcher | 0.8696 | 0.8632 | 0.0064 |
| products_1_products_3 | magellan | 0.5333 | 0.8929 | -0.3595 |
| products_1_products_4 | comem | 0.8421 | 0.8936 | -0.0515 |
| products_1_products_4 | ditto_plm | 0.9362 | 0.9709 | -0.0347 |
| products_1_products_4 | llm_matcher | 0.8205 | 0.8817 | -0.0612 |
| products_1_products_4 | magellan | 0.6111 | 0.8205 | -0.2094 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.6364 | 0.7787 | -0.1423 |
| casefusion_only | 0.6938 | 0.7632 | -0.0694 |
| fusionquery_only | 0.6962 | 0.7620 | -0.0658 |
| llm_only | 0.7644 | 0.7751 | -0.0108 |
| ltm_only | 0.7105 | 0.7536 | -0.0431 |
| prefer_higher_trust_only | 0.6017 | 0.7536 | -0.1519 |
| pydi_per_attribute_optimal | 0.7392 | 0.8002 | -0.0610 |
| truthfinder_only | 0.7010 | 0.7644 | -0.0634 |
| voting_only | 0.6208 | 0.7679 | -0.1471 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.9691 | 0.9691 | 0.0000 | 0.0825 | 0.0825 | 0.0000 |
| bus_type | 0.4423 | 0.4615 | -0.0192 | 0.1731 | 0.2115 | -0.0385 |
| chipset_name | 0.5200 | 0.5200 | 0.0000 | 0.1600 | 0.1200 | 0.0400 |
| color | 0.7692 | 0.8846 | -0.1154 | 0.0769 | 0.1538 | -0.0769 |
| form_factor | 0.4250 | 0.4250 | 0.0000 | 0.2750 | 0.3250 | -0.0500 |
| height_mm | 0.2857 | 0.7143 | -0.4286 | 0.0000 | 0.0476 | -0.0476 |
| interface_type | 0.1250 | 0.1250 | 0.0000 | 0.0833 | 0.0833 | 0.0000 |
| length_mm | 0.7273 | 0.8182 | -0.0909 | 0.0455 | 0.0455 | 0.0000 |
| memory_type | 1.0000 | 1.0000 | 0.0000 | 0.0400 | 0.0800 | -0.0400 |
| model | 0.8108 | 0.8378 | -0.0270 | 0.3243 | 0.3919 | -0.0676 |
| model_number | 0.9157 | 0.9518 | -0.0361 | 0.0723 | 0.0602 | 0.0120 |
| product_type | 1.0000 | 1.0000 | 0.0000 | 0.0300 | 0.0000 | 0.0300 |
| read_speed_mb_s | 0.7907 | 0.8837 | -0.0930 | 0.1395 | 0.0698 | 0.0698 |
| storage_connection_type | 0.8750 | 0.8500 | 0.0250 | 0.2500 | 0.3000 | -0.0500 |
| storage_gb | 0.9467 | 0.9600 | -0.0133 | 0.9067 | 0.0933 | 0.8133 |
| vram_gb | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| weight_g | 0.7500 | 0.7500 | 0.0000 | 0.0833 | 0.0833 | 0.0000 |
| width_mm | 0.4500 | 0.6500 | -0.2000 | 0.0500 | 0.0000 | 0.0500 |
| write_speed_mb_s | 0.8438 | 0.9375 | -0.0938 | 0.0938 | 0.0938 | 0.0000 |
