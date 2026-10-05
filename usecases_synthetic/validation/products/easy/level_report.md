<!-- rendered with validate_variant._write_level_report_md from metrics.json: norm = normalization test-set accuracies (macro_f1 columns hold accuracies) -->
# Validation report - products / easy

_Generated at 2026-09-29T12:07:39.722114+00:00_

- domain: `products`
- level: `easy`
- with_llm: `True`
- committee_versions: fusion=`fusion_committee_products.yaml@38bda2792a80`, sm=`sm_committee.yaml@cb4a6847ac9e`, em_matching=`em_matching_committee_products.yaml@466ca28df1c1`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7906 | 0.7174 | 0.0732 |
| norm | macro_f1 | 0.4508 | 0.4573 | -0.0066 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8987 | 0.8969 | 0.0018 |
| fusion | overall_accuracy | 0.7405 | 0.7411 | -0.0007 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.9796 | 0.8696 | 0.1100 |
| duplicate_majority | 1.0000 | 0.8764 | 0.1236 |
| embedding_sbert | 0.4413 | 0.4343 | 0.0070 |
| instance_tf_cosine | 0.5371 | 0.5278 | 0.0093 |
| label_jw | 0.8136 | 0.5949 | 0.2186 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.7626 | 0.7188 | 0.0439 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.8665 | 0.8520 | 0.0145 |
| passthrough | 0.2983 | 0.3000 | -0.0017 |
| rule_per_attribute_optimal | 0.1875 | 0.2200 | -0.0325 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.9302 | 0.8794 | 0.0508 | 0.9302 | 0.8794 | 0.9302 | 0.8794 |
| ditto_plm | 0.9517 | 0.9552 | -0.0035 | 0.9636 | 0.9552 | 0.8873 | 0.9552 |
| llm_matcher | 0.9139 | 0.8883 | 0.0256 | 0.9139 | 0.8883 | 0.9139 | 0.8883 |
| magellan | 0.7991 | 0.8647 | -0.0656 | 0.8951 | 0.8647 | 0.4910 | 0.8647 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| products_1_products_2 | comem | 0.9333 | 0.9184 | 0.0150 |
| products_1_products_2 | ditto_plm | 0.9200 | 0.9804 | -0.0604 |
| products_1_products_2 | llm_matcher | 0.9333 | 0.9200 | 0.0133 |
| products_1_products_2 | magellan | 0.7500 | 0.8807 | -0.1307 |
| products_1_products_3 | comem | 0.9524 | 0.8261 | 0.1263 |
| products_1_products_3 | ditto_plm | 0.9767 | 0.9143 | 0.0625 |
| products_1_products_3 | llm_matcher | 0.9302 | 0.8632 | 0.0671 |
| products_1_products_3 | magellan | 0.8696 | 0.8929 | -0.0233 |
| products_1_products_4 | comem | 0.9048 | 0.8936 | 0.0111 |
| products_1_products_4 | ditto_plm | 0.9583 | 0.9709 | -0.0125 |
| products_1_products_4 | llm_matcher | 0.8780 | 0.8817 | -0.0037 |
| products_1_products_4 | magellan | 0.7778 | 0.8205 | -0.0427 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.7392 | 0.7787 | -0.0395 |
| casefusion_only | 0.7440 | 0.7632 | -0.0191 |
| fusionquery_only | 0.7488 | 0.7620 | -0.0132 |
| llm_only | 0.7895 | 0.7751 | 0.0144 |
| ltm_only | 0.7452 | 0.7536 | -0.0084 |
| prefer_higher_trust_only | 0.7333 | 0.7536 | -0.0203 |
| pydi_per_attribute_optimal | 0.7919 | 0.8002 | -0.0084 |
| truthfinder_only | 0.7476 | 0.7644 | -0.0167 |
| voting_only | 0.7440 | 0.7679 | -0.0239 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| brand | 0.9691 | 0.9691 | 0.0000 | 0.0619 | 0.0825 | -0.0206 |
| bus_type | 0.4423 | 0.4615 | -0.0192 | 0.1923 | 0.2115 | -0.0192 |
| chipset_name | 0.5200 | 0.5200 | 0.0000 | 0.1200 | 0.1200 | 0.0000 |
| color | 0.8462 | 0.8846 | -0.0385 | 0.0769 | 0.1538 | -0.0769 |
| form_factor | 0.4500 | 0.4250 | 0.0250 | 0.2250 | 0.3250 | -0.1000 |
| height_mm | 0.7143 | 0.7143 | 0.0000 | 0.0476 | 0.0476 | 0.0000 |
| interface_type | 0.1250 | 0.1250 | 0.0000 | 0.0833 | 0.0833 | 0.0000 |
| length_mm | 0.8182 | 0.8182 | 0.0000 | 0.0000 | 0.0455 | -0.0455 |
| memory_type | 1.0000 | 1.0000 | 0.0000 | 0.0400 | 0.0800 | -0.0400 |
| model | 0.8243 | 0.8378 | -0.0135 | 0.3514 | 0.3919 | -0.0405 |
| model_number | 0.9759 | 0.9518 | 0.0241 | 0.0602 | 0.0602 | 0.0000 |
| product_type | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| read_speed_mb_s | 0.8140 | 0.8837 | -0.0698 | 0.1628 | 0.0698 | 0.0930 |
| storage_connection_type | 0.8500 | 0.8500 | 0.0000 | 0.1750 | 0.3000 | -0.1250 |
| storage_gb | 0.9333 | 0.9600 | -0.0267 | 0.3200 | 0.0933 | 0.2267 |
| vram_gb | 1.0000 | 1.0000 | 0.0000 | 0.0400 | 0.0000 | 0.0400 |
| weight_g | 0.7500 | 0.7500 | 0.0000 | 0.0833 | 0.0833 | 0.0000 |
| width_mm | 0.6500 | 0.6500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| write_speed_mb_s | 0.8750 | 0.9375 | -0.0625 | 0.2188 | 0.0938 | 0.1250 |
