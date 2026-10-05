<!-- rendered from baseline_metrics.json with measure_baseline's write_report_md: em_blocking / em_matching from the June committee run, norm = normalization test-set accuracies (macro_f1 aliases hold accuracies) -->
# Baseline report - companies

_Generated at 2026-09-29T12:07:39.702663+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 1.0000 |
| best_member_name | llm_openai |
| macro_f1 | 0.6855 |
| macro_precision | 0.9235 |
| macro_recall | 0.6327 |
| max_f1 | 1.0000 |
| min_f1 | 0.1739 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| dbpedia.annual_income | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| dbpedia.entity_uri | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.established | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.headquarters | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.keypeople_name | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.nation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.org_name | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.sector | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.total_assets_val | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.asset_value | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.business_segment | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.company | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.forbes_url | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.region | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| forbes.sales_figure | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| fullcontact.Attribute_1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| fullcontact.Attribute_2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_4 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_5 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| fullcontact.Attribute_6 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

## Stage: norm

### NORM - aggregated
| metric | value |
|---|---|
| best_member_accuracy | 0.4348 |
| best_member_f1 | 0.4348 |
| best_member_name | llm_only |
| macro_f1 | 0.3587 |
| max_accuracy | 0.4348 |
| max_accuracy_transformations | 0.2000 |
| max_f1 | 0.4348 |
| mean_accuracy | 0.3587 |
| mean_accuracy_transformations | 0.1231 |
| min_accuracy | 0.2935 |
| min_f1 | 0.2935 |
| n_cells | 92.0000 |
| n_members | 3.0000 |

### NORM - per attribute
| attribute | best_member_accuracy | llm_only | n_cells | passthrough | rule_per_attribute_optimal |
|---|---|---|---|---|---|
| country | 0.8182 | 0.8182 | 33.0000 | 0.4545 | 0.6061 |
| industry | 0.1930 | 0.1930 | 57.0000 | 0.1930 | 0.1930 |
| name | 1.0000 | 1.0000 | 2.0000 | 0.5000 | 0.5000 |

## Stage: em_blocking

### EM_BLOCKING - aggregated
| metric | value |
|---|---|
| best_member_name | standard_blocker |
| best_member_pair_recall | 0.9735 |
| best_member_reduction_ratio | 0.9985 |
| macro_pair_recall | 0.9876 |
| macro_pair_recall_baseline_model_on_baseline_test | 0.9876 |
| macro_pair_recall_baseline_model_on_regen_test | 0.9876 |
| macro_pair_recall_variant_model_on_baseline_test | 0.9876 |
| macro_pair_recall_variant_model_on_regen_test | 0.9876 |
| macro_reduction_ratio | 0.9861 |
| max_pair_recall | 1.0000 |
| min_pair_recall | 0.9735 |
| recall_floor | 0.9700 |

## Stage: em_matching

### EM_MATCHING - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.8950 |
| best_member_name | comem |
| macro_f1 | 0.8842 |
| macro_f1_baseline_model_on_baseline_test | 0.8842 |
| macro_f1_baseline_model_on_regen_test | 0.8842 |
| macro_f1_baseline_test | 0.8842 |
| macro_f1_regen_test | 0.8842 |
| macro_f1_variant_model_on_baseline_test | 0.8842 |
| macro_f1_variant_model_on_regen_test | 0.8842 |
| macro_precision | 0.8879 |
| macro_recall | 0.8987 |
| max_f1 | 0.8950 |
| min_f1 | 0.8699 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| best_member_macro_accuracy | 0.5765 |
| macro_accuracy | 0.5343 |
| max_accuracy | 0.5765 |
| min_accuracy | 0.4989 |
| overall_accuracy | 0.5765 |

### FUSION - per attribute
| attribute | accusim_only | best_member_accuracy | casefusion_only | fusionquery_only | llm_only | ltm_only | mean_member_accuracy | prefer_higher_trust_only | pydi_per_attribute_optimal | truthfinder_only | voting_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| assets | 0.2159 | 0.2159 | 0.1705 | 0.1705 | 0.1591 | 0.1705 | 0.1793 | 0.1705 | 0.1705 | 0.1705 | 0.2159 |
| city | 0.6308 | 0.7538 | 0.6308 | 0.6923 | 0.6923 | 0.6923 | 0.6786 | 0.7077 | 0.7538 | 0.6769 | 0.6308 |
| country | 0.8100 | 0.8500 | 0.7800 | 0.8100 | 0.8500 | 0.8400 | 0.7800 | 0.5600 | 0.8200 | 0.7300 | 0.8200 |
| founded | 0.8507 | 0.9254 | 0.8806 | 0.8060 | 0.9254 | 0.8209 | 0.8624 | 0.8209 | 0.8955 | 0.8806 | 0.8806 |
| keypeople | 0.7222 | 0.7222 | 0.7222 | 0.7222 | 0.7222 | 0.7222 | 0.7222 | 0.7222 | 0.7222 | 0.7222 | 0.7222 |
| name | 0.8500 | 0.8600 | 0.6200 | 0.4000 | 0.6300 | 0.6800 | 0.6711 | 0.8600 | 0.8600 | 0.4500 | 0.6900 |
| revenue | 0.2361 | 0.2500 | 0.2500 | 0.2500 | 0.1944 | 0.2500 | 0.2407 | 0.2500 | 0.2500 | 0.2500 | 0.2361 |
