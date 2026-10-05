<!-- rendered from baseline_metrics.json with measure_baseline's write_report_md: em_blocking / em_matching from the June committee run, norm = normalization test-set accuracies (macro_f1 aliases hold accuracies) -->
# Baseline report - games

_Generated at 2026-09-29T12:07:39.712262+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 1.0000 |
| best_member_name | llm_openai |
| macro_f1 | 0.6960 |
| macro_precision | 0.8255 |
| macro_recall | 0.6429 |
| max_f1 | 1.0000 |
| min_f1 | 0.1875 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| dbpedia.franchise | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.genre | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| dbpedia.launch_yr | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.studio | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.system | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.title | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dbpedia.wiki_ref | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.age_rating | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.console | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.game_title | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.genres | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| metacritic.made_by | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.mc_id | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.player_rating | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.press_rating | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| metacritic.year_published | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.age_classification | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.comm_rating | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.dist | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.genre | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| sales.hw | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.launch_dt | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.press_score | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| sales.prod_title | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.rec_id | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| sales.studio | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## Stage: norm

### NORM - aggregated
| metric | value |
|---|---|
| best_member_accuracy | 0.1209 |
| best_member_f1 | 0.1209 |
| best_member_name | llm_only |
| macro_f1 | 0.0806 |
| max_accuracy | 0.1209 |
| max_accuracy_transformations | 0.0643 |
| max_f1 | 0.1209 |
| mean_accuracy | 0.0806 |
| mean_accuracy_transformations | 0.0214 |
| min_accuracy | 0.0604 |
| min_f1 | 0.0604 |
| n_cells | 182.0000 |
| n_members | 3.0000 |

### NORM - per attribute
| attribute | best_member_accuracy | llm_only | n_cells | passthrough | rule_per_attribute_optimal |
|---|---|---|---|---|---|
| ESRB | 0.8333 | 0.8333 | 6.0000 | 0.8333 | 0.8333 |
| genres | 0.0000 | 0.0000 | 109.0000 | 0.0000 | 0.0000 |
| name | 0.6250 | 0.6250 | 8.0000 | 0.5000 | 0.5000 |
| platform | 0.2034 | 0.2034 | 59.0000 | 0.0339 | 0.0339 |

## Stage: em_blocking

### EM_BLOCKING - aggregated
| metric | value |
|---|---|
| best_member_name | standard_blocker |
| best_member_pair_recall | 0.9858 |
| best_member_reduction_ratio | 0.9988 |
| macro_pair_recall | 0.9560 |
| macro_pair_recall_baseline_model_on_baseline_test | 0.9560 |
| macro_pair_recall_baseline_model_on_regen_test | 0.9560 |
| macro_pair_recall_variant_model_on_baseline_test | 0.9560 |
| macro_pair_recall_variant_model_on_regen_test | 0.9560 |
| macro_reduction_ratio | 0.9899 |
| max_pair_recall | 1.0000 |
| min_pair_recall | 0.9155 |
| recall_floor | 0.9700 |

## Stage: em_matching

### EM_MATCHING - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.7155 |
| best_member_name | ditto_plm |
| macro_f1 | 0.6093 |
| macro_f1_baseline_model_on_baseline_test | 0.6093 |
| macro_f1_baseline_model_on_regen_test | 0.6093 |
| macro_f1_baseline_test | 0.6093 |
| macro_f1_regen_test | 0.6093 |
| macro_f1_variant_model_on_baseline_test | 0.6093 |
| macro_f1_variant_model_on_regen_test | 0.6093 |
| macro_precision | 0.8668 |
| macro_recall | 0.4883 |
| max_f1 | 0.7155 |
| min_f1 | 0.5092 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| best_member_macro_accuracy | 0.6948 |
| macro_accuracy | 0.6638 |
| max_accuracy | 0.6948 |
| min_accuracy | 0.6232 |
| overall_accuracy | 0.6948 |

### FUSION - per attribute
| attribute | accusim_only | best_member_accuracy | casefusion_only | fusionquery_only | llm_only | ltm_only | mean_member_accuracy | prefer_higher_trust_only | pydi_per_attribute_optimal | truthfinder_only | voting_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ESRB | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 |
| criticScore | 0.3733 | 0.3733 | 0.3733 | 0.3733 | 0.3733 | 0.3733 | 0.3733 | 0.3733 | 0.3733 | 0.3733 | 0.3733 |
| developer | 0.9263 | 0.9263 | 0.7684 | 0.9263 | 0.7684 | 0.8947 | 0.8678 | 0.9158 | 0.7789 | 0.9158 | 0.9158 |
| genres | 0.1300 | 0.7400 | 0.7400 | 0.7400 | 0.6800 | 0.4900 | 0.5589 | 0.6400 | 0.7400 | 0.7400 | 0.1300 |
| name | 0.9100 | 0.9700 | 0.8900 | 0.9600 | 0.9600 | 0.9200 | 0.9300 | 0.9700 | 0.9000 | 0.9600 | 0.9000 |
| platform | 0.2700 | 0.4400 | 0.2800 | 0.2700 | 0.4400 | 0.2700 | 0.2878 | 0.2600 | 0.2700 | 0.2600 | 0.2700 |
| publisher | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 |
| releaseYear | 0.9200 | 0.9300 | 0.9000 | 0.8900 | 0.5300 | 0.8300 | 0.8522 | 0.9300 | 0.8900 | 0.8900 | 0.8900 |
| userScore | 0.4000 | 0.4000 | 0.3636 | 0.3636 | 0.3636 | 0.3636 | 0.3737 | 0.3818 | 0.3636 | 0.3636 | 0.4000 |
