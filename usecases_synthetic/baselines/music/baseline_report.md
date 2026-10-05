<!-- rendered from baseline_metrics.json with measure_baseline's write_report_md: em_blocking / em_matching from the June committee run, norm = normalization test-set accuracies (macro_f1 aliases hold accuracies) -->
# Baseline report - music

_Generated at 2026-09-29T12:07:39.715150+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 1.0000 |
| best_member_name | llm_openai |
| macro_f1 | 0.6696 |
| macro_precision | 0.9422 |
| macro_recall | 0.6190 |
| max_f1 | 1.0000 |
| min_f1 | 0.0909 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| discogs.category | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.duration | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.imprint | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.origin_loc | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.performer | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.pub_dt | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.rec_uid | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.title_str | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.tracks_track-name | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| lastfm.album_length | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| lastfm.album_title | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| lastfm.band | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| lastfm.item_code | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| lastfm.tracks_track-name | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_1 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_4 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_5 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_6 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| musicbrainz.Attribute_9 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

## Stage: norm

### NORM - aggregated
| metric | value |
|---|---|
| best_member_accuracy | 0.6125 |
| best_member_f1 | 0.6125 |
| best_member_name | llm_only |
| macro_f1 | 0.4875 |
| max_accuracy | 0.6125 |
| max_accuracy_transformations | 0.5714 |
| max_f1 | 0.6125 |
| mean_accuracy | 0.4875 |
| mean_accuracy_transformations | 0.3750 |
| min_accuracy | 0.3000 |
| min_f1 | 0.3000 |
| n_cells | 80.0000 |
| n_members | 3.0000 |

### NORM - per attribute
| attribute | best_member_accuracy | llm_only | n_cells | passthrough | rule_per_attribute_optimal |
|---|---|---|---|---|---|
| genre | 0.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 |
| label | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| name | 0.5385 | 0.5385 | 13.0000 | 0.3846 | 0.0000 |
| release-country | 0.5000 | 0.0000 | 12.0000 | 0.5000 | 0.0000 |
| release-date | 1.0000 | 0.9545 | 44.0000 | 0.2955 | 1.0000 |
| tracks | 0.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 |

## Stage: em_blocking

### EM_BLOCKING - aggregated
| metric | value |
|---|---|
| best_member_name | embedding_blocker |
| best_member_pair_recall | 0.9970 |
| best_member_reduction_ratio | 0.9964 |
| macro_pair_recall | 0.9505 |
| macro_pair_recall_baseline_model_on_baseline_test | 0.9505 |
| macro_pair_recall_baseline_model_on_regen_test | 0.9505 |
| macro_pair_recall_variant_model_on_baseline_test | 0.9505 |
| macro_pair_recall_variant_model_on_regen_test | 0.9505 |
| macro_reduction_ratio | 0.9789 |
| max_pair_recall | 1.0000 |
| min_pair_recall | 0.8498 |
| recall_floor | 0.9700 |

## Stage: em_matching

### EM_MATCHING - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.9531 |
| best_member_name | magellan |
| macro_f1 | 0.8242 |
| macro_f1_baseline_model_on_baseline_test | 0.8242 |
| macro_f1_baseline_model_on_regen_test | 0.8242 |
| macro_f1_baseline_test | 0.8242 |
| macro_f1_regen_test | 0.8242 |
| macro_f1_variant_model_on_baseline_test | 0.8242 |
| macro_f1_variant_model_on_regen_test | 0.8242 |
| macro_precision | 0.9952 |
| macro_recall | 0.7354 |
| max_f1 | 0.9531 |
| min_f1 | 0.6914 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| best_member_macro_accuracy | 0.7511 |
| macro_accuracy | 0.7264 |
| max_accuracy | 0.7511 |
| min_accuracy | 0.6884 |
| overall_accuracy | 0.7511 |

### FUSION - per attribute
| attribute | accusim_only | best_member_accuracy | casefusion_only | fusionquery_only | llm_only | ltm_only | mean_member_accuracy | prefer_higher_trust_only | pydi_per_attribute_optimal | truthfinder_only | voting_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| artist | 0.9400 | 0.9600 | 0.9200 | 0.7400 | 0.9600 | 0.8600 | 0.8967 | 0.9200 | 0.9100 | 0.9100 | 0.9100 |
| duration | 0.8608 | 0.8608 | 0.7848 | 0.7848 | 0.7595 | 0.7848 | 0.7918 | 0.7848 | 0.7848 | 0.7848 | 0.7975 |
| genre | 0.4267 | 0.4267 | 0.4267 | 0.4267 | 0.4267 | 0.4267 | 0.4267 | 0.4267 | 0.4267 | 0.4267 | 0.4267 |
| label | 0.9432 | 0.9659 | 0.9432 | 0.9432 | 0.9659 | 0.9432 | 0.9533 | 0.9659 | 0.9659 | 0.9432 | 0.9659 |
| name | 0.9200 | 0.9300 | 0.9200 | 0.8000 | 0.9300 | 0.9200 | 0.9111 | 0.9200 | 0.9300 | 0.9300 | 0.9300 |
| release-country | 0.9286 | 0.9796 | 0.6531 | 0.8878 | 0.2755 | 0.8980 | 0.8265 | 0.9796 | 0.9286 | 0.9592 | 0.9286 |
| release-date | 0.9896 | 0.9896 | 0.9792 | 0.9896 | 0.9896 | 0.9479 | 0.9826 | 0.9792 | 0.9896 | 0.9896 | 0.9896 |
| tracks | 0.0000 | 0.2000 | 0.0000 | 0.0000 | 0.2000 | 0.0000 | 0.0222 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
