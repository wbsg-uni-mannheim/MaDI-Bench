<!-- rendered from baseline_metrics.json with measure_baseline's write_report_md: em_blocking / em_matching from the June committee run, norm = normalization test-set accuracies (macro_f1 aliases hold accuracies) -->
# Baseline report - papers

_Generated at 2026-09-29T12:07:39.717993+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.9851 |
| best_member_name | llm_openai |
| macro_f1 | 0.7503 |
| macro_precision | 0.7559 |
| macro_recall | 0.7689 |
| max_f1 | 0.9851 |
| min_f1 | 0.4118 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| crossref.cited_total | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| crossref.container_title | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| crossref.contributor_names | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| crossref.id | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| crossref.issue_id | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| crossref.issued_year | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| crossref.page_first | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| crossref.page_last | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| crossref.reference_total | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| crossref.title_text | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| crossref.volume_id | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| crossref.work_type | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dblp.author_list | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dblp.entry_type | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| dblp.id | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| dblp.issue_no | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| dblp.page_finish | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| dblp.page_start | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dblp.pub_year | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dblp.publication_title | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| dblp.venue_name | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| dblp.volume_no | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| open_alex.authors_list | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| open_alex.citations_count | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| open_alex.display_title | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| open_alex.end_page | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| open_alex.id | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| open_alex.issue_tag | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| open_alex.refs_count | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| open_alex.source_name | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| open_alex.start_page | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| open_alex.volume_tag | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| open_alex.work_kind | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| open_alex.year_published | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

## Stage: norm

### NORM - aggregated
| metric | value |
|---|---|
| best_member_accuracy | 0.6694 |
| best_member_f1 | 0.6694 |
| best_member_name | llm_only |
| macro_f1 | 0.3562 |
| max_accuracy | 0.6694 |
| max_accuracy_transformations | 0.5287 |
| max_f1 | 0.6694 |
| mean_accuracy | 0.3562 |
| mean_accuracy_transformations | 0.1762 |
| min_accuracy | 0.1008 |
| min_f1 | 0.1008 |
| n_cells | 248.0000 |
| n_members | 3.0000 |

### NORM - per attribute
| attribute | best_member_accuracy | llm_only | n_cells | passthrough | rule_per_attribute_optimal |
|---|---|---|---|---|---|
| authors | 0.4464 | 0.4464 | 56.0000 | 0.4464 | 0.4464 |
| journal | 0.8571 | 0.8571 | 112.0000 | 0.2232 | 0.0000 |
| title | 0.5625 | 0.5625 | 80.0000 | 0.3000 | 0.0000 |

## Stage: em_blocking

### EM_BLOCKING - aggregated
| metric | value |
|---|---|
| best_member_name | sc_block |
| best_member_pair_recall | 0.9949 |
| best_member_reduction_ratio | 0.9997 |
| macro_pair_recall | 0.9580 |
| macro_pair_recall_baseline_model_on_baseline_test | 0.9580 |
| macro_pair_recall_baseline_model_on_regen_test | 0.9580 |
| macro_pair_recall_variant_model_on_baseline_test | 0.9580 |
| macro_pair_recall_variant_model_on_regen_test | 0.9580 |
| macro_reduction_ratio | 0.9997 |
| max_pair_recall | 0.9949 |
| min_pair_recall | 0.9019 |
| recall_floor | 0.9700 |

## Stage: em_matching

### EM_MATCHING - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.9995 |
| best_member_name | magellan |
| macro_f1 | 0.9664 |
| macro_f1_baseline_model_on_baseline_test | 0.9664 |
| macro_f1_baseline_model_on_regen_test | 0.9664 |
| macro_f1_baseline_test | 0.9664 |
| macro_f1_regen_test | 0.9664 |
| macro_f1_variant_model_on_baseline_test | 0.9664 |
| macro_f1_variant_model_on_regen_test | 0.9664 |
| macro_precision | 0.9997 |
| macro_recall | 0.9389 |
| max_f1 | 0.9995 |
| min_f1 | 0.9318 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| best_member_macro_accuracy | 0.8423 |
| macro_accuracy | 0.7611 |
| max_accuracy | 0.8423 |
| min_accuracy | 0.6861 |
| overall_accuracy | 0.8423 |

### FUSION - per attribute
| attribute | accusim_only | best_member_accuracy | casefusion_only | fusionquery_only | llm_only | ltm_only | mean_member_accuracy | prefer_higher_trust_only | pydi_per_attribute_optimal | truthfinder_only | voting_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| authors | 0.3608 | 0.8454 | 0.8454 | 0.8454 | 0.7113 | 0.6289 | 0.6976 | 0.8454 | 0.8454 | 0.8454 | 0.3505 |
| cited_by_count | 0.2000 | 0.2667 | 0.1333 | 0.1333 | 0.2667 | 0.1333 | 0.1704 | 0.2000 | 0.1333 | 0.1333 | 0.2000 |
| first_page | 0.9667 | 0.9889 | 0.9111 | 0.9111 | 0.9889 | 0.9111 | 0.9407 | 0.9667 | 0.9111 | 0.9111 | 0.9889 |
| issue | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9103 | 1.0000 | 0.9815 | 1.0000 | 0.9231 | 1.0000 | 1.0000 |
| journal | 1.0000 | 1.0000 | 0.0337 | 0.0337 | 1.0000 | 0.8652 | 0.7328 | 0.9438 | 0.9101 | 0.9888 | 0.8202 |
| last_page | 0.9773 | 1.0000 | 0.9205 | 0.9205 | 1.0000 | 0.9205 | 0.9508 | 0.9773 | 0.9205 | 0.9205 | 1.0000 |
| publication_year | 0.9600 | 0.9600 | 0.9600 | 0.9600 | 0.9400 | 0.9300 | 0.9522 | 0.9400 | 0.9600 | 0.9600 | 0.9600 |
| referenced_works_count | 0.8267 | 0.8267 | 0.0667 | 0.0667 | 0.0933 | 0.0667 | 0.3230 | 0.8267 | 0.0667 | 0.0667 | 0.8267 |
| title | 0.5859 | 0.7778 | 0.6869 | 0.6869 | 0.7778 | 0.5758 | 0.6543 | 0.5758 | 0.7172 | 0.7172 | 0.5657 |
| type | 0.9900 | 0.9900 | 0.9900 | 0.9900 | 0.9800 | 0.9000 | 0.9689 | 0.9900 | 0.9900 | 0.9000 | 0.9900 |
| volume | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
