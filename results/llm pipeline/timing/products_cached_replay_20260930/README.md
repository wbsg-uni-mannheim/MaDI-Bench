# P3 Products base: runtime

Wall time: **363.2 s** (return code 0). This is the P3 Products value in Table 9.

Protocol: the same as the other P3 timings in `../base_cached_strict_20260609/`
and `../papers_cached_alias_20260609/`:
`run_pipeline.py --training-set-construction active --fusion-case all --official-validation-fusion-only`
with `PYDI_LABELED_SET_CACHE_ONLY=1` and `PYDI_ACTIVE_LEARNING_CACHE_ONLY=1`, so
the LLM labels come from the cache of the run. The time is the wall-clock time of
the pipeline process.

Steps (s): FAISS 28.5, matcher optimization 60.6, fusion optimization 253.7,
labeling and active learning 0.0 (cache). The timed run selects the same fusion
configuration as the run in `../../products/baseline/`
(`heuristic_stats__opt_provided`). Its correspondences are identical for
products_1/products_3 and products_1/products_4 and have a Jaccard similarity of
0.81 for products_1/products_2, because the matcher optimization runs again. The
quality scores in the paper are those of `../../products/baseline/`.

Files: `pipeline_metrics.csv` (scores of the timed run), `console.log.gz` (its
log).
