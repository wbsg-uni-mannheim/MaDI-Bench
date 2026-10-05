# Origin

This folder holds the Ditto training and inference stack used by the entity-matching committee and the
best-of-breed pipeline (P2).

- Upstream: [megagonlabs/ditto](https://github.com/megagonlabs/ditto) (Li et al., "Deep Entity Matching with
  Pre-Trained Language Models", PVLDB 14(1), 2020), licensed under the Apache License, Version 2.0. A copy of
  the license is in [LICENSE](LICENSE).
- Adapted from upstream and modified for this repository: `augment.py` (data augmentation operators of
  `ditto_light/augment.py`), `knowledge.py` (domain-knowledge injection of `ditto_light/knowledge.py`) and
  `summarize.py` (TF-IDF summarizer of `ditto_light/summarize.py`). Each of these files carries a notice.
- Written for this repository following Ditto's design: `data.py`, `model.py`, `trainer.py`, `runtime.py`,
  `metrics.py`, `pseudolabels.py` and `__init__.py`.
