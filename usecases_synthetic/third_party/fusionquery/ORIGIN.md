# Vendored from upstream FusionQuery

## Source

- Upstream repo: https://github.com/JunHao-Zhu/FusionQuery
- Upstream commit: `3b8b4db184aea88da1971b7b4d77a4ae4ce88594` (2025-03-11)
- Upstream license: Apache License, Version 2.0 — verbatim copy at [LICENSE](LICENSE)
- Vendored on: 2026-04-23
- Vendored for: the fusion committee of the variant generator

## What was copied

| Vendored path | Upstream path | Lines | Modifications |
|---|---|---|---|
| [baseline.py](baseline.py) | `fusion/baseline.py` | 361 | Verbatim — no edits |
| [fusion.py](fusion.py) | `fusion/fusion.py` | 112 | Verbatim — no edits |
| [LICENSE](LICENSE) | `LICENSE` | 202 | Verbatim |
| [`__init__.py`](__init__.py) | — | 0 | Empty marker, written fresh |

Both `.py` files are self-contained (only import `numpy`, `random`, `collections`, `warnings`) and are imported as
`usecases_synthetic.third_party.fusionquery.baseline` / `usecases_synthetic.third_party.fusionquery.fusion`. No
intra-package import rewrites were required because the upstream files do not import from sibling modules.

## What was deliberately excluded

The upstream repo's matching subsystem and training driver are out of scope for the fusion committee: the
`sentence-transformers` / FAISS dependencies are stripped and only the numpy core is kept.

| Upstream path | Reason for exclusion |
|---|---|
| `FusionQuery/framework.py` | Orchestrator that ties matching + fusion. PyDI's `DataFusionEngine` already plays this role. |
| `query/graph.py`, `query/linegraph_match.py` | Graph-matching for the on-demand query interface. Not used in PyDI's record-group fusion. |
| `utils/statistic.py`, `utils/utility.py` | I/O helpers for the upstream demo dataset. Not needed for adapter use. |
| `main.py` | Upstream training/eval driver. Not needed when adapters call the classes directly. |
| `data/` | Demo data. Not needed. |
| `requirements.txt` | Upstream pins (`sentence-transformers`, `faiss-gpu`, `torch`, etc.). The vendored numpy core only requires `numpy` from the existing PyDI baseline; nothing new added to `pyproject.toml`. |

## Classes exposed

| Class | File | Adapter file |
|---|---|---|
| `MajorityVoter` | baseline.py | (used internally by `CASEFusion`) |
| `DARTFusion` | baseline.py | not adapted — paradigm overlap with `TruthFinder` |
| `TruthFinder` | baseline.py | [`usecases_synthetic/lib/td_batch_fusion.py`](../../lib/td_batch_fusion.py) |
| `CASEFusion` | baseline.py | [`usecases_synthetic/lib/td_batch_fusion.py`](../../lib/td_batch_fusion.py) |
| `LTMFusion` | baseline.py | [`usecases_synthetic/lib/td_batch_fusion.py`](../../lib/td_batch_fusion.py) |
| `EMFusioner` | fusion.py | [`usecases_synthetic/lib/td_batch_fusion.py`](../../lib/td_batch_fusion.py) |

`td_batch_fusion.py` fits each upstream method once per attribute on the full attribute corpus and
exposes the winning values through PyDI's per-cell `ConflictResolutionFunction` contract
(`(values, **kwargs) -> (value, confidence, metadata)`).
