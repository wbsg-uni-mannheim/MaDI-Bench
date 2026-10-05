"""Source-name and record-id translation for the products base task.

products BASE ships three spellings of its four sources:
  * source data:       files dataset_1..4.json, raw record ids (12198483)
  * EM test gold:      raw ids
  * fusion/panel gold: 'products_<N>_<raw>' prefixed ids, sources products_N
  * SM gold:           source_dataset = 'products_<N>'

A system can only speak the visible convention (dataset_N names + raw ids),
so the SCORING layer translates: dataset names are canonicalized and
membership / fused record ids get the products_N_ prefix before fusion and
panel scoring. EM gold is raw, so EM needs no translation. The raw <->
prefixed mapping is one-to-one on this data. Every other task is untouched.
"""

from __future__ import annotations

import re

import pandas as pd

from .tasks import TaskSpec

# every spelling of the four products sources in circulation: the visible
# data files (dataset_N), the gold (products_N), the LLM pipeline (prodN)
_PRODUCTS_BASE_SOURCE_ALIASES = {
    f"{spelling}{n}": f"products_{n}"
    for n in range(1, 5) for spelling in ("dataset_", "products_", "prod")
}
_PRODUCTS_PREFIX = re.compile(r"^products_[1-4]_")


def canonical_source_name(task: TaskSpec, name: str) -> str:
    """Canonical dataset name for scoring (products base: products_<N>)."""
    if task.domain == "products" and task.tier == "base":
        return _PRODUCTS_BASE_SOURCE_ALIASES.get(str(name), str(name))
    return str(name)


def _prefix_record_id(source: str, record_id: str) -> str:
    """``products_<N>_<raw>``; an id that already carries a products prefix
    is left alone whatever source name came with it."""
    rid = str(record_id)
    if _PRODUCTS_PREFIX.match(rid):
        return rid
    return f"{source}_{rid}"


def normalize_membership(
    task: TaskSpec, membership: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Rewrite membership (record_id, source, cluster_id) into the gold
    currency. No-op except products base."""
    if not (task.domain == "products" and task.tier == "base") or membership is None:
        return membership
    out = membership.copy()
    out["source"] = out["source"].map(lambda s: canonical_source_name(task, s))
    out["record_id"] = [
        _prefix_record_id(source, record_id)
        for source, record_id in zip(out["source"], out["record_id"])
    ]
    return out


def normalize_source_ids(
    task: TaskSpec, frame: pd.DataFrame, name: str, id_column: str = "id"
) -> pd.DataFrame:
    """Rewrite a SOURCE table's id column into the gold currency (products
    base only; every other task returns the frame untouched).

    The e2e panel indexes source records by this column and looks them up
    with the membership record ids, which are products_N_-prefixed on both
    sides; without the prefix no source value would be found. Missing ids
    stay missing.
    """
    if not (task.domain == "products" and task.tier == "base"):
        return frame
    if id_column not in frame.columns:
        return frame
    source = canonical_source_name(task, name)
    out = frame.copy()
    out[id_column] = [
        rid if rid is None or (isinstance(rid, float) and pd.isna(rid))
        else _prefix_record_id(source, rid)
        for rid in out[id_column]
    ]
    return out


def normalize_fused_sources(task: TaskSpec, fused: pd.DataFrame) -> pd.DataFrame:
    """Rewrite the fused frame's _fusion_sources/_fusion_source_datasets into
    the gold currency (products base only)."""
    if not (task.domain == "products" and task.tier == "base"):
        return fused
    if "_fusion_sources" not in fused.columns:
        return fused
    from .panel_gold import _parse_id_list

    out = fused.copy()
    new_sources = []
    new_datasets = []
    for sources, datasets in zip(
        out["_fusion_sources"], out.get("_fusion_source_datasets", [None] * len(out))
    ):
        ids = _parse_id_list(sources)
        names = [
            canonical_source_name(task, d) for d in (_parse_id_list(datasets) or [])
        ]
        if len(names) < len(ids):
            names += [""] * (len(ids) - len(names))
        new_sources.append([
            _prefix_record_id(name, rid) if name else rid
            for rid, name in zip(ids, names)
        ])
        new_datasets.append(names)
    out["_fusion_sources"] = new_sources
    if "_fusion_source_datasets" in out.columns:
        out["_fusion_source_datasets"] = new_datasets
    return out
