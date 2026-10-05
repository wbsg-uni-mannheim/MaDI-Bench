"""Schema-matching view of a task, read exactly as shipped.

Why this exists
-------------------------------------------
The committee / pipeline bundles (:func:`variant_loader.load_variant`,
:mod:`pipelines.lib.canonical_loader`) rename source columns before any
stage sees them: every configured ``id_column`` becomes ``id``
(``loaders.load_source``), papers maps every raw column onto its target
name and pads the missing target columns with all-NA
(``loaders.normalize_loaded_source``), and the products base bundle of the
committee reads the generator's own copy of the task. That is what the
normalization, EM and fusion stages expect, but it hands every schema
matcher free correspondences (``id`` -> ``id``, papers ``title`` ->
``title``, ...) and so inflates the schema-matching scores.

The SM stage therefore reads its own view of the task:

* **sources** -- the shipped files with their on-disk column names. Base
  task: the files the SM gold declares in ``source_files`` (falling back to
  the domain config ``file``); variants: ``input/data/<source>.csv``. Read
  with the same PyDI readers as the bundles (dtype inference unchanged),
  minus the id-minting reader options (``add_index`` /
  ``index_column_name`` / ``id_prefix``), so no provenance id column is
  recorded and a shipped ``id`` column stays visible to
  ``PyDI.schemamatching.base.get_schema_columns``. Afterwards only three
  steps: list/dict cells become their ``str()`` form (the matchers crash
  on list cells), the products ``cluster_id`` column (the
  WDC gold grouping; the SM gold says it "is dropped before schema
  matching") is dropped at every tier, and ``attrs["dataset_name"]`` is
  set. No id rename, no papers rename or NA padding, no id prefixing, no
  discogs ``duration`` 0 -> NA coalesce.
* **gold** -- the shipped SM gold as is: ``sm_mapping_gold.json`` (base,
  raw on-disk names, positives only) or ``sm_mapping.csv`` (variants, K8
  names). No reconciliation into the renamed space.
* **target schema** and **fusion frames** (the target instance values) --
  the task's own ``target_schema.json`` and fusion val + test files, as
  read, without the anchor ``id`` column the products bundles insert.
* **EM gold** for the duplicate-based member -- the task's own
  ``input/entitymatching`` files (products base: ``prod1_to_prod<m>_*.csv``
  with bare ids), in the id space of the view's sources.

Every existing bundle field stays untouched; the bundle only carries
``sm_task_root``, and the view is loaded lazily inside the SM stage.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from PyDI.io import load_csv, load_json, load_xml

from .domain_config import (
    DomainConfig,
    SourceSpec,
    _resolve_knob_config_alias,
    load_domain_config,
)
from .loaders import _strip_xml_namespaces
from .variant_loader import (
    _load_em_gold,
    _load_fusion,
    _load_fusion_file,
    _load_sm_mapping,
    _load_target_schema,
)

logger = logging.getLogger(__name__)

#: Provenance marker written into every SM member's notes when the member
#: read the shipped task through this view.
SM_INPUT_TAG = "shipped_raw_20260926"
#: Marker for the legacy path (a hand-built bundle without ``sm_task_root``).
SM_INPUT_TAG_LEGACY = "bundle_legacy"

#: Source columns removed from the SM view, per (canonical) domain. products
#: ``cluster_id`` is the WDC gold grouping: never a matcher feature.
_SM_DROPPED_SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {"products": ("cluster_id",)}

#: Reader options that mint / register an id column; dropped for the view.
_ID_MINTING_READER_KWARGS: tuple[str, ...] = ("add_index", "index_column_name", "id_prefix")

#: Fusion file names tried for a base task, in order, after the domain
#: config's ``fusion_files`` (the products base task ships CSVs).
_BASE_FUSION_FALLBACKS: tuple[tuple[str, str], ...] = (
    ("test_set.xml", "validation_set.xml"),
    ("fusion_test_set.csv", "fusion_validation_set.csv"),
)


@dataclass
class SMView:
    """What the schema-matching stage reads for one task.

    Parameters
    ----------
    domain, level : str
        Task identity (``level`` is ``"baseline"`` for the base task).
    task_root : Path
        The task directory the view was read from.
    sources : dict[str, DataFrame]
        Source frames with their shipped column names.
    gold : DataFrame or None
        Shipped SM gold (``source_dataset``, ``source_column``,
        ``target_dataset``, ``target_column``, ``score``), positives only.
    target_schema : dict
        Parsed ``target_schema.json``.
    fusion_frames : list[DataFrame]
        Fusion validation + test frames (validation first), as read; the
        target reference values of instance-based matchers.
    em_gold : dict[tuple[str, str], DataFrame]
        EM gold (``id1``, ``id2``, ``label``) per declared source pair, ids
        in the id space of ``sources``.
    source_pairs : list[tuple[str, str]]
        The declared source pairs (duplicate-based member).
    record_id_columns : dict[str, str | None]
        Per source, the column that holds the record id (``wiki_ref``,
        ``id``, ...), used as the duplicate-based member's id hint.
    input_tag : str
        :data:`SM_INPUT_TAG` or :data:`SM_INPUT_TAG_LEGACY`.
    source_files : dict[str, Path]
        The file each source was read from (empty on the legacy path).
    """

    domain: str
    level: str
    task_root: Path | None
    sources: dict[str, pd.DataFrame]
    gold: pd.DataFrame | None
    target_schema: dict[str, Any]
    fusion_frames: list[pd.DataFrame]
    em_gold: dict[tuple[str, str], pd.DataFrame]
    source_pairs: list[tuple[str, str]]
    record_id_columns: dict[str, str | None] = field(default_factory=dict)
    input_tag: str = SM_INPUT_TAG
    source_files: dict[str, Path] = field(default_factory=dict)

    @property
    def strict_em_pairs(self) -> bool:
        """A declared pair without EM gold is an error on the shipped view
        (every task ships it) and silently skipped on the legacy path."""
        return self.input_tag == SM_INPUT_TAG


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _canonical_domain(domain: str) -> str:
    return _resolve_knob_config_alias(domain) or domain


def _gold_source_files(sm_dir: Path) -> dict[str, str]:
    """``source_files`` of the base SM gold JSON (empty when absent)."""
    path = sm_dir / "sm_mapping_gold.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    files = payload.get("source_files") or {}
    return {str(k): str(v) for k, v in files.items()}


def _base_source_path(
    task_root: Path, spec: SourceSpec, gold_files: dict[str, str]
) -> Path:
    """Resolve a base-task source file: the gold's ``source_files`` entry
    (relative to ``input/schemamatching/``), else the config ``file``, else
    ``<source>.csv`` (fixture trees), under ``input/data/``."""
    sm_dir = task_root / "input" / "schemamatching"
    data_dir = task_root / "input" / "data"
    candidates: list[Path] = []
    rel = gold_files.get(spec.name)
    if rel:
        candidates.append(Path(os.path.normpath(sm_dir / rel)))
    candidates.append(data_dir / spec.file)
    candidates.append(data_dir / f"{spec.name}.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"SM view: no source file for {spec.name!r} under {task_root} "
        f"(tried {[str(p) for p in candidates]})"
    )


def _read_source(path: Path, name: str, reader_kwargs: dict[str, Any]) -> pd.DataFrame:
    """Read one source file with the PyDI reader for its extension."""
    kwargs = {k: v for k, v in reader_kwargs.items() if k not in _ID_MINTING_READER_KWARGS}
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        kwargs.setdefault("lines", True)
        return load_json(path, name=name, **kwargs)
    if suffix == ".json":
        return load_json(path, name=name, **kwargs)
    if suffix == ".xml":
        kwargs.setdefault("nested_handling", "aggregate")
        # Namespace stripping only removes the ``{uri}`` qualifier PyDI's XML
        # reader keeps; the local element names stay as shipped.
        return _strip_xml_namespaces(load_xml(path, name=name, **kwargs))
    return load_csv(path, name=name, **kwargs)


def _is_container(value: Any) -> bool:
    return isinstance(value, (list, tuple, dict))


def prepare_sm_source(df: pd.DataFrame, *, domain: str, source_name: str) -> pd.DataFrame:
    """The only post-read steps of the SM view: drop the domain's
    non-matchable columns (products ``cluster_id``), stringify list / dict
    cells, set ``attrs["dataset_name"]``. Column names are not changed."""
    attrs = dict(df.attrs)
    drop = [
        c
        for c in _SM_DROPPED_SOURCE_COLUMNS.get(_canonical_domain(domain), ())
        if c in df.columns
    ]
    out = df.drop(columns=drop) if drop else df
    list_cols = [
        c
        for c in out.columns
        if out[c].dtype == object and out[c].map(_is_container).any()
    ]
    if list_cols:
        out = out.copy()
        for c in list_cols:
            out[c] = out[c].map(lambda v: str(v) if _is_container(v) else v)
    out.attrs = attrs
    out.attrs["dataset_name"] = source_name
    return out


def _load_sources(
    config: DomainConfig, task_root: Path, *, is_base: bool
) -> tuple[dict[str, pd.DataFrame], dict[str, Path], dict[str, str | None]]:
    sm_dir = task_root / "input" / "schemamatching"
    data_dir = task_root / "input" / "data"
    gold_files = _gold_source_files(sm_dir) if is_base else {}
    sources: dict[str, pd.DataFrame] = {}
    files: dict[str, Path] = {}
    id_cols: dict[str, str | None] = {}
    for spec in config.sources:
        if is_base:
            path = _base_source_path(task_root, spec, gold_files)
            reader_kwargs = dict(spec.reader_kwargs)
        else:
            path = data_dir / f"{spec.name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"SM view: variant source missing: {path}")
            reader_kwargs = {}
        df = _read_source(path, spec.name, reader_kwargs)
        df = prepare_sm_source(df, domain=config.domain, source_name=spec.name)
        sources[spec.name] = df
        files[spec.name] = path
        candidates = [spec.id_column] if is_base and spec.id_column else []
        candidates.append("id")
        id_cols[spec.name] = next((c for c in candidates if c in df.columns), None)
    return sources, files, id_cols


# ---------------------------------------------------------------------------
# Fusion frames (target instance values)
# ---------------------------------------------------------------------------


def _read_fusion_any(path: Path, name: str, domain: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = load_csv(path, name=name)
        df.attrs["dataset_name"] = name
        return df
    return _load_fusion_file(path, name, domain)


def _load_fusion_frames(
    config: DomainConfig, task_root: Path, domain: str, *, is_base: bool
) -> list[pd.DataFrame]:
    """Fusion validation + test (validation first, as the bundle runner
    ordered them), read as shipped: no anchor ``id`` column is inserted."""
    fusion_dir = task_root / "input" / "fusion"
    if not is_base:
        test, val = _load_fusion(fusion_dir, domain=domain)
        return [f for f in (val, test) if f is not None]
    name_pairs = [(config.fusion_files["test"], config.fusion_files["validation"])]
    name_pairs += [p for p in _BASE_FUSION_FALLBACKS if p not in name_pairs]
    for test_name, val_name in name_pairs:
        test_path = fusion_dir / test_name
        if not test_path.exists():
            continue
        frames: list[pd.DataFrame] = []
        val_path = fusion_dir / val_name
        if val_path.exists():
            frames.append(_read_fusion_any(val_path, "fusion_validation_set", domain))
        frames.append(_read_fusion_any(test_path, "fusion_test_set", domain))
        return frames
    raise FileNotFoundError(
        f"SM view: no fusion gold under {fusion_dir} (tried {name_pairs})"
    )


# ---------------------------------------------------------------------------
# EM gold (duplicate-based member)
# ---------------------------------------------------------------------------


def _products_base_em_stem(pair: tuple[str, str]) -> str | None:
    """``("products_1", "products_3")`` -> ``"prod1_to_prod3"``."""
    try:
        n1 = int(pair[0].rsplit("_", 1)[1])
        n2 = int(pair[1].rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return f"prod{n1}_to_prod{n2}"


def _read_products_base_em(em_dir: Path, pair: tuple[str, str]) -> pd.DataFrame | None:
    """The shipped products base EM gold (``prod1_to_prod<m>_{all,test}.csv``:
    ``id1,id2,label[,split]``, bare ids as in ``dataset_<n>.json``, 0/1
    labels), returned as ``id1, id2, label`` without id prefixes."""
    stem = _products_base_em_stem(pair)
    if stem is None:
        return None
    for split in ("all", "test"):
        path = em_dir / f"{stem}_{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            return df[["id1", "id2", "label"]].reset_index(drop=True)
    return None


def _load_view_em_gold(
    config: DomainConfig, task_root: Path, domain: str
) -> dict[tuple[str, str], pd.DataFrame]:
    em_dir = task_root / "input" / "entitymatching"
    em = _load_em_gold(em_dir, config.source_pairs)
    if _canonical_domain(domain) == "products":
        for pair in config.source_pairs:
            if pair not in em:
                frame = _read_products_base_em(em_dir, pair)
                if frame is not None:
                    em[pair] = frame
    return em


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_sm_view(domain: str, level: str, task_root: Path | str) -> SMView:
    """Read the SM view of ``(domain, level)`` from ``task_root``.

    ``level == "baseline"`` applies the base-task rules (gold JSON, its
    ``source_files``, the config's fusion files); any other level the
    packaged-variant rules (``<source>.csv``, ``sm_mapping.csv``,
    ``test_set.xml`` / ``validation_set.xml``).
    """
    task_root = Path(task_root)
    config = load_domain_config(domain)
    is_base = level == "baseline"
    sm_dir = task_root / "input" / "schemamatching"

    sources, files, id_cols = _load_sources(config, task_root, is_base=is_base)
    gold = _load_sm_mapping(sm_dir, baseline=is_base)
    target_schema = _load_target_schema(sm_dir)
    fusion_frames = _load_fusion_frames(config, task_root, domain, is_base=is_base)
    em_gold = _load_view_em_gold(config, task_root, domain)

    logger.info(
        "SM view %s/%s from %s: sources=%s gold_rows=%s em_pairs=%d",
        domain,
        level,
        task_root,
        {k: len(v.columns) for k, v in sources.items()},
        None if gold is None else len(gold),
        len(em_gold),
    )
    return SMView(
        domain=domain,
        level=level,
        task_root=task_root,
        sources=sources,
        gold=gold,
        target_schema=target_schema,
        fusion_frames=fusion_frames,
        em_gold=em_gold,
        source_pairs=list(config.source_pairs),
        record_id_columns=id_cols,
        input_tag=SM_INPUT_TAG,
        source_files=files,
    )


def load_sm_view_for_bundle(bundle: Any) -> SMView | None:
    """The shipped SM view for ``bundle``, or ``None`` when the bundle has
    no ``sm_task_root`` (hand-built bundles)."""
    root = getattr(bundle, "sm_task_root", None)
    if root is None:
        return None
    return load_sm_view(bundle.domain, bundle.level, root)


def legacy_sm_view(bundle: Any) -> SMView:
    """The legacy SM inputs, taken from the bundle itself (renamed
    frames, reconciled gold). Only for hand-built bundles without
    ``sm_task_root``."""
    fusion_frames = [
        f for f in (bundle.fusion_validation, bundle.fusion_gold) if f is not None
    ]
    return SMView(
        domain=bundle.domain,
        level=bundle.level,
        task_root=None,
        sources=bundle.sources,
        gold=bundle.sm_mapping,
        target_schema=bundle.target_schema,
        fusion_frames=fusion_frames,
        em_gold=dict(bundle.em_gold),
        source_pairs=list(bundle.source_pairs),
        record_id_columns={},
        input_tag=SM_INPUT_TAG_LEGACY,
    )


def sm_inputs_for_bundle(bundle: Any) -> SMView:
    """What the SM stage reads for ``bundle``: the shipped view when the
    bundle names its task (``sm_task_root``), else the bundle's own frames
    with a warning."""
    view = load_sm_view_for_bundle(bundle)
    if view is not None:
        return view
    logger.warning(
        "SM stage for %s/%s: bundle has no sm_task_root; scoring the bundle's "
        "own (possibly renamed) frames against its sm_mapping (legacy path).",
        getattr(bundle, "domain", "?"),
        getattr(bundle, "level", "?"),
    )
    return legacy_sm_view(bundle)


__all__ = [
    "SM_INPUT_TAG",
    "SM_INPUT_TAG_LEGACY",
    "SMView",
    "legacy_sm_view",
    "load_sm_view",
    "load_sm_view_for_bundle",
    "prepare_sm_source",
    "sm_inputs_for_bundle",
]
