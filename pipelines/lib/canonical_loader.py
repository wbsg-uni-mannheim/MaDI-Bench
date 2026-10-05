"""Canonical bundle loader for the best-of-breed pipeline.

Reads a domain's public MaDI-Bench task folder
(``use cases/<domain>/base/``, resolved by
:func:`usecases_synthetic.lib.domain_config.task_dir`) and returns a
:class:`VariantBundle` shaped like the synthetic-side
``variant_loader.load_variant(domain, "baseline")`` output.

Why this exists
---------------
``usecases_synthetic/config/domains/products.yaml`` ships a
``data_root: usecases_synthetic/usecases`` override so the difficulty
generator works on its own copy of the products base task.
``load_variant(products, "baseline")`` therefore lands on
``usecases_synthetic/usecases/products/``; the best-of-breed pipeline (P2)
must run on the released base task instead, so products gets a dedicated
loader here. The papers base task is loaded here too
(:func:`load_canonical_papers_bundle`). music / games / companies go
through ``load_variant``, which already lands on the public tree. The
products VARIANTS go through ``load_variant`` too
(``use cases/products/<level>/``); :mod:`pipelines.lib.bundle` finishes both
products paths with :func:`finalize_products_bundle`.

Products base task, public v2 layout
------------------------------------------------------
1. **Sources** -- ``input/data/dataset_<n>.json`` with source-native column
   names (``manufacturer`` / ``brandName`` / ``Brand`` / ``mfr`` ...) and
   bare-integer ids. The loader prefixes every id with ``products_<n>_`` (the
   id scheme of the fusion gold, the variants and the public scorer) and keeps
   the raw column names: the EM / fusion committees translate them through
   their YAML ``column_mapping``. The SM committee reads the task files
   itself, as shipped (bare ids; ``sm_task_root`` ->
   :mod:`usecases_synthetic.lib.sm_view`).

   The generator's copy of the sources carries a ``cluster_id`` column: the
   WDC gold grouping (not in the released task files; it must never
   feed a matcher or reach the fused output). The loader drops it from every
   source frame before any stage sees it (:func:`drop_products_gold_grouping`;
   the SM gold likewise treats it as "dropped before schema matching").

2. **EM gold** -- ``input/entitymatching/prod1_to_prod<m>_{train,val,test,all}.csv``
   (``id1,id2,label``; ``_all`` adds a ``split`` column), ``id1`` from
   ``dataset_1`` and ``id2`` from ``dataset_<m>``, bare integers, label 0/1.
   Both ids are prefixed like the sources and the label becomes the
   ``true``/``false`` strings of the synthetic side. The pooled
   ``{train,val,test}_gt.csv`` files (all source combinations, unprefixed) are
   not read: the pipeline's pair set is products_1 x {2, 3, 4}.

3. **Fusion gold** -- ``input/fusion/fusion_{test,validation}_set.csv``: one row
   per fused product, a ``source_ids`` column (comma-joined member record ids,
   ``products_<n>_<id>``) and the 19 graded attributes; v2 blanked cells are
   empty. Each record is keyed on its ANCHOR, the anchor rule of the public
   scorer (:func:`madi_bench.evaluation.fusion_gold.pick_anchor`: the first
   ``products_1_`` member, else the smallest id of the best fallback source).
   :func:`prepare_products_fusion_gold` adds the anchor as ``id`` and puts it
   first in ``source_ids`` (the fusion committee scores with
   ``gold_id_column: source_ids``, and PyDI's evaluator tries the listed ids in
   order).

4. **SM gold** -- ``input/schemamatching/sm_mapping_gold.json`` (100 positive
   correspondences over the raw source columns; ``cluster_id`` is no target
   attribute).

The workflow silver the e2e panel scores against is built from the same
``source_ids`` records by :func:`load_products_workflow_silver` (base CSV and
variant XML alike).
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from madi_bench.evaluation.fusion_gold import parse_id_list, pick_anchor
from usecases_synthetic.lib.domain_config import task_dir
from usecases_synthetic.lib.variant_loader import (
    VariantBundle,
    _load_sm_mapping as _load_sm_gold,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


_PRODUCTS_SOURCE_FILES = {
    "products_1": "dataset_1.json",
    "products_2": "dataset_2.json",
    "products_3": "dataset_3.json",
    "products_4": "dataset_4.json",
}

#: Products source column holding the WDC gold grouping (generator copy only,
#: not in the released task files): never a matcher feature or fused value.
PRODUCTS_GOLD_GROUPING_COLUMN = "cluster_id"

#: Fusion-gold column listing a record's member ids (the committee YAML's
#: ``gold_id_column``) and the column the loader writes the anchor into.
PRODUCTS_FUSION_MEMBERS_COLUMN = "source_ids"
PRODUCTS_FUSION_ANCHOR_COLUMN = "id"

_PRODUCTS_EM_PAIRS: tuple[tuple[str, str, str], ...] = (
    # (src1, src2, public EM file stem)
    ("products_1", "products_2", "prod1_to_prod2"),
    ("products_1", "products_3", "prod1_to_prod3"),
    ("products_1", "products_4", "prod1_to_prod4"),
)

# EM gold directory: the public name first; ``entity_matching_gt`` is the
# pre-release name of the same files (kept as a fallback for old checkouts).
_PRODUCTS_EM_DIRS: tuple[str, ...] = ("entitymatching", "entity_matching_gt")

_PRODUCTS_FUSION_FILES = {
    "test": "fusion_test_set.csv",
    "validation": "fusion_validation_set.csv",
}


def _prefix_ids(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = prefix + df[column].astype(str)
    return df


def drop_products_gold_grouping(
    sources: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return ``sources`` with the gold-grouping column removed.

    The products sources ship ``cluster_id`` (the WDC gold grouping). It must
    never feed a matcher, a blocker, the fusion engine (which would copy it
    into the fused output as a "first non-null" attribute) or the SM
    committee (the SM gold excludes it). Frame ``attrs`` are preserved.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        if PRODUCTS_GOLD_GROUPING_COLUMN in df.columns:
            slim = df.drop(columns=[PRODUCTS_GOLD_GROUPING_COLUMN])
            slim.attrs = dict(df.attrs)
            out[name] = slim
        else:
            out[name] = df
    return out


def products_anchor(source_ids: Any) -> str | None:
    """The anchor of a fusion-gold record's member list (public scorer rule)."""
    return pick_anchor(parse_id_list(source_ids), "products")


def prepare_products_fusion_gold(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Key a products fusion-gold frame on its anchor.

    Accepts the base CSV frame and the variant XML frame (``load_variant``,
    which also carries ``<attr>_provenance`` columns). Adds the anchor as
    ``id`` (first column) and rewrites ``source_ids`` as a comma-joined list
    with the anchor first, so PyDI's ``DataFusionEvaluator`` (which aligns a
    ``source_ids``-keyed gold record on the first listed member it finds in the
    fused output) tries the anchor first. Records without any
    ``products_<n>_`` member are dropped (none in the released gold). A
    leftover ``cluster_id`` column is dropped.
    """
    if frame is None:
        return None
    if PRODUCTS_FUSION_MEMBERS_COLUMN not in frame.columns:
        raise ValueError(
            "products fusion gold has no 'source_ids' column (columns: "
            f"{list(frame.columns)[:8]}...). The pre-v2 pair-shaped layout "
            "(id_left / id_right / filled / cluster_id) is no longer supported."
        )
    df = frame.copy()
    if "filled" in df.columns:  # mirrors madi_bench load_gold on legacy files
        df = df[df["filled"] == "y"].copy()
    members = df[PRODUCTS_FUSION_MEMBERS_COLUMN].map(parse_id_list)
    anchors = members.map(lambda ids: pick_anchor(ids, "products"))
    n_unanchored = int(anchors.isna().sum())
    if n_unanchored:
        logger.warning(
            "products fusion gold: %d record(s) without a products_<n>_ member "
            "dropped (no anchor).",
            n_unanchored,
        )
    keep = anchors.notna()
    df, members, anchors = df[keep].copy(), members[keep], anchors[keep]
    df[PRODUCTS_FUSION_MEMBERS_COLUMN] = [
        ",".join([anchor] + [m for m in ids if m != anchor])
        for ids, anchor in zip(members, anchors)
    ]
    df = df.drop(
        columns=[
            c
            for c in (PRODUCTS_FUSION_ANCHOR_COLUMN, PRODUCTS_GOLD_GROUPING_COLUMN)
            if c in df.columns
        ]
    )
    df.insert(0, PRODUCTS_FUSION_ANCHOR_COLUMN, list(anchors))
    df = df.reset_index(drop=True)
    df.attrs = dict(frame.attrs)
    return df


def _load_canonical_sources(
    products_root: Path,
) -> dict[str, pd.DataFrame]:
    """Load the 4 product sources with their RAW per-source column names
    (manufacturer / brandName / Brand / mfr, ...) and ``products_<n>_``
    prefixed ids; the gold-grouping ``cluster_id`` column is dropped."""
    sources: dict[str, pd.DataFrame] = {}
    for source_name, fname in _PRODUCTS_SOURCE_FILES.items():
        path = products_root / "input" / "data" / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Canonical products source missing: {path}. "
                "Check use cases/products/base/input/data/."
            )
        with path.open(encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        if "id" not in df.columns:
            raise KeyError(
                f"Canonical source {source_name} has no 'id' column; cannot "
                "prefix. Check the dataset file."
            )
        df = _prefix_ids(df, "id", f"{source_name}_")
        df.attrs["dataset_name"] = source_name
        # Raw column names: the SM committee scores them; the EM / fusion
        # committee YAMLs translate them via their ``column_mapping`` blocks.
        df.attrs["needs_sm_column_translation"] = True
        sources[source_name] = df
    return drop_products_gold_grouping(sources)


def _read_canonical_em_csv(path: Path, src1: str, src2: str) -> pd.DataFrame:
    """Read a public EM CSV into the synthetic shape: ``id1`` / ``id2``
    prefixed with their source, label as ``true`` / ``false``. Extra columns
    (``split`` in the ``_all`` files) are dropped."""
    df = pd.read_csv(path)
    if not {"id1", "id2", "label"}.issubset(df.columns):
        raise ValueError(
            f"Canonical EM CSV {path} missing required cols id1/id2/label; "
            f"got {list(df.columns)}"
        )
    df = _prefix_ids(df, "id1", f"{src1}_")
    df = _prefix_ids(df, "id2", f"{src2}_")
    df["label"] = df["label"].apply(lambda v: "true" if int(v) == 1 else "false")
    return df[["id1", "id2", "label"]].reset_index(drop=True)


def _products_em_dir(products_root: Path) -> Path:
    """The EM gold directory of the products base task (public name first)."""
    for name in _PRODUCTS_EM_DIRS:
        em_dir = products_root / "input" / name
        if any(
            (em_dir / f"{stem}_all.csv").exists() for _, _, stem in _PRODUCTS_EM_PAIRS
        ):
            return em_dir
    raise FileNotFoundError(
        "No products EM gold (prod1_to_prod<m>_all.csv) under "
        + " or ".join(str(products_root / "input" / n) for n in _PRODUCTS_EM_DIRS)
    )


def _load_canonical_em(
    products_root: Path,
) -> tuple[
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], dict[str, pd.DataFrame]],
]:
    """Build ``em_gold`` (the ``_all`` file per pair) + ``em_splits``
    (train / val / test / all) from ``input/entitymatching/``."""
    em_dir = _products_em_dir(products_root)
    em_gold: dict[tuple[str, str], pd.DataFrame] = {}
    em_splits: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for src1, src2, stem in _PRODUCTS_EM_PAIRS:
        all_path = em_dir / f"{stem}_all.csv"
        if not all_path.exists():
            logger.warning(
                "Canonical EM gold missing for %s-%s at %s; skipping pair.",
                src1,
                src2,
                all_path,
            )
            continue
        em_gold[(src1, src2)] = _read_canonical_em_csv(all_path, src1, src2)

        pair_splits: dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test", "all"):
            sp_path = em_dir / f"{stem}_{split}.csv"
            if sp_path.exists():
                pair_splits[split] = _read_canonical_em_csv(sp_path, src1, src2)
        if pair_splits:
            em_splits[(src1, src2)] = pair_splits
    return em_gold, em_splits


def _check_em_ids_against_sources(
    em_splits: Mapping[tuple[str, str], Mapping[str, pd.DataFrame]],
    sources: Mapping[str, pd.DataFrame],
) -> None:
    """Log EM gold ids that are no record of their source (none expected)."""
    for (src1, src2), splits in em_splits.items():
        ids1 = set(sources[src1]["id"].astype(str)) if src1 in sources else set()
        ids2 = set(sources[src2]["id"].astype(str)) if src2 in sources else set()
        for split, frame in splits.items():
            bad1 = int((~frame["id1"].astype(str).isin(ids1)).sum())
            bad2 = int((~frame["id2"].astype(str).isin(ids2)).sum())
            if bad1 or bad2:
                logger.warning(
                    "products EM gold %s-%s %s: %d id1 / %d id2 not in the sources",
                    src1,
                    src2,
                    split,
                    bad1,
                    bad2,
                )


def _load_canonical_fusion(
    products_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read ``fusion_{test,validation}_set.csv`` (v2: ``source_ids`` + the 19
    graded attributes) and key every record on its anchor."""
    fusion_dir = products_root / "input" / "fusion"

    def _load_one(split: str) -> pd.DataFrame | None:
        path = fusion_dir / _PRODUCTS_FUSION_FILES[split]
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df.attrs["dataset_name"] = f"fusion_{split}_set"
        return prepare_products_fusion_gold(df)

    test = _load_one("test")
    if test is None:
        raise FileNotFoundError(
            "Canonical products fusion test gold missing at "
            f"{fusion_dir / _PRODUCTS_FUSION_FILES['test']}"
        )
    return test, _load_one("validation")


def _load_canonical_sm_gold(products_root: Path) -> pd.DataFrame | None:
    """Read the SM gold (``sm_mapping_gold.json``, legacy CSV as fallback).

    The canonical path keeps *raw* source column names, which the JSON gold is
    authored against, so it is consumed as-is (no source-column
    reconciliation, unlike the synthetic ``load_variant`` path).
    """
    sm_dir = products_root / "input" / "schemamatching"
    mapping = _load_sm_gold(sm_dir, baseline=True)
    if mapping is None:
        logger.warning(
            "Canonical SM gold not found at %s; SM evaluation will be "
            "unavailable for this run.",
            sm_dir / "sm_mapping_gold.json",
        )
    return mapping


def _load_canonical_target_schema(products_root: Path) -> dict:
    """Read the products target schema. Prefers ``target_schema.json``,
    falls back to ``products_target_schema.json``."""
    sm_dir = products_root / "input" / "schemamatching"
    for fname in ("target_schema.json", "products_target_schema.json"):
        path = sm_dir / fname
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No target_schema.json or products_target_schema.json under {sm_dir}"
    )


def load_canonical_products_bundle(products_root: Path | None = None) -> VariantBundle:
    """Return a baseline :class:`VariantBundle` for the products base task
    (``use cases/products/base/`` unless ``products_root`` is given)."""
    if products_root is None:
        products_root = task_dir("products")
    products_root = Path(products_root)
    if not products_root.exists():
        raise FileNotFoundError(f"Canonical products root not found at {products_root}")

    sources = _load_canonical_sources(products_root)
    em_gold, em_splits = _load_canonical_em(products_root)
    _check_em_ids_against_sources(em_splits, sources)
    fusion_gold, fusion_validation = _load_canonical_fusion(products_root)
    sm_mapping = _load_canonical_sm_gold(products_root)
    target_schema = _load_canonical_target_schema(products_root)

    logger.info(
        "Loaded canonical products bundle from %s "
        "(sources=%d, em_pairs=%d, fusion_test_rows=%d, fusion_val_rows=%s, sm_gold=%s)",
        products_root,
        len(sources),
        len(em_gold),
        len(fusion_gold),
        "-" if fusion_validation is None else len(fusion_validation),
        "yes" if sm_mapping is not None else "no",
    )

    return VariantBundle(
        domain="products",
        level="baseline",
        sources=sources,
        target_schema=target_schema,
        sm_mapping=sm_mapping,
        em_gold=em_gold,
        em_splits=em_splits,
        fusion_gold=fusion_gold,
        fusion_validation=fusion_validation,
        pooled_positives=None,
        variant_root=products_root,
        # Schema matching reads this task as shipped (bare ids, no anchor
        # ``id`` in the fusion frames), see usecases_synthetic.lib.sm_view.
        sm_task_root=products_root,
    )


def finalize_products_bundle(bundle: VariantBundle) -> VariantBundle:
    """Apply the products invariants to any products bundle (base or variant):
    drop the gold-grouping ``cluster_id`` from the sources and key both fusion-gold
    frames on the anchor. Idempotent."""
    bundle.sources = drop_products_gold_grouping(bundle.sources)
    bundle.fusion_gold = prepare_products_fusion_gold(bundle.fusion_gold)
    bundle.fusion_validation = prepare_products_fusion_gold(bundle.fusion_validation)
    return bundle


# ---------------------------------------------------------------------------
# Products workflow silver (the e2e panel's gold reference)
# ---------------------------------------------------------------------------

#: Fusion-gold columns that are bookkeeping, never a fused attribute.
_PRODUCTS_SILVER_BOOKKEEPING = {
    PRODUCTS_FUSION_ANCHOR_COLUMN,
    PRODUCTS_FUSION_MEMBERS_COLUMN,
    PRODUCTS_GOLD_GROUPING_COLUMN,
    "filled",
}


def _source_of(record_id: str, prefix_map: Mapping[str, str]) -> str:
    best = ""
    for prefix in prefix_map:
        if record_id.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return prefix_map[best] if best else "unknown"


def _products_fusion_gold_path(task_root: Path, split: str) -> Path:
    """Base task: ``fusion_<split>_set.csv``; variant / synthetic-side tree:
    ``<split>_set.xml`` (``<source_ids>`` element per record)."""
    if split not in ("test", "validation"):
        raise ValueError(f"split must be 'test' or 'validation', got {split!r}")
    fusion_dir = Path(task_root) / "input" / "fusion"
    csv_path = fusion_dir / _PRODUCTS_FUSION_FILES[split]
    if csv_path.exists():
        return csv_path
    xml_path = fusion_dir / f"{split}_set.xml"
    if xml_path.exists():
        return xml_path
    raise FileNotFoundError(
        f"No products fusion gold for split {split!r} under {fusion_dir}"
    )


def _parse_products_gold_xml(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(records, cell_provenance)`` of a ``<source_ids>``-keyed products XML.

    ``records``: one dict per ``<product>`` with ``source_ids`` (the element
    text) and every attribute's text (blank / v2-blanked cells -> None).
    ``cell_provenance``: ``(source_ids-text, attribute, [member ids])`` for every
    cell with a non-empty ``provenance`` attribute. An ``<id>`` element (the
    generator's synthetic-side copy has one) is ignored: the key is the
    anchor of ``source_ids``.
    """
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for elem in ET.parse(path).getroot():
        members_el = elem.find(PRODUCTS_FUSION_MEMBERS_COLUMN)
        if members_el is None or not (members_el.text or "").strip():
            logger.warning(
                "products fusion gold %s: record without <source_ids> skipped", path
            )
            continue
        key = members_el.text.strip()
        row: dict[str, Any] = {PRODUCTS_FUSION_MEMBERS_COLUMN: key}
        for child in elem:
            if child.tag in (PRODUCTS_FUSION_MEMBERS_COLUMN, "id"):
                continue
            text = (child.text or "").strip()
            row[child.tag] = text if text else None
            prov = (child.get("provenance") or "").strip()
            if prov:
                provenance.append(
                    {
                        "key": key,
                        "attribute": child.tag,
                        "source_ids": [p.strip() for p in prov.split("+") if p.strip()],
                    }
                )
        records.append(row)
    return records, provenance


def load_products_workflow_silver(
    task_root: Path | None = None,
    split: str = "test",
    *,
    prefix_map: Mapping[str, str] | None = None,
    key_column: str = "cluster_id",
) -> "Any":
    """Build the e2e panel's :class:`SilverStandard` from a products fusion gold.

    Works on the base task (``fusion_<split>_set.csv``) and on a variant or the
    generator's copy (``<split>_set.xml``). Every gold record is keyed on the
    anchor of its ``source_ids`` (``key_column`` of ``fused``; the
    ``cluster_id`` column of ``membership`` and ``cell_provenance``, as PyDI
    requires). ``membership`` lists every ``source_ids`` member (plus any
    provenance id outside the list) with its source from ``prefix_map``.
    ``cell_provenance`` comes from the XML ``provenance`` attributes (the base
    CSV has none). The products data column ``cluster_id`` is never
    read: ``key_column`` holds the anchor.
    """
    from PyDI.evaluation.silver_standard import SilverStandard

    task_root = Path(task_root) if task_root is not None else task_dir("products")
    prefix_map = dict(
        prefix_map or {f"products_{n}_": f"products_{n}" for n in (1, 2, 3, 4)}
    )
    path = _products_fusion_gold_path(task_root, split)

    prov_rows: list[dict[str, Any]] = []
    if path.suffix == ".csv":
        raw = pd.read_csv(path)
    else:
        records, prov_rows = _parse_products_gold_xml(path)
        raw = pd.DataFrame(records)
    gold = prepare_products_fusion_gold(raw)

    attrs = [c for c in gold.columns if c not in _PRODUCTS_SILVER_BOOKKEEPING]
    fused = gold[attrs].copy()
    fused.insert(0, key_column, gold[PRODUCTS_FUSION_ANCHOR_COLUMN].astype(str))

    def _member(rid: str, anchor: str) -> dict[str, str]:
        source = _source_of(rid, prefix_map)
        return {"record_id": rid, "source": source, "cluster_id": anchor}

    member_rows: list[dict[str, str]] = [
        _member(rid, str(anchor))
        for anchor, members in zip(
            gold[PRODUCTS_FUSION_ANCHOR_COLUMN], gold[PRODUCTS_FUSION_MEMBERS_COLUMN]
        )
        for rid in parse_id_list(members)
    ]
    cell_rows: list[dict[str, Any]] = []
    for row in prov_rows:
        anchor = products_anchor(row["key"])
        if anchor is None:
            continue
        cell_rows.append(
            {
                "cluster_id": anchor,
                "attribute": row["attribute"],
                "source_ids": row["source_ids"],
            }
        )
        member_rows.extend(_member(rid, anchor) for rid in row["source_ids"])
    membership = (
        pd.DataFrame(member_rows, columns=["record_id", "source", "cluster_id"])
        .drop_duplicates(subset=["record_id", "source", "cluster_id"])
        .reset_index(drop=True)
    )
    cell_provenance = (
        pd.DataFrame(cell_rows, columns=["cluster_id", "attribute", "source_ids"])
        if cell_rows
        else None
    )
    logger.info(
        "Loaded products workflow silver (%s, %s): %d records, %d membership rows, "
        "%s provenance cells",
        path.name,
        split,
        len(fused),
        len(membership),
        0 if cell_provenance is None else len(cell_provenance),
    )
    return SilverStandard(
        fused=fused, membership=membership, cell_provenance=cell_provenance
    )


def load_canonical_products_workflow_silver(split: str = "test") -> "Any":
    """Back-compat wrapper: the products BASE task's workflow silver."""
    return load_products_workflow_silver(task_dir("products"), split)


# ===========================================================================
# Papers
# ===========================================================================
#
# The papers base task (``use cases/papers/base/``) has 3 sources
# (dblp, crossref, open_alex), all read from JSONL with
# PyDI.io.load_json(add_index=True); every record carries an ``id``
# of the form ``<source>-<NNNNN>`` (dash separator, zero-padded). The
# EM gold under input/entitymatching/ uses the domain-specific columns
# ``id_dblp + id_<other>`` (NOT the standard ``id1/id2/label``); the
# loader renames them. Fusion gold is JSONL (NOT XML/CSV), one record
# per fused paper keyed by ``source_ids`` (the ids of its source
# records) — the loader returns it verbatim. SM gold is
# input/schemamatching/sm_mapping_gold.json (raw source column names).

_PAPERS_SOURCE_FILES = {
    "dblp": "dblp.jsonl",
    "crossref": "crossref.jsonl",
    "open_alex": "open_alex.jsonl",
}

# EM pair declarations: (src1, src2, em-csv-stem).
# Filenames are ``<stem>_{train,val,test}.csv``.
_PAPERS_EM_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("dblp", "crossref", "dblp_crossref"),
    ("dblp", "open_alex", "dblp_openalex"),
)


def _load_canonical_papers_sources(
    papers_root: Path,
) -> dict[str, pd.DataFrame]:
    """Load papers' 3 JSONL sources via :func:`PyDI.io.load_json`. The
    records' own ``id`` values are in the dash-prefixed format the EM and
    fusion gold reference (``dblp-NNNNN``, ``crossref-NNNNN``,
    ``open_alex-NNNNN``). ``add_index=True`` also adds a ``<source>_id``
    column with the same ids; it is renamed to ``id`` only for a file
    without an ``id`` column."""
    from PyDI.io import load_json

    sources: dict[str, pd.DataFrame] = {}
    for name, fname in _PAPERS_SOURCE_FILES.items():
        path = papers_root / "input" / "data" / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Canonical papers source missing: {path}. "
                f"Check usecases/papers/input/data/."
            )
        df = load_json(path, add_index=True, lines=True, name=name)
        id_col = f"{name}_id"
        if id_col in df.columns and "id" not in df.columns:
            df = df.rename(columns={id_col: "id"})
        # Canonicalize the heterogeneous per-source papers schema (dblp
        # ``publication_title`` / crossref ``title_text`` / open_alex
        # ``display_title`` -> ``title``; ``author_list`` etc. -> ``authors``;
        # ...) so the EM/fusion committees see the canonical columns their
        # ``blocking_name_column: title`` / ``text_cols: [title, ...]`` and
        # ditto fields expect (the synthetic ``normalize_loaded_source``
        # applies the same map). Without it, the committee
        # ``column_mapping: {}`` (correct for the canonicalizing
        # loader) leaves blocking with no ``title`` column -> zero candidates
        # -> empty EM. Schema matching does not read these frames: it reads
        # the shipped files (raw column names) and the shipped SM gold
        # through ``sm_task_root`` (usecases_synthetic.lib.sm_view), so this
        # rename gives the schema matchers no free correspondences.
        from usecases_synthetic.lib.loaders import _PAPERS_SOURCE_COLUMN_MAP

        canon = {
            k: v
            for k, v in _PAPERS_SOURCE_COLUMN_MAP.get(name, {}).items()
            if k in df.columns and v not in df.columns
        }
        if canon:
            df = df.rename(columns=canon)
        # Authors ship as JSON arrays, so ``load_json`` loads them as
        # Python lists. List-valued cells break every scalar-assuming
        # SM/EM matcher (coma_hybrid: pd.notna(list) array-truthiness;
        # magneto: list in NULL_REPRESENTATIONS unhashable). After the
        # rename above the list column is ``authors`` (dblp
        # ``author_list`` / crossref ``contributor_names`` / open_alex
        # ``authors_list``); schema matching reads the raw task, not these
        # frames. Stringify any column that contains list cells; the fusion
        # runner re-parses via literal_eval (fusion_committee_papers
        # declares gold_list_columns: [authors]). Mirrors
        # usecases_synthetic.lib.loaders.normalize_loaded_source,
        # which the BoB pipeline bypasses by calling ``load_json``
        # directly.
        for col in df.columns:
            if df[col].apply(lambda v: isinstance(v, list)).any():
                df[col] = df[col].apply(lambda v: str(v) if isinstance(v, list) else v)
        df.attrs["dataset_name"] = name
        sources[name] = df
    return sources


def _read_papers_em_csv(path: Path, src1: str, src2: str) -> pd.DataFrame:
    """Read a papers EM CSV. The columns are ``id_<src1>,id_<src2>,label``;
    rename to ``id1,id2,label`` and convert 0/1 labels to ``true``/``false``
    matching the synthetic-side convention. IDs are already prefixed."""
    df = pd.read_csv(path)
    col1, col2 = f"id_{src1}", f"id_{src2}"
    # The crossref+openalex files use the non-anchor source's prefix
    # directly in the column name; accept short aliases too.
    if col1 not in df.columns or col2 not in df.columns:
        # Fall back to the two id_* columns present, taken in column order
        # as src1, src2 (the shipped files list ``id_dblp`` first).
        id_cols = [c for c in df.columns if c.startswith("id_")]
        if len(id_cols) != 2:
            raise ValueError(
                f"Papers EM CSV {path} has unexpected columns {list(df.columns)}"
            )
        col1, col2 = id_cols
    rename = {col1: "id1", col2: "id2"}
    df = df.rename(columns=rename)
    if "label" not in df.columns:
        raise ValueError(f"Papers EM CSV {path} missing label column")
    df["label"] = df["label"].apply(lambda v: "true" if int(v) == 1 else "false")
    return df[["id1", "id2", "label"]]


def _load_canonical_papers_em(
    papers_root: Path,
) -> tuple[
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], dict[str, pd.DataFrame]],
]:
    em_dir = papers_root / "input" / "entitymatching"
    em_gold: dict[tuple[str, str], pd.DataFrame] = {}
    em_splits: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for src1, src2, stem in _PAPERS_EM_PAIRS:
        # Treat the union of train+val+test as ``all`` since the
        # canonical papers tree doesn't ship an explicit ``_all.csv``.
        pair_splits: dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            sp_path = em_dir / f"{stem}_{split}.csv"
            if sp_path.exists():
                pair_splits[split] = _read_papers_em_csv(sp_path, src1, src2)
        if not pair_splits:
            logger.warning(
                "Papers EM gold missing for %s-%s under %s; skipping pair.",
                src1,
                src2,
                em_dir,
            )
            continue
        all_df = pd.concat(pair_splits.values(), ignore_index=True)
        pair_splits["all"] = all_df
        em_gold[(src1, src2)] = all_df
        em_splits[(src1, src2)] = pair_splits
    return em_gold, em_splits


def _load_canonical_papers_fusion(
    papers_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read the JSONL fusion gold (``fusion_test.jsonl``, and
    ``fusion_val.jsonl`` when present); one row per fused paper, keyed by
    ``source_ids`` (the ids of its source records)."""
    fusion_dir = papers_root / "input" / "fusion"
    test_path = fusion_dir / "fusion_test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Canonical papers fusion test silver missing at {test_path}"
        )
    def _read_jsonl(path: Path) -> pd.DataFrame:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(records)

    test = _read_jsonl(test_path)
    val_path = fusion_dir / "fusion_val.jsonl"
    validation = _read_jsonl(val_path) if val_path.exists() else None
    return test, validation


def load_canonical_papers_bundle() -> VariantBundle:
    """Return a baseline :class:`VariantBundle` for the papers base task
    (``use cases/papers/base/``).

    The papers SM gold (``sm_mapping_gold.json``) is authored against
    the RAW on-disk source columns; it is reconciled here to the
    canonical column names that :func:`_load_canonical_papers_sources`
    produces, the columns the bundle's frames carry. The SM committee
    does not read the reconciled gold: it reads the shipped task (raw
    columns, raw SM gold) through ``sm_task_root``.
    """
    papers_root = task_dir("papers")
    if not papers_root.exists():
        raise FileNotFoundError(f"Canonical papers root not found at {papers_root}")

    sources = _load_canonical_papers_sources(papers_root)
    em_gold, em_splits = _load_canonical_papers_em(papers_root)
    fusion_gold, fusion_validation = _load_canonical_papers_fusion(papers_root)
    target_schema_path = papers_root / "input" / "schemamatching" / "target_schema.json"
    if not target_schema_path.exists():
        raise FileNotFoundError(target_schema_path)
    with target_schema_path.open() as f:
        target_schema = json.load(f)

    sm_mapping = _load_sm_gold(papers_root / "input" / "schemamatching", baseline=True)

    # The committed papers SM gold (sm_mapping_gold.json) is authored against the
    # RAW on-disk source columns (display_title/work_kind/...), but
    # _load_canonical_papers_sources renames the raw source columns to their
    # canonical names via _PAPERS_SOURCE_COLUMN_MAP before any committee sees the
    # frames. Reconcile the gold's source_column to the same canonical names so
    # the gold names columns the frames carry. Mirrors
    # variant_loader._reconcile_sm_gold_source_columns; rows whose source_column
    # has no entry in the map (the three ``id`` rows) pass through unchanged.
    # The SM committee itself does not read these frames or
    # this reconciled gold (sm_task_root below -> usecases_synthetic.lib.sm_view:
    # raw columns, raw gold); the reconciled gold stays for its other readers
    # (normalization resolution, the e2e panel).
    if sm_mapping is not None and not sm_mapping.empty:
        from usecases_synthetic.lib.loaders import _PAPERS_SOURCE_COLUMN_MAP

        sm_mapping = sm_mapping.copy()
        sm_mapping["source_column"] = [
            _PAPERS_SOURCE_COLUMN_MAP.get(str(ds), {}).get(str(sc), sc)
            for ds, sc in zip(
                sm_mapping["source_dataset"], sm_mapping["source_column"]
            )
        ]

    logger.info(
        "Loaded canonical papers bundle from %s "
        "(sources=%d, em_pairs=%d, fusion_test_rows=%d, sm_gold=%s)",
        papers_root,
        len(sources),
        len(em_gold),
        len(fusion_gold),
        "yes" if sm_mapping is not None else "no",
    )

    return VariantBundle(
        domain="papers",
        level="baseline",
        sources=sources,
        target_schema=target_schema,
        sm_mapping=sm_mapping,
        em_gold=em_gold,
        em_splits=em_splits,
        fusion_gold=fusion_gold,
        fusion_validation=fusion_validation,
        pooled_positives=None,
        variant_root=papers_root,
        # Schema matching reads this task as shipped (raw column names, raw
        # SM gold), not the canonicalised frames above; see
        # usecases_synthetic.lib.sm_view.
        sm_task_root=papers_root,
    )


def load_canonical_papers_workflow_silver(split: str = "test") -> "Any":
    """Build a :class:`SilverStandard` for papers from a fusion gold with a
    ``doi`` column (``fusion_<split>.jsonl``) + the source records' ``doi``
    values.

    The pipeline's e2e panel
    (``BestOfBreedPipeline._compute_panel``) skips papers, whose fusion gold
    is flat JSONL rather than the provenance XML that ``load_workflow_silver``
    reads. This builds the SilverStandard shape the panel needs:

    * ``fused`` — one row per gold record with a non-empty ``doi``
      (``cluster_id`` = normalized doi) with the canonical fused attribute
      values from ``fusion_<split>.jsonl``.
    * ``membership`` — ``(record_id, source, cluster_id)`` for every source
      record whose DOI is in the gold, mirroring
      ``fusion_perfect_clusters._doi_to_record_ids``. Record ids use the same
      ``<source>-<NNNNN>`` scheme as the pipeline's fused ``_fusion_sources``, so
      the panel aligns pipe and gold clusters by shared record ids.
    """
    from PyDI.evaluation.silver_standard import SilverStandard

    from usecases_synthetic.lib.fusion_perfect_clusters import _normalize_doi

    papers_root = task_dir("papers")
    gold_path = papers_root / "input" / "fusion" / f"fusion_{split}.jsonl"
    if not gold_path.exists():
        raise FileNotFoundError(gold_path)
    fused = pd.read_json(gold_path, lines=True)
    fused["cluster_id"] = [_normalize_doi(d) for d in fused["doi"]]
    fused = fused[fused["cluster_id"].notna()].reset_index(drop=True)
    gold_dois = set(fused["cluster_id"])

    # Membership: scan the canonical sources for records whose DOI is in the
    # gold and group by normalized DOI (== cluster_id). source name is the
    # dataset (dblp/crossref/open_alex), matching the pipe's
    # _fusion_source_datasets; record_id is the <source>-<NNNNN> id.
    sources = _load_canonical_papers_sources(papers_root)
    rows: list[dict[str, Any]] = []
    for name, df in sources.items():
        if "doi" not in df.columns or "id" not in df.columns:
            continue
        for rid, doi in zip(df["id"].astype(str), df["doi"], strict=False):
            nd = _normalize_doi(doi)
            if nd is not None and nd in gold_dois:
                rows.append({"record_id": rid, "source": name, "cluster_id": nd})
    membership = (
        pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])
        .drop_duplicates(subset=["record_id", "source", "cluster_id"])
        .reset_index(drop=True)
    )

    logger.info(
        "Loaded canonical papers workflow silver (%s): %d clusters, %d "
        "membership rows",
        split,
        len(fused),
        len(membership),
    )
    return SilverStandard(fused=fused, membership=membership, cell_provenance=None)
