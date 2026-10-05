"""Fusion-gold helpers the end-to-end panel needs (``score_e2e.py``).

The fusion ACCURACY itself is scored by ``madi_bench.evaluation.score_fusion``
(the scorer behind ``results/scores_v2.json``); this module only builds the
panel's gold tables exactly as the paper's panels were computed.

In every task of this repository ``input/fusion/GOLD_VERSION`` says ``v2``:
the canonical file names hold the v2 gold (the sets with every cell blanked
that no source could yield).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from madi_bench.evaluation.fusion_gold import anchors_by_cluster, parse_id_list, pick_anchor

from .tasks import TaskSpec

ANCHOR_ID = "_anchor_id"

# The primary anchor source of each domain (the full priority lists, with the
# fallback sources used when a cluster has no primary-source record, are
# madi_bench.evaluation.ANCHOR_PRIORITY).
DOMAIN_ANCHOR_PREFIX = {
    "products": "products_1_",
    "music": "mbrainz_",
    "games": "metacritic_",
    "companies": "http://www.forbes.com/",
    "papers": "dblp-",
}

_NUMERIC_PRODUCT_COLS = (
    "vram_gb", "storage_gb", "read_speed_mb_s", "write_speed_mb_s",
    "width_mm", "length_mm", "height_mm", "weight_g",
)

# Columns compared by year; they must reach the comparators as strings a date
# parser can read (an integral float such as 2006.0 would be read as an epoch
# offset).
_YEAR_COLS = {
    "games": ("releaseYear",),
    "music": ("release-date",),
    "companies": ("founded",),
}

GOLD_VERSION_FILE = "GOLD_VERSION"


def _parse_id_list(value: Any) -> list[str]:
    return parse_id_list(value)


def _canon_scalar_str(value):
    """Canonical string for scalar cells: integral floats lose the trailing
    ``.0`` (read_json infers ``42`` as ``42.0`` while read_csv keeps
    ``"42"``)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text if text else None


def _canon_author_set(value):
    """Canonical author-set string for papers ``authors``: a sorted,
    deduplicated, casefolded '|'-join of the names."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if isinstance(value, (list, tuple, set)):
        parts = [str(v) for v in value]
    else:
        text = str(value).strip()
        if not text:
            return None
        parts = None
        if text.startswith("[") and text.endswith("]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, (list, tuple)):
                    parts = [str(v) for v in parsed]
                    break
        if parts is None:
            parts = re.split(r"[|;]", text)
    canon = sorted({p.strip().casefold() for p in parts if p and p.strip()})
    return "|".join(canon) if canon else None


def _symmetric_prep(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Value coercions applied to gold and submission alike."""
    frame = frame.copy()
    for col in _YEAR_COLS.get(domain, ()):
        if col in frame.columns:
            frame[col] = frame[col].apply(_canon_scalar_str)
    if domain == "games":
        if "genres_genre" in frame.columns and "genres" not in frame.columns:
            frame = frame.rename(columns={"genres_genre": "genres"})
    elif domain == "companies":
        def from_sci(v):
            if isinstance(v, str) and ("e" in v or "E" in v):
                try:
                    return int(float(v))
                except (TypeError, ValueError, OverflowError):
                    return v
            return v

        for col in ("assets", "revenue"):
            if col in frame.columns:
                frame[col] = frame[col].apply(from_sci)
    elif domain == "products":
        for col in _NUMERIC_PRODUCT_COLS:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    elif domain == "papers":
        for col in ("volume", "issue", "first_page", "last_page",
                    "publication_year", "referenced_works_count",
                    "cited_by_count"):
            if col in frame.columns:
                frame[col] = frame[col].apply(_canon_scalar_str)
        if "authors" in frame.columns:
            frame["authors"] = frame["authors"].apply(_canon_author_set)
    return frame


def _domain_gold_prep(gold: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Gold-side prep: the symmetric coercions plus gold-only rules (the
    products ``filled == 'y'`` row filter, companies' keypeople_name
    unwrapping)."""
    gold = _symmetric_prep(gold, domain)
    if domain == "companies" and "keypeople_name" in gold.columns:
        gold = gold.copy()
        gold["keypeople"] = gold["keypeople_name"].apply(
            lambda x: [x] if isinstance(x, str) else x
        )
    if domain == "products" and "filled" in gold.columns:
        gold = gold[gold["filled"] == "y"].copy()
    return gold


def _nested_list_columns_from_xml(xml_path: Path, id_tag: str = "id",
                                  separator: str = "|") -> pd.DataFrame:
    """Recover LIST-valued gold columns that PyDI's silver loader flattens.

    The gold XML nests repeated values one level down::

        <videogame><id>metacritic_...</id>
          <genres><genre>...</genre><genre>...</genre></genres></videogame>

    ``PyDI.evaluation.silver_standard.load_workflow_silver`` returns such a
    column present but entirely empty (games ``genres``, companies
    ``keypeople``), so the attribute would silently drop out of the panel.
    Returns one row per record with ``id_tag`` plus every nested column,
    grandchild texts joined by ``separator``.
    """
    import xml.etree.ElementTree as ET

    rows: list[dict] = []
    for record in ET.parse(xml_path).getroot():
        row: dict = {}
        for child in record:
            if child.tag == id_tag:
                row[id_tag] = (child.text or "").strip()
                continue
            grandchildren = list(child)
            if not grandchildren:
                continue          # scalar -- the silver loader handles it
            values = [
                (g.text or "").strip() for g in grandchildren
                if (g.text or "").strip()
            ]
            if values:
                row[child.tag] = separator.join(values)
        if row.get(id_tag):
            rows.append(row)
    return pd.DataFrame(rows)


def _repair_nested_gold_columns(gold: pd.DataFrame, task: TaskSpec,
                                id_column: str,
                                xml_path: Path | None = None) -> pd.DataFrame:
    """Fill gold columns the silver loader left entirely empty (see above).
    Only ever fills columns that are 100 % empty."""
    if xml_path is None:
        xml_path = task.root / "input" / "fusion" / "test_set.xml"
    if not xml_path.is_file():
        return gold
    empty = [c for c in gold.columns
             if c != id_column and gold[c].notna().sum() == 0]
    if not empty:
        return gold
    try:
        nested = _nested_list_columns_from_xml(xml_path)
    except Exception:  # noqa: BLE001 -- scoring must survive malformed gold
        return gold
    repairable = [c for c in empty if c in nested.columns]
    if not repairable:
        return gold
    gold = gold.copy()
    lookup = nested.set_index("id")
    ids = gold[id_column].astype(str)
    for column in repairable:
        gold[column] = ids.map(lookup[column]).where(lambda s: s.notna())
    return gold


def canonical_is_v2(task: TaskSpec) -> bool:
    """Do the canonical fusion files of this task carry the v2 gold?"""
    marker = task.root / "input" / "fusion" / GOLD_VERSION_FILE
    if not marker.is_file():
        return False
    version = marker.read_text(encoding="utf-8").strip()
    if version != "v2":
        raise ValueError(f"{task.task_id}: unknown GOLD_VERSION {version!r} (expected 'v2')")
    fdir = task.root / "input" / "fusion"
    stray = sorted(p.name for pat in ("*_v2.xml", "*_v2.csv", "*_v2.jsonl",
                                      "*_v2_better_readability.csv")
                   for p in fdir.glob(pat))
    if stray:
        raise ValueError(f"{task.task_id}: GOLD_VERSION says v2 but '_v2' twins remain: {stray}")
    return True


def papers_gold_path(task: TaskSpec) -> Path:
    """The papers fusion test gold file: flat JSON lines in EVERY tier (the
    variants' file is named test_set.xml but contains JSON lines)."""
    return (task.root / "input" / "fusion"
            / ("fusion_test.jsonl" if task.tier == "base" else "test_set.xml"))


def load_fusion_gold(task: TaskSpec) -> pd.DataFrame:
    """Fusion test gold with ``_anchor_id``, for the tasks whose gold carries
    a ``source_ids`` list per record: papers (every tier) and products base.
    The XML tasks are read with PyDI's ``load_workflow_silver`` in
    ``score_e2e.load_panel_gold``."""
    if not canonical_is_v2(task):
        raise ValueError(f"{task.task_id}: no GOLD_VERSION marker; this module reads the v2 layout only")
    if task.domain == "papers":
        gold = pd.read_json(papers_gold_path(task), lines=True)
        gold = _symmetric_prep(gold, "papers")
        gold[ANCHOR_ID] = gold["source_ids"].map(
            lambda v: pick_anchor(_parse_id_list(v), task.domain)
        )
        return gold.dropna(subset=[ANCHOR_ID])
    if task.domain == "products" and task.tier == "base":
        gold = pd.read_csv(task.root / "input" / "fusion" / "fusion_test_set.csv")
        gold = _domain_gold_prep(gold, "products")
        gold[ANCHOR_ID] = gold["source_ids"].map(
            lambda v: pick_anchor(_parse_id_list(v), task.domain)
        )
        return gold.dropna(subset=[ANCHOR_ID])
    raise ValueError(f"{task.task_id}: the gold is XML; use score_e2e.load_panel_gold")


__all__ = ["ANCHOR_ID", "DOMAIN_ANCHOR_PREFIX", "anchors_by_cluster", "canonical_is_v2",
           "load_fusion_gold", "papers_gold_path", "_parse_id_list", "_repair_nested_gold_columns"]
