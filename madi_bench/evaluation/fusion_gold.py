"""Load a task's fusion gold set and prepare it for scoring.

A task folder is ``use cases/<domain>/<tier>/`` (or ``usecases/<domain>/<tier>/``
in other copies of the task tree). The fusion gold lives in ``input/fusion/``:

* companies / games / music (every tier) and the products variants:
  ``test_set.xml`` -- one element per record with an ``<id>`` (products
  variants: a ``<source_ids>`` element instead) and one child per attribute;
  a list-valued attribute nests its elements one level down; a blanked cell
  is an empty element carrying ``v2_removed="<reason>"``;
* papers (every tier): JSON lines with a ``source_ids`` list --
  ``fusion_test.jsonl`` at base; in the variants the file is named
  ``test_set.xml`` but holds JSON lines, not XML;
* products base: ``fusion_test_set.csv`` with a ``source_ids`` column.

``input/fusion/GOLD_VERSION`` names the gold version the files carry (``v2``:
the sets with every cell blanked that no source could yield; the scorer
skips blanked cells, so the graded cells are the kept ones).

Every record is keyed on its ANCHOR: the member record from the
highest-priority source present (``ANCHOR_PRIORITY``; forbes URL > dbpedia >
fullcontact, metacritic_ > dbpedia_ > sales_, mbrainz_ > discogs_ > lastFM_,
dblp- > crossref- > open_alex-, products_1_ > ... > products_4_). Members of
an XML record are its ``<id>`` and the ids in the ``provenance`` attributes;
they are sorted before the pick, so among several members of one source the
lexicographically smallest id is the anchor. At the base tier every record
carries its primary-source member; in a variant whose knobs dropped anchor
records (the hard tiers) the affected records were re-keyed to a surviving
member of another source.

The loaded frame carries the anchor in ``_anchor_id``, only target-schema
attributes plus the bookkeeping columns, and every value prepared exactly as
the scorer prepares a submission (``symmetric_prep``).
"""

from __future__ import annotations

import ast
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

csv.field_size_limit(10 ** 8)          # as normalization_score: portable (a C long on every platform)

ANCHOR_ID = "_anchor_id"

ANCHOR_PRIORITY: dict[str, tuple[str, ...]] = {
    "products": ("products_1_", "products_2_", "products_3_", "products_4_"),
    "music": ("mbrainz_", "discogs_", "lastFM_"),
    "games": ("metacritic_", "dbpedia_", "sales_"),
    "companies": ("http://www.forbes.com/", "http://dbpedia.org/", "fullcontact_"),
    "papers": ("dblp-", "crossref-", "open_alex-"),
}

GOLD_VERSION_FILE = "GOLD_VERSION"

_YEAR_COLS = {"games": ("releaseYear",), "music": ("release-date",), "companies": ("founded",)}
_NUMERIC_PRODUCT_COLS = ("vram_gb", "storage_gb", "read_speed_mb_s", "write_speed_mb_s",
                         "width_mm", "length_mm", "height_mm", "weight_g")
_PAPERS_SCALARS = ("volume", "issue", "first_page", "last_page", "publication_year",
                   "referenced_works_count", "cited_by_count")


# ----------------------------------------------------------------- task

def domain_of(task_dir: Path) -> str:
    """The domain from the task folder (``.../<domain>/<tier>``)."""
    task_dir = Path(task_dir).resolve()
    for candidate in (task_dir.parent.name, task_dir.name):
        if candidate in ANCHOR_PRIORITY:
            return candidate
    raise ValueError(f"cannot tell the domain from {task_dir}")


def tier_of(task_dir: Path) -> str:
    return Path(task_dir).resolve().name


def gold_path(task_dir: Path, split: str = "test") -> Path:
    fdir = Path(task_dir) / "input" / "fusion"
    domain, tier = domain_of(task_dir), tier_of(task_dir)
    if tier == "base":
        name = {"papers": {"test": "fusion_test.jsonl", "validation": "fusion_val.jsonl"},
                "products": {"test": "fusion_test_set.csv", "validation": "fusion_validation_set.csv"}} \
            .get(domain, {"test": "test_set.xml", "validation": "validation_set.xml"})[split]
    else:
        name = {"test": "test_set.xml", "validation": "validation_set.xml"}[split]
    return fdir / name


def gold_version(task_dir: Path) -> str:
    """The gold version the canonical fusion files carry: the content of
    ``input/fusion/GOLD_VERSION`` ("v2" for every published task: the sets
    with the cells no source could yield blanked)
    or "v1" when the marker is absent (the sets as first published, never
    blanked). An unknown marker, or a marker next to leftover ``*_v2.*`` gold
    files, raises so that a mixed layout can
    never score silently."""
    fdir = Path(task_dir) / "input" / "fusion"
    marker = fdir / GOLD_VERSION_FILE
    if not marker.is_file():
        return "v1"
    version = marker.read_text(encoding="utf-8").strip()
    if version != "v2":
        raise ValueError(f"{task_dir}: unknown GOLD_VERSION {version!r} (expected 'v2')")
    stray = sorted(p.name for pattern in ("*_v2.xml", "*_v2.csv", "*_v2.jsonl", "*_v2_better_readability.csv")
                   for p in fdir.glob(pattern))
    if stray:
        raise ValueError(f"{task_dir}: GOLD_VERSION says v2 but '_v2' twins remain: {stray}")
    return version


def target_schema_attributes(task_dir: Path) -> list[str]:
    schema = json.loads((Path(task_dir) / "input" / "schemamatching" / "target_schema.json")
                        .read_text(encoding="utf-8"))
    return [a for a in schema.get("properties", {}) if a != "id"]


# ------------------------------------------------------------- anchors

def parse_id_list(value: Any) -> list[str]:
    """A list, a JSON / Python list literal, or a comma-joined string."""
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    if not isinstance(value, str):
        return []
    s = value.strip()
    if not s or s in {"[]", "nan", "None"}:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s.replace("'", '"'))
            return [str(x) for x in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(s)
                return [str(x) for x in parsed] if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                return []
    return [part.strip() for part in s.split(",") if part.strip()]


def pick_anchor(ids: list[str], domain: str, exclude=None) -> str | None:
    """The anchor of a member list: the first id (list order) from the
    primary source, else the lexicographically smallest id from the next
    source in priority order that is present. Ids in ``exclude`` never
    anchor (a submission to a variant: its synthetic records,
    ``variant_only_record_ids``)."""
    if exclude:
        ids = [record_id for record_id in ids if record_id not in exclude]
    for rank, prefix in enumerate(ANCHOR_PRIORITY[domain]):
        hits = [record_id for record_id in ids if record_id.startswith(prefix)]
        if hits:
            return hits[0] if rank == 0 else min(hits)
    return None


def anchors_by_cluster(membership: pd.DataFrame, domain: str, exclude=None) -> pd.Series:
    """cluster_id -> anchor for a membership table (record_id, cluster_id):
    the first member in row order from the primary source, else the smallest
    id from the best fallback source present; members in ``exclude`` never
    anchor (see ``pick_anchor``)."""
    membership = membership.dropna(subset=["cluster_id"])
    if exclude:
        membership = membership[~membership["record_id"].astype(str).isin(exclude)]
    record_ids = membership["record_id"].astype(str)
    chosen: dict = {}
    for rank, prefix in enumerate(ANCHOR_PRIORITY[domain]):
        hit = membership[record_ids.str.startswith(prefix)]
        if rank:
            hit = hit.assign(_rid=hit["record_id"].astype(str)).sort_values("_rid", kind="stable")
        hit = hit.drop_duplicates("cluster_id")
        for cluster_id, record_id in zip(hit["cluster_id"], hit["record_id"].astype(str)):
            chosen.setdefault(cluster_id, record_id)
    return pd.Series(chosen, dtype=object)


# ------------------------------------------------ variant-only records
#
# A difficulty variant adds records of its own (the generator's copies and
# fabrications). Their ids look like any other id of their source, so the
# anchor rule could pick one as a submitted cluster's anchor; a submission is
# therefore anchored on real records only. The variant-only records are the
# record ids of the variant's source tables that are no record id of the
# base task's sources, read from the sibling base folder
# (``use cases/<domain>/base``). Record ids: a CSV's ``id`` column (else its
# first column), a JSON / JSON-lines record's ``id`` key; products base ids
# are prefixed ``products_<n>_`` from ``dataset_<n>.json``, as in the variants.

_SOURCE_SUFFIXES = (".csv", ".json", ".jsonl")
_PRODUCTS_BASE_FILE = re.compile(r"dataset_(\d+)")
_JSONL_ID = re.compile(r'\s*\{\s*"id"\s*:\s*"([^"\\]*)"')


def _source_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir()
                  if p.is_file() and p.suffix in _SOURCE_SUFFIXES and not p.name.endswith("_metadata.json"))


def _file_record_ids(path: Path, domain: str) -> list[str]:
    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            col = header.index("id") if "id" in header else 0
            return [row[col] for row in reader if len(row) > col]
    if path.suffix == ".jsonl":
        ids = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = _JSONL_ID.match(line)
                if m:
                    ids.append(m.group(1))
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict) and rec.get("id") is not None:
                    ids.append(str(rec["id"]))
        return ids
    records = json.loads(path.read_text(encoding="utf-8"))
    ids = ([str(r["id"]) for r in records if isinstance(r, dict) and r.get("id") is not None]
           if isinstance(records, list) else [])
    m = _PRODUCTS_BASE_FILE.fullmatch(path.stem)
    if domain == "products" and m:
        ids = [f"products_{m.group(1)}_{i}" for i in ids]
    return ids


@lru_cache(maxsize=64)
def _record_ids(data_dir: str, domain: str, key: tuple) -> frozenset[str]:
    ids: set[str] = set()
    for path in _source_files(Path(data_dir)):
        ids.update(_file_record_ids(path, domain))
    return frozenset(ids)


def source_record_ids(task_dir: Path, domain: str) -> frozenset[str]:
    """Every record id of a task's source tables (``input/data``), cached
    per file state."""
    data_dir = Path(task_dir) / "input" / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{task_dir}: no input/data to read the record ids from")
    key = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in _source_files(data_dir))
    return _record_ids(str(data_dir.resolve()), domain, key)


@lru_cache(maxsize=64)
def _difference(variant: frozenset[str], base: frozenset[str]) -> frozenset[str]:
    return variant - base


def variant_only_record_ids(task_dir: Path) -> frozenset[str]:
    """The records a variant task added: its source record ids that are no
    record id of the base task's sources (the sibling ``<domain>/base``
    folder, which must be present). Empty for a base task."""
    task_dir = Path(task_dir)
    domain = domain_of(task_dir)
    if tier_of(task_dir) == "base":
        return frozenset()
    base_dir = task_dir.resolve().parent / "base"
    if not (base_dir / "input" / "data").is_dir():
        raise FileNotFoundError(
            f"{task_dir} is a variant task: scoring it needs the base task folder next to it "
            f"({base_dir}, with input/data) to tell the variant's own records from the base records")
    return _difference(source_record_ids(task_dir, domain), source_record_ids(base_dir, domain))


# ------------------------------------------------------- value prep

def canon_scalar_str(value: Any) -> Any:
    """Integral floats lose the trailing '.0' (a JSON 42 and a CSV '42' must
    meet as the same string); blanks become None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text if text else None


def canon_author_set(value: Any) -> Any:
    """papers authors: a sorted, deduplicated, casefolded '|'-join, whatever
    the serialization (list, list literal, '|' or ';' separated string)."""
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


def symmetric_prep(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Type coercions applied to BOTH the gold and a submission so the
    comparators see the same representations (years as strings, integral
    floats without '.0', products numerics as numbers, papers author sets)."""
    frame = frame.copy()
    for col in _YEAR_COLS.get(domain, ()):
        if col in frame.columns:
            frame[col] = frame[col].apply(canon_scalar_str)
    if domain == "games":
        if "genres_genre" in frame.columns and "genres" not in frame.columns:
            frame = frame.rename(columns={"genres_genre": "genres"})
    elif domain == "companies":
        def from_sci(v: Any) -> Any:
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
        for col in _PAPERS_SCALARS:
            if col in frame.columns:
                frame[col] = frame[col].apply(canon_scalar_str)
        if "authors" in frame.columns:
            frame["authors"] = frame["authors"].apply(canon_author_set)
    return frame


def is_present(value: Any) -> bool:
    """Does a gold cell carry a value to be graded? NaN/None, '' and an
    empty list do not."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() not in ("", "nan", "None", "[]")


# ------------------------------------------------------------ parsers

def _coerce_xml_value(text: str | None) -> Any:
    """A list literal becomes a list, other text is stripped, blanks are None."""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return stripped


def parse_gold_xml(path: Path, domain: str, id_tag: str = "id") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(records, membership) of a fusion gold XML.

    Values: a scalar element's text (list literals parsed); a list attribute
    (an element with child elements) as its children's texts joined with
    '|'; a blanked or absent cell None. A tag repeated inside one record
    keeps the last occurrence, unless that is empty and exactly one
    occurrence carries text (the companies test file repeats ``<name>`` on
    nine records). Membership: every provenance id plus the record key.
    """
    rows, members = [], []
    for record in ET.parse(path).getroot():
        key_el = record.find(id_tag)
        if key_el is None or key_el.text is None:
            continue
        key = key_el.text.strip()
        row: dict[str, Any] = {"cluster_id": key}
        texts: dict[str, list[str | None]] = {}
        nested: dict[str, str | None] = {}
        record_members: set[str] = {key}
        for child in record:
            if child.tag == id_tag:
                continue
            grandchildren = list(child)
            if grandchildren:
                values = [(g.text or "").strip() for g in grandchildren if (g.text or "").strip()]
                nested[child.tag] = "|".join(values) if values else None
            texts.setdefault(child.tag, []).append(child.text)
            provenance = (child.get("provenance") or "").strip()
            if provenance:
                record_members.update(p.strip() for p in provenance.split("+") if p.strip())
        for tag, occurrences in texts.items():
            value = _coerce_xml_value(occurrences[-1])
            if value is None and len(occurrences) > 1:
                present = [t.strip() for t in occurrences if t and t.strip()]
                if len(present) == 1:
                    value = _coerce_xml_value(present[0])
            if value is None and tag in nested:
                value = nested[tag]
            row[tag] = value
        rows.append(row)
        members += [{"record_id": m, "cluster_id": key} for m in sorted(record_members)]
    return pd.DataFrame(rows), pd.DataFrame(members, columns=["record_id", "cluster_id"])


def load_gold(task_dir: Path, split: str = "test") -> pd.DataFrame:
    """The gold set of a task and split, keyed on ``_anchor_id``, restricted
    to the target-schema attributes, values prepared for the comparators."""
    task_dir = Path(task_dir)
    domain, tier = domain_of(task_dir), tier_of(task_dir)
    version = gold_version(task_dir)
    path = gold_path(task_dir, split)
    if not path.is_file():
        raise FileNotFoundError(f"no fusion gold for {domain}/{tier} ({split}): {path}")
    head = path.read_bytes().lstrip()[:1]
    if head == b"{":                                     # papers: JSON lines
        gold = pd.read_json(path, lines=True)
        gold = symmetric_prep(gold, domain)
        gold[ANCHOR_ID] = gold["source_ids"].map(lambda v: pick_anchor(parse_id_list(v), domain))
        bookkeeping = ["source_ids"]
    elif head == b"<":                                   # XML domains
        id_tag = "source_ids" if domain == "products" else "id"
        records, membership = parse_gold_xml(path, domain, id_tag)
        gold = symmetric_prep(records, domain)
        gold[ANCHOR_ID] = gold["cluster_id"].map(anchors_by_cluster(membership, domain))
        bookkeeping = ["cluster_id"]
    else:                                                # products base: CSV
        gold = pd.read_csv(path)
        gold = symmetric_prep(gold, domain)
        if "filled" in gold.columns:
            gold = gold[gold["filled"] == "y"].copy()
        gold[ANCHOR_ID] = gold["source_ids"].map(lambda v: pick_anchor(parse_id_list(v), domain))
        bookkeeping = ["source_ids"]
    gold = gold.dropna(subset=[ANCHOR_ID])
    keep = set(target_schema_attributes(task_dir)) | {ANCHOR_ID} | set(bookkeeping)
    gold = gold[[c for c in gold.columns if c in keep]].reset_index(drop=True)
    gold.attrs["domain"], gold.attrs["tier"], gold.attrs["gold_version"] = domain, tier, version
    return gold


def graded_cells(gold: pd.DataFrame, attributes: list[str]) -> int:
    return sum(int(gold[a].map(is_present).sum()) for a in attributes if a in gold.columns)
