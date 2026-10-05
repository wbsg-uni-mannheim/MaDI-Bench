"""Keys and member ids of fusion-gold records (the anchor rule).

The benchmark keys a fusion-gold record in one of two ways:

* an ``id`` (XML ``<id>``) holding the anchor record id -- companies, games
  and music at every tier, and the generator's own products copy
  (``usecases_synthetic/usecases/products``), which carries ``<source_ids>``
  as well;
* a ``source_ids`` list of the member record ids and no id -- papers at
  every tier (``fusion_{test,val}.jsonl`` at base, JSON lines under the
  ``{test,validation}_set.xml`` names in the variants) and the products task
  variants (XML ``<source_ids>`` element, comma-joined).

A ``source_ids`` record is keyed on its ANCHOR, the member the anchor rule
picks: the first id (list order) from the domain's primary source, else the
lexicographically smallest id from the next source in priority order
(``ANCHOR_PRIORITY``: forbes URL > dbpedia > fullcontact, metacritic_ >
dbpedia_ > sales_, mbrainz_ > discogs_ > lastFM_, dblp- > crossref- >
open_alex-, products_1_ > ... > products_4_). That is the key the public
scorer grades on, so the generator's committees, protection sets and
target-value maps key the same record the same way.

``ANCHOR_PRIORITY`` / ``pick_anchor`` / ``parse_id_list`` are the public
scorer's own (``madi_bench.evaluation.fusion_gold``) whenever it is
importable -- the MaDI-Bench checkout, where the scorer sits next to
``usecases_synthetic``. A checkout without the scorer uses the verbatim
mirror below; ``tests/test_fusion_gold_keys.py`` pins that the two agree.
"""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Anchor rule: mirror of madi_bench.evaluation.fusion_gold (kept verbatim)
# ---------------------------------------------------------------------------

MIRROR_ANCHOR_PRIORITY: dict[str, tuple[str, ...]] = {
    "products": ("products_1_", "products_2_", "products_3_", "products_4_"),
    "music": ("mbrainz_", "discogs_", "lastFM_"),
    "games": ("metacritic_", "dbpedia_", "sales_"),
    "companies": ("http://www.forbes.com/", "http://dbpedia.org/", "fullcontact_"),
    "papers": ("dblp-", "crossref-", "open_alex-"),
}


def mirror_parse_id_list(value: Any) -> list[str]:
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


def mirror_pick_anchor(ids: list[str], domain: str, exclude=None) -> str | None:
    """The anchor of a member list: the first id (list order) from the
    primary source, else the lexicographically smallest id from the next
    source in priority order that is present. Ids in ``exclude`` never
    anchor."""
    if exclude:
        ids = [record_id for record_id in ids if record_id not in exclude]
    for rank, prefix in enumerate(MIRROR_ANCHOR_PRIORITY[domain]):
        hits = [record_id for record_id in ids if record_id.startswith(prefix)]
        if hits:
            return hits[0] if rank == 0 else min(hits)
    return None


try:  # the public scorer, when this checkout ships it
    from madi_bench.evaluation.fusion_gold import (  # type: ignore[import-not-found]
        ANCHOR_PRIORITY,
        parse_id_list,
        pick_anchor,
    )

    ANCHOR_RULE_SOURCE = "madi_bench.evaluation.fusion_gold"
except ImportError:  # generator checkout without the scorer
    ANCHOR_PRIORITY = MIRROR_ANCHOR_PRIORITY
    parse_id_list = mirror_parse_id_list
    pick_anchor = mirror_pick_anchor
    ANCHOR_RULE_SOURCE = "usecases_synthetic.lib.fusion_gold_keys (mirror)"


# ---------------------------------------------------------------------------
# Domain resolution
# ---------------------------------------------------------------------------


def anchor_domain(domain: str | None, ids: Iterable[str] = ()) -> str | None:
    """The ``ANCHOR_PRIORITY`` domain for ``domain`` (``<d>-small`` -> ``<d>``);
    when ``domain`` is unknown or None, the domain whose source prefixes
    cover the most of ``ids`` (the prefixes are disjoint across domains)."""
    if domain:
        if domain in ANCHOR_PRIORITY:
            return domain
        base = domain[: -len("-small")] if domain.endswith("-small") else domain
        if base in ANCHOR_PRIORITY:
            return base
    best, best_hits = None, 0
    ids = [str(i) for i in ids]
    for name, prefixes in ANCHOR_PRIORITY.items():
        hits = sum(1 for i in ids if i.startswith(prefixes))
        if hits > best_hits:
            best, best_hits = name, hits
    return best


# ---------------------------------------------------------------------------
# source_ids
# ---------------------------------------------------------------------------


def source_id_members(value: Any) -> list[str]:
    """The member ids of a ``source_ids`` cell (list, list literal or
    comma-joined string), stripped, blanks and duplicates dropped, order
    kept."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in parse_id_list(value):
        rid = str(raw).strip()
        if rid and rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def gold_anchor(ids: Iterable[str], domain: str | None = None) -> str | None:
    """The anchor of a member list (``pick_anchor``); ``None`` when no
    member belongs to a source of the domain."""
    ids = [str(i) for i in ids]
    resolved = anchor_domain(domain, ids)
    if resolved is None:
        return None
    return pick_anchor(ids, resolved)


def join_source_ids(value: Any, domain: str | None = None) -> str | None:
    """A ``source_ids`` cell as the comma-joined string the fusion evaluator
    splits (``PyDI.fusion.evaluation`` aligns an unmatched gold row through
    ``str(source_ids).split(',')``, first hit wins), the anchor first and
    the other members in their original order. ``None`` for an empty cell."""
    members = source_id_members(value)
    if not members:
        return None
    anchor = gold_anchor(members, domain)
    if anchor is not None and members[0] != anchor:
        members = [anchor] + [m for m in members if m != anchor]
    return ",".join(members)


def normalize_source_ids_column(frame: pd.DataFrame, domain: str | None = None) -> pd.DataFrame:
    """Return ``frame`` with its ``source_ids`` column (when present) as
    :func:`join_source_ids` strings. Idempotent; a frame without the column
    is returned unchanged (same object)."""
    if "source_ids" not in frame.columns:
        return frame
    out = frame.copy()
    out["source_ids"] = [join_source_ids(v, domain) for v in frame["source_ids"]]
    out.attrs = dict(frame.attrs)
    return out


def gold_member_ids(frame: pd.DataFrame, id_column: str) -> set[str]:
    """Every record id a gold frame names in ``id_column``: the member ids
    for ``source_ids``, the plain values otherwise."""
    if id_column not in frame.columns:
        return set()
    out: set[str] = set()
    for value in frame[id_column].tolist():
        if id_column == "source_ids":
            out.update(source_id_members(value))
            continue
        try:
            present = bool(pd.notna(value))
        except (TypeError, ValueError):  # a list-like cell
            present = value is not None
        if present:
            out.add(str(value))
    return out


# ---------------------------------------------------------------------------
# Records (JSON and XML)
# ---------------------------------------------------------------------------


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return bool(str(value).strip())


def record_key(record: dict[str, Any], domain: str | None = None) -> str | None:
    """The key of a JSON gold record: its ``id`` when present, else the
    anchor of its ``source_ids``; ``None`` when it has neither."""
    if _present(record.get("id")):
        return str(record["id"]).strip()
    members = source_id_members(record.get("source_ids"))
    return gold_anchor(members, domain) if members else None


def _xml_source_ids(elem: ET.Element) -> list[str]:
    node = elem.find("source_ids")
    if node is None:
        return []
    children = [c.text.strip() for c in node if c.text and c.text.strip()]
    return source_id_members(children if children else (node.text or ""))


def xml_record_key(elem: ET.Element, domain: str | None = None) -> tuple[str | None, str | None]:
    """``(key, key_tag)`` of an XML gold record: its ``<id>`` text (key_tag
    ``"id"``), else the anchor of its ``<source_ids>`` (key_tag
    ``"source_ids"``); ``(None, None)`` when it has neither.

    The attribute loops of the target / gold-value readers skip the key_tag
    child. An id-keyed record therefore parses exactly as before this
    helper existed (only ``<id>`` skipped; the generator's products copy
    keeps its ``<source_ids>`` child in the loop as it always did)."""
    id_elem = elem.find("id")
    if id_elem is not None and id_elem.text and id_elem.text.strip():
        return id_elem.text.strip(), "id"
    members = _xml_source_ids(elem)
    if members:
        return gold_anchor(members, domain), "source_ids"
    return None, None


def xml_record_members(elem: ET.Element) -> list[str]:
    """Every record id an XML gold record names: ``<id>``, the
    ``<source_ids>`` members and the ``provenance`` ids (``+``-joined) of
    its attributes, first-seen order."""
    seen: set[str] = set()
    out: list[str] = []

    def add(rid: str) -> None:
        rid = rid.strip()
        if rid and rid not in seen:
            seen.add(rid)
            out.append(rid)

    id_elem = elem.find("id")
    if id_elem is not None and id_elem.text:
        add(id_elem.text)
    for rid in _xml_source_ids(elem):
        add(rid)
    for child in elem.iter():
        for token in (child.attrib.get("provenance") or "").split("+"):
            add(token)
    return out


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def sniff_fusion_format(path: Path) -> str:
    """``"jsonl"`` / ``"json"`` / ``"xml"`` from the first non-blank byte of
    ``path`` (the papers variants carry JSON lines under ``*.xml`` names),
    else ``"csv"`` for a ``.csv`` file and ``""`` otherwise."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096).lstrip()[:1]
    except OSError:
        return ""
    if head == b"{":
        return "jsonl"
    if head == b"[":
        return "json"
    if head == b"<":
        return "xml"
    return "csv" if Path(path).suffix.lower() == ".csv" else ""


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """The records of a JSON-lines or JSON-array gold file (content-sniffed,
    whatever the file name)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


__all__ = [
    "ANCHOR_PRIORITY",
    "ANCHOR_RULE_SOURCE",
    "MIRROR_ANCHOR_PRIORITY",
    "anchor_domain",
    "gold_anchor",
    "gold_member_ids",
    "join_source_ids",
    "mirror_parse_id_list",
    "mirror_pick_anchor",
    "normalize_source_ids_column",
    "parse_id_list",
    "pick_anchor",
    "read_json_records",
    "record_key",
    "sniff_fusion_format",
    "source_id_members",
    "xml_record_key",
    "xml_record_members",
]
