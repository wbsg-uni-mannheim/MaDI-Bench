"""Unit tests for ``lib/fusion_gold_keys`` (fusion-gold keys, anchors).

Pins that the generator keys a ``source_ids`` gold record exactly as the
public scorer does (``madi_bench.evaluation.fusion_gold.pick_anchor``), that
the verbatim mirror used without the scorer agrees with it, and the record /
file helpers every gold consumer shares.
"""

from __future__ import annotations

import importlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

from usecases_synthetic.lib import fusion_gold_keys as fgk

# ---------------------------------------------------------------------------
# Mirror == public scorer
# ---------------------------------------------------------------------------

_ID_LIST_CASES = [
    ["dblp-00007", "crossref-00001", "open_alex-00002"],
    ["crossref-00009", "crossref-00001", "open_alex-00000"],
    ["open_alex-00003", "open_alex-00001"],
    ["products_3_9", "products_2_7", "products_4_1"],
    ["products_1_20", "products_1_10", "products_2_1"],
    ["discogs_9", "lastFM_1", "discogs_10"],
    ["mbrainz_5", "discogs_1"],
    ["dbpedia_3", "sales_1", "dbpedia_20"],
    ["http://dbpedia.org/resource/B", "fullcontact_1", "http://dbpedia.org/resource/A"],
    ["http://www.forbes.com/companies/x/", "http://dbpedia.org/resource/X"],
    ["unrelated-1"],
    [],
]
_RAW_CELLS = [
    "dblp-1,crossref-2",
    " dblp-1 , crossref-2 ,",
    "['dblp-1', 'crossref-2']",
    '["dblp-1", "crossref-2"]',
    ["dblp-1", "crossref-2"],
    ("products_1_1",),
    "",
    "[]",
    "nan",
    None,
    float("nan"),
    12,
]


def _public():
    try:
        return importlib.import_module("madi_bench.evaluation.fusion_gold")
    except ImportError:  # pragma: no cover - generator checkout without the scorer
        pytest.skip("madi_bench (public scorer) not importable in this checkout")


def test_mirror_priority_equals_public() -> None:
    public = _public()
    assert fgk.MIRROR_ANCHOR_PRIORITY == public.ANCHOR_PRIORITY


@pytest.mark.parametrize("ids", _ID_LIST_CASES)
@pytest.mark.parametrize("domain", sorted(fgk.MIRROR_ANCHOR_PRIORITY))
def test_mirror_pick_anchor_equals_public(ids: list[str], domain: str) -> None:
    public = _public()
    assert fgk.mirror_pick_anchor(list(ids), domain) == public.pick_anchor(list(ids), domain)
    excl = set(ids[:1])
    assert fgk.mirror_pick_anchor(list(ids), domain, exclude=excl) == public.pick_anchor(
        list(ids), domain, exclude=excl
    )


@pytest.mark.parametrize("raw", _RAW_CELLS)
def test_mirror_parse_id_list_equals_public(raw: object) -> None:
    public = _public()
    assert fgk.mirror_parse_id_list(raw) == public.parse_id_list(raw)


def test_uses_public_rule_when_importable() -> None:
    public = _public()
    assert fgk.ANCHOR_RULE_SOURCE == "madi_bench.evaluation.fusion_gold"
    assert fgk.pick_anchor is public.pick_anchor
    assert fgk.ANCHOR_PRIORITY is public.ANCHOR_PRIORITY


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ids", "domain", "anchor"),
    [
        # primary source present: its FIRST id in list order
        (["crossref-1", "dblp-9", "dblp-2"], "papers", "dblp-9"),
        # fallback source: the lexicographically SMALLEST id
        (["open_alex-1", "crossref-9", "crossref-2"], "papers", "crossref-2"),
        (["open_alex-5", "open_alex-3"], "papers", "open_alex-3"),
        (["products_2_9", "products_3_1", "products_2_10"], "products", "products_2_10"),
        (["products_4_1", "products_1_7"], "products", "products_1_7"),
        (["lastFM_1", "discogs_2"], "music", "discogs_2"),
        # alias: <d>-small resolves to <d>
        (["products_2_9", "products_1_1"], "products-small", "products_1_1"),
        # domain inferred from the prefixes when omitted
        (["open_alex-5", "crossref-7"], None, "crossref-7"),
        # no member of a known source
        (["zzz-1"], "papers", None),
        ([], "papers", None),
    ],
)
def test_gold_anchor(ids: list[str], domain: str | None, anchor: str | None) -> None:
    assert fgk.gold_anchor(ids, domain) == anchor


def test_anchor_domain() -> None:
    assert fgk.anchor_domain("games") == "games"
    assert fgk.anchor_domain("music-small") == "music"
    assert fgk.anchor_domain("nonexistent", ["dblp-1", "crossref-1"]) == "papers"
    assert fgk.anchor_domain(None, ["products_1_1"]) == "products"
    assert fgk.anchor_domain(None, ["nothing"]) is None


# ---------------------------------------------------------------------------
# source_ids cells and frames
# ---------------------------------------------------------------------------


def test_source_id_members_strips_dedups_keeps_order() -> None:
    assert fgk.source_id_members(["b", " a ", "b", ""]) == ["b", "a"]
    assert fgk.source_id_members("x-1, y-2,x-1") == ["x-1", "y-2"]
    assert fgk.source_id_members(None) == []
    assert fgk.source_id_members(float("nan")) == []


@pytest.mark.parametrize(
    ("cell", "joined"),
    [
        (["dblp-1", "crossref-2", "open_alex-3"], "dblp-1,crossref-2,open_alex-3"),
        (["crossref-9", "open_alex-3", "crossref-2"], "crossref-2,crossref-9,open_alex-3"),
        ("products_2_5,products_1_9", "products_1_9,products_2_5"),
        ("products_1_1,products_2_2", "products_1_1,products_2_2"),
        ("['dblp-1', 'crossref-2']", "dblp-1,crossref-2"),
        ([], None),
        ("", None),
        (None, None),
    ],
)
def test_join_source_ids_anchor_first(cell: object, joined: str | None) -> None:
    assert fgk.join_source_ids(cell) == joined
    if joined is not None:  # idempotent
        assert fgk.join_source_ids(joined) == joined


def test_normalize_source_ids_column() -> None:
    no_column = pd.DataFrame({"id": ["a"], "name": ["x"]})
    assert fgk.normalize_source_ids_column(no_column, "papers") is no_column

    frame = pd.DataFrame(
        {"source_ids": [["crossref-2", "dblp-1"], None], "title": ["t1", "t2"]}
    )
    frame.attrs["dataset_name"] = "fusion_test_set"
    out = fgk.normalize_source_ids_column(frame, "papers")
    assert out["source_ids"].tolist() == ["dblp-1,crossref-2", None]
    assert out.attrs["dataset_name"] == "fusion_test_set"
    assert frame["source_ids"].iloc[0] == ["crossref-2", "dblp-1"]  # input untouched


def test_gold_member_ids() -> None:
    frame = pd.DataFrame(
        {"source_ids": ["dblp-1,crossref-2", ["open_alex-3"], None], "id": ["a", None, "c"]}
    )
    assert fgk.gold_member_ids(frame, "source_ids") == {"dblp-1", "crossref-2", "open_alex-3"}
    assert fgk.gold_member_ids(frame, "id") == {"a", "c"}
    assert fgk.gold_member_ids(frame, "missing") == set()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_record_key() -> None:
    assert fgk.record_key({"id": " x_1 ", "source_ids": ["dblp-1"]}) == "x_1"
    assert fgk.record_key({"source_ids": ["open_alex-2", "crossref-5"]}, "papers") == "crossref-5"
    assert fgk.record_key({"source_ids": []}, "papers") is None
    assert fgk.record_key({"title": "no key"}, "papers") is None


def _elem(xml: str) -> ET.Element:
    return ET.fromstring(xml)


def test_xml_record_key_id_keyed_record_keeps_legacy_key() -> None:
    rec = _elem(
        "<product><id>products_1_5</id>"
        "<source_ids>products_2_1,products_1_5</source_ids><brand>A</brand></product>"
    )
    assert fgk.xml_record_key(rec, "products") == ("products_1_5", "id")


def test_xml_record_key_source_ids_keyed_record() -> None:
    rec = _elem(
        "<product><source_ids>products_3_9,products_2_4,products_2_10</source_ids>"
        '<brand provenance="products_3_9">A</brand></product>'
    )
    assert fgk.xml_record_key(rec, "products") == ("products_2_10", "source_ids")
    nested = _elem(
        "<paper><source_ids><source_id>crossref-2</source_id>"
        "<source_id>dblp-7</source_id></source_ids></paper>"
    )
    assert fgk.xml_record_key(nested, "papers") == ("dblp-7", "source_ids")
    blank_id = _elem("<p><id> </id><source_ids>dblp-3</source_ids></p>")
    assert fgk.xml_record_key(blank_id, "papers") == ("dblp-3", "source_ids")
    assert fgk.xml_record_key(_elem("<p><title>t</title></p>"), "papers") == (None, None)


def test_xml_record_members() -> None:
    rec = _elem(
        "<game><id>metacritic_1</id><source_ids>dbpedia_2</source_ids>"
        '<name provenance="metacritic_1+sales_3">N</name>'
        '<genres provenance="dbpedia_2"><genre provenance="sales_4">x</genre></genres></game>'
    )
    assert fgk.xml_record_members(rec) == [
        "metacritic_1", "dbpedia_2", "sales_3", "sales_4",
    ]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_sniff_and_read_json_records(tmp_path: Path) -> None:
    jsonl_as_xml = tmp_path / "test_set.xml"
    jsonl_as_xml.write_text(
        '\n{"source_ids": ["dblp-1"], "title": "a"}\n\n{"source_ids": ["crossref-2"], "title": "b"}\n',
        encoding="utf-8",
    )
    assert fgk.sniff_fusion_format(jsonl_as_xml) == "jsonl"
    assert [r["title"] for r in fgk.read_json_records(jsonl_as_xml)] == ["a", "b"]

    array = tmp_path / "gold.json"
    array.write_text('[{"id": "x"}, 3]', encoding="utf-8")
    assert fgk.sniff_fusion_format(array) == "json"
    assert fgk.read_json_records(array) == [{"id": "x"}]

    xml = tmp_path / "validation_set.xml"
    xml.write_text("  <?xml version='1.0'?><r/>", encoding="utf-8")
    assert fgk.sniff_fusion_format(xml) == "xml"

    csv = tmp_path / "fusion_test_set.csv"
    csv.write_text("source_ids,brand\n", encoding="utf-8")
    assert fgk.sniff_fusion_format(csv) == "csv"
    assert fgk.sniff_fusion_format(tmp_path / "missing.xml") == ""
