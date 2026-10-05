"""Every generator consumer of fusion gold accepts source_ids-keyed gold.

The benchmark's papers gold (every tier) and the products task variants key
a record by its ``source_ids`` member list, with no ``id`` and no DOI. These
tests pin, per consumer, that such a record is keyed on its anchor (the
key the public scorer grades on) and that id-keyed gold (companies / games /
music, the generator's products copy) is read exactly as before:

* ``variant_loader._load_fusion_file`` / ``_load_fusion``
* ``fusion_perfect_clusters.build_perfect_clusters``
* ``protection._load_fusion_protected_ids`` / ``load_fusion_target_values``
  (and through them K3 / K4 / K6 / silver targets)
* ``reliability.load_fusion_gold`` (K10)
* ``committee_norm_c12._load_targets``
* ``scripts.downsample_domain._collect_fusion_ids``
* ``scripts.build_statistics._xml_entity_count``

plus real-data checks against the installed task gold (read-only).
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib import fusion_perfect_clusters as fpc  # noqa: E402
from usecases_synthetic.lib import protection  # noqa: E402
from usecases_synthetic.lib.committee_norm_c12 import _load_targets  # noqa: E402
from usecases_synthetic.lib.fusion_gold_keys import (  # noqa: E402
    gold_anchor,
    source_id_members,
)
from usecases_synthetic.lib.reliability import load_fusion_gold  # noqa: E402
from usecases_synthetic.lib.variant_loader import (  # noqa: E402
    VariantBundle,
    _load_fusion,
    _load_fusion_file,
)

USE_CASES = REPO_ROOT / "use cases"

# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

_PAPERS_JSONL = [
    {"source_ids": ["dblp-00007", "crossref-00002", "open_alex-00009"],
     "title": "Paper A", "authors": ["X Y"], "publication_year": 2019,
     "publisher": "ACM", "cited_by_count": 3},
    # dblp member dropped (a hard-tier re-anchor): crossref is the anchor
    {"source_ids": ["open_alex-00004", "crossref-00008", "crossref-00001"],
     "title": "Paper B", "authors": [], "publication_year": None},
    {"source_ids": [], "title": "unkeyed"},
]

_PRODUCTS_VARIANT_XML = """<?xml version='1.0' encoding='utf-8'?>
<products>
  <product>
    <source_ids>products_1_11,products_2_22,products_3_33</source_ids>
    <brand provenance="products_1_11+products_2_22">ADATA</brand>
    <storage_gb provenance="products_1_11">32.0</storage_gb>
    <color provenance="" v2_removed="v2a-absent" />
  </product>
  <product>
    <source_ids>products_3_5,products_2_9,products_2_10</source_ids>
    <brand provenance="products_2_9">Samsung</brand>
  </product>
</products>
"""

_ID_KEYED_XML = """<?xml version='1.0' encoding='utf-8'?>
<games>
  <game>
    <id>metacritic_1</id>
    <name provenance="metacritic_1+dbpedia_2">Doom</name>
    <genres provenance="metacritic_1"><genre>Shooter</genre><genre>Action</genre></genres>
  </game>
  <game>
    <id>dbpedia_7</id>
    <name provenance="dbpedia_7+sales_3">Quake</name>
  </game>
</games>
"""


def _write_papers_jsonl(path: Path) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in _PAPERS_JSONL) + "\n", encoding="utf-8")
    return path


def _stub_config(paths: list[Path]) -> SimpleNamespace:
    return SimpleNamespace(fusion_paths=lambda: list(paths))


# ---------------------------------------------------------------------------
# variant_loader
# ---------------------------------------------------------------------------


def test_load_fusion_file_jsonl_under_xml_name_joins_source_ids(tmp_path: Path) -> None:
    path = _write_papers_jsonl(tmp_path / "test_set.xml")  # papers variant naming
    df = _load_fusion_file(path, "fusion_test_set", "papers")
    assert "id" not in df.columns
    assert df["source_ids"].tolist() == [
        "dblp-00007,crossref-00002,open_alex-00009",
        "crossref-00001,open_alex-00004,crossref-00008",
        None,
    ]
    assert df.attrs["dataset_name"] == "fusion_test_set"


def test_load_fusion_products_variant_xml_without_id(tmp_path: Path) -> None:
    (tmp_path / "test_set.xml").write_text(_PRODUCTS_VARIANT_XML, encoding="utf-8")
    (tmp_path / "validation_set.xml").write_text(_PRODUCTS_VARIANT_XML, encoding="utf-8")
    gold, val = _load_fusion(tmp_path, domain="products")
    assert "id" not in gold.columns
    assert gold["source_ids"].tolist() == [
        "products_1_11,products_2_22,products_3_33",
        "products_2_10,products_3_5,products_2_9",
    ]
    assert val is not None and val["source_ids"].tolist() == gold["source_ids"].tolist()


def test_load_fusion_id_keyed_xml_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "test_set.xml"
    path.write_text(_ID_KEYED_XML, encoding="utf-8")
    df = _load_fusion_file(path, "fusion_test_set", "games")
    assert "source_ids" not in df.columns
    assert df["id"].tolist() == ["metacritic_1", "dbpedia_7"]


# ---------------------------------------------------------------------------
# fusion_perfect_clusters
# ---------------------------------------------------------------------------


def _bundle(domain: str, fusion_gold: pd.DataFrame) -> VariantBundle:
    return VariantBundle(
        domain=domain,
        level="hard",
        sources={},
        target_schema={},
        sm_mapping=None,
        em_gold={},
        em_splits={},
        fusion_gold=fusion_gold,
        fusion_validation=None,
        pooled_positives=None,
        variant_root=Path("/nonexistent"),
    )


@pytest.fixture
def tiny_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pool_dir = tmp_path / "pools"
    for domain, pairs in {
        "papers": [("dblp-00007", "crossref-00002"), ("crossref-00002", "open_alex-00009"),
                   ("crossref-00001", "open_alex-00077")],
        "games": [("metacritic_1", "dbpedia_2"), ("dbpedia_2", "sales_3")],
    }.items():
        (pool_dir / domain).mkdir(parents=True)
        pd.DataFrame(pairs, columns=["id1", "id2"]).to_csv(
            pool_dir / domain / "pooled_positives.csv", index=False
        )
    monkeypatch.setattr(fpc, "POOL_DIR", pool_dir)
    return pool_dir


def test_perfect_clusters_source_ids_keyed(tiny_pool: Path) -> None:
    gold = pd.DataFrame({"source_ids": [
        "dblp-00007,crossref-00002",                        # pool adds open_alex-00009
        ["open_alex-00004", "crossref-00008", "crossref-00001"],  # list form; pool adds open_alex-00077
        None,                                               # unkeyed row: skipped
    ], "title": ["A", "B", "C"]})
    clusters = fpc.build_perfect_clusters("papers", _bundle("papers", gold))
    assert clusters == {
        "dblp-00007": {"dblp-00007", "crossref-00002", "open_alex-00009"},
        "crossref-00001": {"crossref-00001", "crossref-00008", "open_alex-00004",
                           "open_alex-00077"},
    }
    corr = fpc.build_perfect_clusters_correspondences("papers", _bundle("papers", gold))
    assert len(corr) == (3 - 1) + (4 - 1)
    assert set(corr["score"]) == {1.0}


def test_perfect_clusters_id_keyed_unchanged(tiny_pool: Path) -> None:
    gold = pd.DataFrame({"id": ["metacritic_1", "sales_9"], "name": ["Doom", "Solo"]})
    clusters = fpc.build_perfect_clusters("games", _bundle("games", gold))
    assert clusters == {
        "metacritic_1": {"metacritic_1", "dbpedia_2", "sales_3"},
        "sales_9": {"sales_9"},
    }


def test_perfect_clusters_without_key_column_raises(tiny_pool: Path) -> None:
    gold = pd.DataFrame({"doi": ["10.1/x"], "title": ["A"]})
    with pytest.raises(KeyError, match="source_ids"):
        fpc.build_perfect_clusters("papers", _bundle("papers", gold))


# ---------------------------------------------------------------------------
# protection (K3 / K4 / K6 / silver targets go through these two loaders)
# ---------------------------------------------------------------------------


def test_protection_papers_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_papers_jsonl(tmp_path / "fusion_test.jsonl")
    monkeypatch.setattr(protection, "load_domain_config", lambda d: _stub_config([path]))
    assert protection._load_fusion_protected_ids("papers") == {"dblp-00007", "crossref-00001"}
    targets = protection.load_fusion_target_values("papers")
    assert targets == {
        "dblp-00007": {"title": ["Paper A"], "authors": ["X Y"], "publication_year": ["2019"]},
        "crossref-00001": {"title": ["Paper B"]},
    }


def test_protection_products_variant_xml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "test_set.xml"
    path.write_text(_PRODUCTS_VARIANT_XML, encoding="utf-8")
    monkeypatch.setattr(protection, "load_domain_config", lambda d: _stub_config([path]))
    assert protection._load_fusion_protected_ids("products") == {"products_1_11", "products_2_10"}
    targets = protection.load_fusion_target_values("products")
    assert targets == {
        "products_1_11": {"brand": ["ADATA"], "storage_gb": ["32.0"]},
        "products_2_10": {"brand": ["Samsung"]},
    }


def test_protection_id_keyed_xml_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "test_set.xml"
    path.write_text(_ID_KEYED_XML, encoding="utf-8")
    monkeypatch.setattr(protection, "load_domain_config", lambda d: _stub_config([path]))
    assert protection._load_fusion_protected_ids("games") == {"metacritic_1", "dbpedia_7"}
    assert protection.load_fusion_target_values("games") == {
        "metacritic_1": {"name": ["Doom"], "genres": ["Shooter", "Action"]},
        "dbpedia_7": {"name": ["Quake"]},
    }


def test_protection_id_keyed_record_keeps_legacy_source_ids_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generator's products copy has <id> AND <source_ids>: it stays
    keyed on <id> and parses exactly as before (the <source_ids> child is
    not skipped for an id-keyed record)."""
    path = tmp_path / "test_set.xml"
    path.write_text(
        "<products><product><id>products_1_11</id>"
        "<source_ids>products_1_11,products_2_22</source_ids>"
        "<brand>ADATA</brand></product></products>",
        encoding="utf-8",
    )
    monkeypatch.setattr(protection, "load_domain_config", lambda d: _stub_config([path]))
    assert protection._load_fusion_protected_ids("products") == {"products_1_11"}
    assert protection.load_fusion_target_values("products") == {
        "products_1_11": {"source_ids": ["products_1_11,products_2_22"], "brand": ["ADATA"]},
    }


# ---------------------------------------------------------------------------
# reliability (K10) and the normalization committee targets
# ---------------------------------------------------------------------------


def test_reliability_load_fusion_gold(tmp_path: Path) -> None:
    papers = _write_papers_jsonl(tmp_path / "validation_set.xml")  # JSONL under an .xml name
    assert load_fusion_gold(papers, "papers") == {
        "dblp-00007": {"title": "Paper A", "authors": "X Y", "publication_year": "2019"},
        "crossref-00001": {"title": "Paper B"},
    }
    # domain inferred from the member prefixes when not given
    assert set(load_fusion_gold(papers)) == {"dblp-00007", "crossref-00001"}

    products = tmp_path / "test_set.xml"
    products.write_text(_PRODUCTS_VARIANT_XML, encoding="utf-8")
    assert load_fusion_gold(products, "products") == {
        "products_1_11": {"brand": "ADATA", "storage_gb": "32.0"},
        "products_2_10": {"brand": "Samsung"},
    }

    games = tmp_path / "games_test_set.xml"
    games.write_text(_ID_KEYED_XML, encoding="utf-8")
    assert load_fusion_gold(games) == {
        "metacritic_1": {"name": "Doom"},
        "dbpedia_7": {"name": "Quake"},
    }


def test_norm_c12_load_targets(tmp_path: Path) -> None:
    papers = _write_papers_jsonl(tmp_path / "test_set.xml")
    assert _load_targets("papers", papers) == {
        "dblp-00007": {"title": ["Paper A"], "authors": ["X Y"], "publication_year": ["2019"]},
        "crossref-00001": {"title": ["Paper B"]},
    }
    products = tmp_path / "validation_set.xml"
    products.write_text(_PRODUCTS_VARIANT_XML, encoding="utf-8")
    assert _load_targets("products", products) == {
        "products_1_11": {"brand": ["ADATA"], "storage_gb": ["32.0"]},
        "products_2_10": {"brand": ["Samsung"]},
    }
    games = tmp_path / "games.xml"
    games.write_text(_ID_KEYED_XML, encoding="utf-8")
    assert _load_targets("games", games) == {
        "metacritic_1": {"name": ["Doom"], "genres": ["Shooter", "Action"]},
        "dbpedia_7": {"name": ["Quake"]},
    }
    assert _load_targets("games", tmp_path / "absent.xml") == {}


# ---------------------------------------------------------------------------
# scripts: downsample_domain + build_statistics
# ---------------------------------------------------------------------------


def test_downsample_collect_fusion_ids(tmp_path: Path) -> None:
    from usecases_synthetic.scripts.downsample_domain import _collect_fusion_ids

    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    _write_papers_jsonl(papers_dir / "fusion_test.jsonl")
    got = _collect_fusion_ids(
        papers_dir, {"dblp": "dblp-", "crossref": "crossref-", "open_alex": "open_alex-"}
    )
    assert got == {
        "dblp": {"dblp-00007"},
        "crossref": {"crossref-00002", "crossref-00008", "crossref-00001"},
        "open_alex": {"open_alex-00009", "open_alex-00004"},
    }

    products_dir = tmp_path / "products"
    products_dir.mkdir()
    (products_dir / "test_set.xml").write_text(_PRODUCTS_VARIANT_XML, encoding="utf-8")
    got = _collect_fusion_ids(
        products_dir, {f"products_{i}": f"products_{i}_" for i in (1, 2, 3, 4)}
    )
    assert got == {
        "products_1": {"products_1_11"},
        "products_2": {"products_2_22", "products_2_9", "products_2_10"},
        "products_3": {"products_3_33", "products_3_5"},
        "products_4": set(),
    }

    games_dir = tmp_path / "games"
    games_dir.mkdir()
    (games_dir / "test_set.xml").write_text(_ID_KEYED_XML, encoding="utf-8")
    got = _collect_fusion_ids(
        games_dir, {"metacritic": "metacritic_", "dbpedia": "dbpedia_", "sales": "sales_"}
    )
    assert got == {
        "metacritic": {"metacritic_1"},
        "dbpedia": {"dbpedia_2", "dbpedia_7"},
        "sales": {"sales_3"},
    }


def test_build_statistics_counts_json_gold(tmp_path: Path) -> None:
    from usecases_synthetic.scripts.build_statistics import _xml_entity_count

    assert _xml_entity_count(_write_papers_jsonl(tmp_path / "fusion_val.jsonl")) == 3
    xml = tmp_path / "test_set.xml"
    xml.write_text(_PRODUCTS_VARIANT_XML, encoding="utf-8")
    assert _xml_entity_count(xml) == 2
    assert _xml_entity_count(tmp_path / "absent.xml") is None


# ---------------------------------------------------------------------------
# Real task gold (read-only): generator anchors == public scorer anchors
# ---------------------------------------------------------------------------

_REAL_CASES = [
    ("papers", "base", "fusion_test.jsonl", "test"),
    ("papers", "hard", "test_set.xml", "test"),
    ("papers", "hard", "validation_set.xml", "validation"),
    ("products", "hard", "test_set.xml", "test"),
    ("products", "hard", "validation_set.xml", "validation"),
]


@pytest.mark.parametrize(("domain", "tier", "name", "split"), _REAL_CASES)
def test_real_gold_anchors_match_public_scorer(
    domain: str, tier: str, name: str, split: str
) -> None:
    task = USE_CASES / domain / tier
    path = task / "input" / "fusion" / name
    if not path.is_file():
        pytest.skip(f"{path} not present in this checkout")
    fusion_gold = pytest.importorskip("madi_bench.evaluation.fusion_gold")
    frame = _load_fusion_file(path, "gold", domain)
    assert "id" not in frame.columns
    ours = [gold_anchor(source_id_members(v), domain) for v in frame["source_ids"]]
    assert all(ours)
    # anchor first in the evaluator-ready string
    assert [str(v).split(",")[0] for v in frame["source_ids"]] == ours
    theirs = fusion_gold.load_gold(task, split)["_anchor_id"].tolist()
    assert sorted(ours) == sorted(theirs)


def test_real_products_synthetic_copy_unchanged() -> None:
    """The generator's products copy (<id> + <source_ids>) loads exactly as
    before: its source_ids strings are already anchor-first, and every <id>
    is the anchor of its members."""
    from usecases_synthetic.lib.domain_config import load_domain_config

    for path in load_domain_config("products").fusion_paths():
        if not path.is_file():
            pytest.skip(f"{path} not present in this checkout")
        root = ET.parse(path).getroot()
        raw = [(rec.findtext("source_ids") or "").strip() for rec in root]
        loaded = _load_fusion_file(path, "gold", "products")
        assert loaded["source_ids"].tolist() == raw
        assert loaded["id"].tolist() == [
            gold_anchor(source_id_members(s), "products") for s in raw
        ]
