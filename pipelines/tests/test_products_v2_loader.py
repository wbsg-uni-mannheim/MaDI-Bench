"""Products base + variant loading on the public v2 layout.

Covers the best-of-breed pipeline's products path:

* base task ``use cases/products/base/``: EM gold from ``input/entitymatching``
  (bare ids prefixed like the sources), fusion gold from
  ``fusion_{test,validation}_set.csv`` keyed on the anchor of
  ``source_ids``, SM gold from ``sm_mapping_gold.json``;
* variants ``use cases/products/<level>/``: fusion gold XML with
  ``<source_ids>`` and no ``<id>``;
* the sources' ``cluster_id`` (the WDC gold grouping) never
  reaches a stage;
* the e2e panel's workflow silver, keyed on the anchor;
* ``--no-llm-sm`` really removes the SM LLM members.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from madi_bench.evaluation.fusion_gold import load_gold, parse_id_list, pick_anchor
from pipelines.lib.bundle import load_pipeline_bundle
from pipelines.lib.canonical_loader import (
    PRODUCTS_GOLD_GROUPING_COLUMN,
    load_products_workflow_silver,
    prepare_products_fusion_gold,
)
from usecases_synthetic.lib.domain_config import task_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTEE_DIR = REPO_ROOT / "usecases_synthetic" / "config" / "committees"
BASE = task_dir("products")
VARIANTS = ("easy", "medium", "hard")
EM_STEMS = {
    ("products_1", "products_2"): "prod1_to_prod2",
    ("products_1", "products_3"): "prod1_to_prod3",
    ("products_1", "products_4"): "prod1_to_prod4",
}


@pytest.fixture(scope="module")
def base_bundle():
    return load_pipeline_bundle("products", bundle_source="canonical")


@pytest.fixture(scope="module", params=VARIANTS)
def variant_bundle(request):
    return request.param, load_pipeline_bundle(
        "products", level=request.param, bundle_source="canonical"
    )


def _fusion_committee() -> dict:
    return yaml.safe_load((COMMITTEE_DIR / "fusion_committee_products.yaml").read_text())


def _assert_anchor_keyed(frame: pd.DataFrame) -> None:
    for anchor, members in zip(frame["id"], frame["source_ids"]):
        ids = parse_id_list(members)
        assert pick_anchor(ids, "products") == anchor
        assert ids[0] == anchor, "anchor must be listed first in source_ids"
    assert frame["id"].is_unique


# ------------------------------------------------------------------ base


def test_base_bundle_reads_public_task_folder(base_bundle) -> None:
    assert base_bundle.variant_root == BASE
    assert base_bundle.sm_task_root == BASE  # schema matching: same task, as shipped
    assert base_bundle.level == "baseline"
    assert set(base_bundle.sources) == {f"products_{n}" for n in (1, 2, 3, 4)}


def test_base_sources_drop_gold_grouping_and_keep_raw_names(base_bundle) -> None:
    raw_brand = {"products_1": "manufacturer", "products_2": "brandName",
                 "products_3": "Brand", "products_4": "mfr"}
    for name, df in base_bundle.sources.items():
        assert PRODUCTS_GOLD_GROUPING_COLUMN not in df.columns, name
        assert raw_brand[name] in df.columns
        assert df["id"].astype(str).str.startswith(f"{name}_").all()
        assert df["id"].is_unique
        assert df.attrs.get("dataset_name") == name


def test_base_em_gold_from_entitymatching_resolves_to_sources(base_bundle) -> None:
    em_dir = BASE / "input" / "entitymatching"
    assert set(base_bundle.em_gold) == set(EM_STEMS)
    for pair, stem in EM_STEMS.items():
        src1, src2 = pair
        ids1 = set(base_bundle.sources[src1]["id"])
        ids2 = set(base_bundle.sources[src2]["id"])
        splits = base_bundle.em_splits[pair]
        assert set(splits) == {"train", "val", "test", "all"}
        for split, frame in splits.items():
            on_disk = pd.read_csv(em_dir / f"{stem}_{split}.csv")
            assert len(frame) == len(on_disk)
            assert list(frame.columns) == ["id1", "id2", "label"]
            assert frame["id1"].isin(ids1).all(), f"{pair} {split}: id1 outside {src1}"
            assert frame["id2"].isin(ids2).all(), f"{pair} {split}: id2 outside {src2}"
            assert set(frame["label"]) <= {"true", "false"}
            assert (frame["label"] == "true").sum() == int((on_disk["label"] == 1).sum())
        assert len(base_bundle.em_gold[pair]) == len(splits["all"])


def test_base_fusion_gold_keyed_on_anchor(base_bundle) -> None:
    committee = _fusion_committee()
    for split, frame in (("test", base_bundle.fusion_gold),
                         ("validation", base_bundle.fusion_validation)):
        assert frame is not None and len(frame) == 100
        assert committee["gold_id_column"] in frame.columns
        assert PRODUCTS_GOLD_GROUPING_COLUMN not in frame.columns
        _assert_anchor_keyed(frame)
        # Same anchors as the public scorer's gold.
        public = load_gold(BASE, split)
        assert sorted(frame["id"]) == sorted(public["_anchor_id"])
        # Every graded attribute of the committee is carried.
        assert set(committee["attribute_types"]) <= set(frame.columns)
        members = [m for v in frame["source_ids"] for m in parse_id_list(v)]
        all_ids = set().union(*(set(df["id"]) for df in base_bundle.sources.values()))
        assert set(members) <= all_ids


def test_base_sm_gold_json_matches_raw_source_columns(base_bundle) -> None:
    sm = base_bundle.sm_mapping
    assert sm is not None and len(sm) == 100
    assert PRODUCTS_GOLD_GROUPING_COLUMN not in set(sm["source_column"])
    assert PRODUCTS_GOLD_GROUPING_COLUMN not in set(sm["target_column"])
    for source, column in zip(sm["source_dataset"], sm["source_column"]):
        assert column in base_bundle.sources[source].columns, (source, column)


# -------------------------------------------------------------- variants


def test_variant_bundle_loads_source_ids_gold(variant_bundle) -> None:
    level, bundle = variant_bundle
    root = task_dir("products", level)
    assert bundle.variant_root == root
    assert bundle.sm_task_root == root
    all_ids = set()
    for name, df in bundle.sources.items():
        assert PRODUCTS_GOLD_GROUPING_COLUMN not in df.columns, (level, name)
        all_ids |= set(df["id"].astype(str))
    frames = (("test", bundle.fusion_gold), ("validation", bundle.fusion_validation))
    for split, frame in frames:
        assert frame is not None and len(frame) == 100
        _assert_anchor_keyed(frame)
        assert sorted(frame["id"]) == sorted(load_gold(root, split)["_anchor_id"])
        members = {m for v in frame["source_ids"] for m in parse_id_list(v)}
        assert members <= all_ids, f"{level} {split}: gold members outside the sources"
    assert set(bundle.em_splits) == set(EM_STEMS)


# ---------------------------------------------------------- panel silver


@pytest.mark.parametrize("tier", ("base",) + VARIANTS)
def test_workflow_silver_keyed_on_anchor(tier: str) -> None:
    root = task_dir("products", tier)
    silver = load_products_workflow_silver(root, "test", key_column="_anchor_id")
    fused, membership = silver.fused, silver.membership
    assert len(fused) == 100 and fused["_anchor_id"].is_unique
    assert PRODUCTS_GOLD_GROUPING_COLUMN not in fused.columns
    assert "source_ids" not in fused.columns and "id" not in fused.columns
    assert set(membership["cluster_id"]) == set(fused["_anchor_id"])
    assert set(membership["source"]) <= {f"products_{n}" for n in (1, 2, 3, 4)}
    public = load_gold(root, "test")
    assert set(fused["_anchor_id"]) == set(public["_anchor_id"])
    if tier == "base":
        assert silver.cell_provenance is None
    else:
        assert silver.cell_provenance is not None
        assert set(silver.cell_provenance["cluster_id"]) <= set(fused["_anchor_id"])


def test_workflow_silver_from_synthetic_xml(tmp_path: Path) -> None:
    fusion = tmp_path / "input" / "fusion"
    fusion.mkdir(parents=True)
    (fusion / "test_set.xml").write_text(
        "<products>"
        "<product><id>ignored_1</id>"
        "<source_ids>products_3_9,products_2_5,products_2_10</source_ids>"
        '<brand provenance="products_2_5+products_3_9">ACME</brand>'
        '<color provenance="" v2_removed="v2a-absent" />'
        "</product>"
        "<product><source_ids>products_1_7,products_4_1</source_ids>"
        '<brand provenance="products_1_7">Foo</brand></product>'
        "</products>",
        encoding="utf-8",
    )
    silver = load_products_workflow_silver(tmp_path, "test", key_column="_anchor_id")
    fused = silver.fused.set_index("_anchor_id")
    # Anchor rule: no products_1 member -> smallest products_2 id (lexicographic).
    assert list(fused.index) == ["products_2_10", "products_1_7"]
    assert fused.loc["products_2_10", "brand"] == "ACME"
    assert pd.isna(fused.loc["products_2_10", "color"])
    members = silver.membership.groupby("cluster_id")["record_id"].apply(set).to_dict()
    assert members["products_2_10"] == {"products_2_10", "products_2_5", "products_3_9"}
    assert members["products_1_7"] == {"products_1_7", "products_4_1"}
    assert len(silver.cell_provenance) == 2


def test_prepare_fusion_gold_rules() -> None:
    frame = pd.DataFrame(
        {
            "source_ids": [
                "products_3_9,products_2_5,products_2_10",
                "other_1",
                "products_1_2,products_1_1",
            ],
            "brand": ["A", "B", "C"],
            "cluster_id": [1, 2, 3],
        }
    )
    out = prepare_products_fusion_gold(frame)
    assert list(out["id"]) == ["products_2_10", "products_1_2"]
    assert list(out["source_ids"]) == [
        "products_2_10,products_3_9,products_2_5",
        "products_1_2,products_1_1",
    ]
    assert "cluster_id" not in out.columns
    with pytest.raises(ValueError, match="source_ids"):
        prepare_products_fusion_gold(pd.DataFrame({"id_left": [1], "filled": ["y"]}))


# ---------------------------------------------------------------- config


def test_products_config_keys_panel_on_anchor() -> None:
    from pipelines.lib.pipeline import PipelineConfig

    cfg = PipelineConfig.from_yaml(REPO_ROOT / "pipelines" / "configs" / "products.yaml")
    assert cfg.gold_id_column == "_anchor_id"
    assert PRODUCTS_GOLD_GROUPING_COLUMN not in cfg.column_types
    music = PipelineConfig.from_yaml(REPO_ROOT / "pipelines" / "configs" / "music.yaml")
    assert music.gold_id_column == "cluster_id"


def test_no_llm_sm_removes_llm_members(tmp_path: Path) -> None:
    from pipelines.lib.pipeline import BestOfBreedPipeline, PipelineConfig

    cfg = PipelineConfig.from_yaml(REPO_ROOT / "pipelines" / "configs" / "products.yaml")
    sm_yaml = COMMITTEE_DIR / "sm_committee.yaml"
    off = BestOfBreedPipeline(
        cfg, committee_dir=COMMITTEE_DIR, with_llm_sm=False, out_dir=tmp_path
    )
    filtered = off._maybe_filter_sm_yaml(sm_yaml)
    assert filtered != sm_yaml and filtered.parent == tmp_path / "effective_committees"
    members = yaml.safe_load(filtered.read_text())["members"]
    assert members and all(m.get("signal_type") != "llm" for m in members)
    on = BestOfBreedPipeline(
        cfg, committee_dir=COMMITTEE_DIR, with_llm_sm=True, out_dir=tmp_path
    )
    assert on._maybe_filter_sm_yaml(sm_yaml) == sm_yaml
