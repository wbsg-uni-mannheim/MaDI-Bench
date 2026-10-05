"""Tests for the baseline SM-gold generator.

Confirms the regenerated ``sm_mapping_gold.csv`` is the K8 ``sm_mapping``
ground truth with the original (un-renamed) source column names — so the
baseline SM committee scores on the same attribute set + target mapping
the variant ``sm_mapping.csv`` is generated from (apples-to-apples).
"""

from __future__ import annotations

import pandas as pd

from usecases_synthetic.lib.domain_config import load_domain_config, load_knob_config
from usecases_synthetic.lib.variant_loader import _load_sm_mapping as _load_sm_gold
from usecases_synthetic.lib.variant_loader import _reconcile_sm_gold_source_columns
from usecases_synthetic.scripts.build_sm_mapping_gold import (
    build_sm_mapping_gold,
    gold_path,
)


def _truth_triples(domain: str) -> set[tuple[str, str, str]]:
    sm = load_knob_config(8, domain)["sm_mapping"]
    return {
        (source, col, target)
        for source, mapping in sm.items()
        for col, target in mapping.items()
    }


class TestBuildSmMappingGold:
    def test_products_full_scope(self) -> None:
        df = build_sm_mapping_gold("products")
        # 4 sources x 25 columns (the as-extracted native schemas of the
        # sources: 24 target attributes + id). cluster_id stays in
        # the data but is not a target attribute, so it is not SM gold
        # (the published base gold has the same 100 rows).
        assert len(df) == 100
        assert "cluster_id" not in set(df["target_column"])
        assert list(df.columns) == [
            "source_dataset",
            "source_column",
            "target_dataset",
            "target_column",
            "score",
        ]
        assert (df["score"] == 1.0).all()
        assert (df["target_dataset"] == "products").all()

    def test_products_includes_r1_attrs_with_original_names(self) -> None:
        df = build_sm_mapping_gold("products")
        cols = set(df["source_column"])
        # The products extension attrs are present as targets...
        assert {"chipset_name", "write_speed_mb_s", "vram_gb"} <= set(df["target_column"])
        # ...mapped from the sources' ORIGINAL (native) column names, not K8
        # rename tokens (products_1 / products_3 / products_4 spellings).
        assert {"video_memory_gb", "MemorySizeGB", "rd_mbs"} <= cols
        assert "t_desc" not in cols  # abbreviated rename for title_description
        assert "prd_ttl" not in cols  # abbreviated rename for title

    def test_matches_k8_ground_truth(self) -> None:
        df = build_sm_mapping_gold("products")
        triples = {
            (r.source_dataset, r.source_column, r.target_column)
            for r in df.itertuples(index=False)
        }
        assert triples == _truth_triples("products")

    def test_other_domains_already_full_scope(self) -> None:
        """companies/games/music gold on disk already equals the K8 truth,
        so the generator is a content no-op there.

        The published gold names each source's raw on-disk columns, including
        the raw id column (``entity_uri`` / ``wiki_ref`` / ``rec_uid`` ...),
        while the K8 ``sm_mapping`` is keyed on the loaded frames, where the
        loader has renamed that id column to ``id``. Reconcile the disk gold
        the same way the variant loader does before comparing, so the check
        compares the same column vocabulary.
        """
        for domain in ("companies", "games", "music"):
            gen = build_sm_mapping_gold(domain)
            gen_set = {
                (r.source_dataset, r.source_column, r.target_column)
                for r in gen.itertuples(index=False)
            }
            # the public tree ships the gold as sm_mapping_gold.json (the CSV
            # is the legacy form): read it through the loader that handles both
            disk = _load_sm_gold(gold_path(domain).parent, baseline=True)
            assert disk is not None, domain
            disk = _reconcile_sm_gold_source_columns(
                disk, load_domain_config(domain), domain
            )
            assert disk is not None, domain
            disk_set = {
                (str(r.source_dataset), str(r.source_column), str(r.target_column))
                for r in disk.itertuples(index=False)
            }
            assert gen_set == disk_set, domain

    def test_products_gold_on_disk_refreshed(self) -> None:
        """The committed products gold carries the full 100-row scope."""
        disk = pd.read_csv(gold_path("products"))
        assert len(disk) == 100
