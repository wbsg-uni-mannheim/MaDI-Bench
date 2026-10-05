"""P2's SM stage reads the task as shipped.

The pipeline bundles rename / re-key the sources for the later stages
(products: ``products_<n>_`` id prefixes, anchor ``id`` in the fusion
frames; papers: raw -> target names). ``run_sm`` must score the SM members on
the shipped files instead, i.e. on the same
:class:`usecases_synthetic.lib.sm_view.SMView` the committee reads, so the P2
SM numbers and the committee SM numbers agree for the same task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pipelines.lib.bundle import PipelineState, load_pipeline_bundle
from pipelines.lib.stage_runners import run_sm
from usecases_synthetic.lib.committee_sm import SMCommitteeRunner
from usecases_synthetic.lib.domain_config import task_dir
from usecases_synthetic.lib.sm_view import SM_INPUT_TAG
from usecases_synthetic.lib.variant_loader import load_variant

REPO_ROOT = Path(__file__).resolve().parents[2]
SM_ROSTER = REPO_ROOT / "usecases_synthetic" / "config" / "committees" / "sm_committee.yaml"


def _roster(tmp_path: Path, names: tuple[str, ...]) -> Path:
    raw = yaml.safe_load(SM_ROSTER.read_text(encoding="utf-8"))
    raw["members"] = [m for m in raw["members"] if m["name"] in names]
    path = tmp_path / "sm_committee.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _raw_columns(domain: str) -> dict[str, set[str]]:
    sm_dir = task_dir(domain) / "input" / "schemamatching"
    files = json.loads((sm_dir / "sm_mapping_gold.json").read_text())["source_files"]
    out: dict[str, set[str]] = {}
    for src, rel in files.items():
        records = json.loads((sm_dir / rel).read_text(encoding="utf-8"))
        out[src] = {k for rec in records for k in rec}
    return out


def test_products_run_sm_reads_the_shipped_task_like_the_committee(tmp_path: Path) -> None:
    roster = _roster(tmp_path, ("label_jw", "instance_tf_cosine", "duplicate_majority"))

    bundle = load_pipeline_bundle("products", bundle_source="canonical")
    # The bundle the later P2 stages read: prefixed ids, anchor id in fusion.
    assert bundle.sources["products_1"]["id"].astype(str).str.startswith("products_1_").all()
    assert bundle.fusion_gold["id"].notna().all()

    state = PipelineState(bundle=bundle)
    sel = run_sm(state, sm_yaml=roster, with_llm=False)
    assert sel.notes["sm_input"] == SM_INPUT_TAG
    raw = _raw_columns("products")
    pred = state.sm_mapping_df
    assert not pred.empty
    for src, col in zip(pred["source_dataset"], pred["source_column"]):
        assert col in raw[src] and col != "cluster_id", (src, col)

    # The committee's products base bundle is the generator copy, yet its SM
    # stage reads the same shipped task: identical per-member scores.
    committee = SMCommitteeRunner(roster).run(load_variant("products", "baseline"))
    assert {k: m.metrics["f1"] for k, m in committee.per_member.items()} == pytest.approx(
        sel.per_member_val
    )


def test_papers_canonical_bundle_names_its_task_for_sm() -> None:
    bundle = load_pipeline_bundle("papers", bundle_source="canonical")
    assert bundle.sm_task_root == task_dir("papers")
    # The later stages keep the canonicalised frames.
    assert "title" in bundle.sources["dblp"].columns
