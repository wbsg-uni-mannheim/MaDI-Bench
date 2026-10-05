"""The schema-matching stage reads the tasks as shipped.

The committee bundles rename source columns for the later stages (every
``id_column`` -> ``id``, papers raw -> target names + all-NA padding, products
base from the generator copy). Schema matching must not see those renames:
they hand every matcher free correspondences. These tests pin, on the real
task files:

* the SM view (:mod:`usecases_synthetic.lib.sm_view`) of every task has the
  shipped column names and scores against the shipped SM gold, per domain
  shape: id-renaming domains (games / companies / music), papers
  (canonicalised + padded by the bundle loader), products base (generator
  copy in the bundle), and the 15 variants;
* the committee runner reads the view (not the bundle frames), including the
  duplicate-based member (record-id hint);
* the bundles the other stages read keep their renamed frames and reconciled
  gold (the fix changes nothing outside schema matching).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from PyDI.schemamatching.base import get_schema_columns
from usecases_synthetic.lib.committee_sm import (
    SMCommitteeRunner,
    _target_df_from_schema,
    _with_record_id_hint,
)
from usecases_synthetic.lib.domain_config import (
    PUBLIC_USECASES_DIR,
    REPO_ROOT,
    load_domain_config,
    task_dir,
)
from usecases_synthetic.lib.loaders import _PAPERS_SOURCE_COLUMN_MAP
from usecases_synthetic.lib.sm_view import (
    SM_INPUT_TAG,
    SM_INPUT_TAG_LEGACY,
    legacy_sm_view,
    load_sm_view,
    load_sm_view_for_bundle,
)
from usecases_synthetic.lib.variant_loader import VariantBundle, load_variant

DOMAINS = ("games", "companies", "music", "products", "papers")
LEVELS = ("baseline", "easy", "medium", "hard")
ID_RENAMING_DOMAINS = ("games", "companies", "music")
SM_ROSTER = REPO_ROOT / "usecases_synthetic" / "config" / "committees" / "sm_committee.yaml"
GOLD_COLS = ["source_dataset", "source_column", "target_dataset", "target_column"]

pytestmark = pytest.mark.skipif(
    not (PUBLIC_USECASES_DIR / "games" / "base" / "input").is_dir(),
    reason="public MaDI-Bench task tree not present",
)


# ---------------------------------------------------------------------------
# Independent readers of the shipped files (no PyDI, no pandas inference)
# ---------------------------------------------------------------------------


def _shipped_columns(path: Path) -> set[str]:
    """Column names of a shipped source file, read without pandas."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return set(next(csv.reader(f)))
    keys: dict[str, None] = {}
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    keys.update(dict.fromkeys(json.loads(line)))
    else:
        with open(path, encoding="utf-8") as f:
            for rec in json.load(f):
                keys.update(dict.fromkeys(rec))
    return set(keys)


def _tier(level: str) -> str:
    return "base" if level == "baseline" else level


def _shipped_gold(domain: str, level: str) -> set[tuple[str, str, str, str]]:
    sm_dir = task_dir(domain, _tier(level)) / "input" / "schemamatching"
    if level == "baseline":
        payload = json.loads((sm_dir / "sm_mapping_gold.json").read_text(encoding="utf-8"))
        return {
            (m["source_dataset"], m["source_column"], m["target_dataset"], m["target_column"])
            for m in payload["mappings"]
            if m.get("label", True)
        }
    with open(sm_dir / "sm_mapping.csv", newline="", encoding="utf-8") as f:
        return {tuple(r[c] for c in GOLD_COLS) for r in csv.DictReader(f)}


def _gold_set(frame: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    return set(frame[GOLD_COLS].astype(str).itertuples(index=False, name=None))


def _positives(em: pd.DataFrame) -> pd.DataFrame:
    return em[em["label"].astype(str).str.lower().isin(("true", "1", "yes"))]


def _roster(tmp_path: Path, names: tuple[str, ...]) -> Path:
    """A copy of the real SM roster restricted to ``names`` (no LLM members)."""
    raw = yaml.safe_load(SM_ROSTER.read_text(encoding="utf-8"))
    raw["members"] = [m for m in raw["members"] if m["name"] in names]
    path = tmp_path / "sm_committee.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Every task: the view is the shipped task
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_view_is_the_shipped_task(domain: str, level: str) -> None:
    root = task_dir(domain, _tier(level))
    view = load_sm_view(domain, level, root)
    config = load_domain_config(domain)
    dropped = {"cluster_id"} if domain == "products" else set()

    assert view.input_tag == SM_INPUT_TAG
    assert list(view.sources) == config.source_names
    for name, df in view.sources.items():
        path = view.source_files[name]
        if level != "baseline":
            assert path == root / "input" / "data" / f"{name}.csv"
        assert path.is_relative_to(root / "input" / "data"), path
        # Exactly the shipped header: no renamed id column, no NA-padded
        # target column, nothing minted; products minus cluster_id.
        assert set(map(str, df.columns)) == _shipped_columns(path) - dropped, (name, path)
        assert df.attrs["dataset_name"] == name
        # No provenance id column: every shipped column is visible to matchers.
        assert set(get_schema_columns(df)) == set(map(str, df.columns))
        for col in df.columns:
            if df[col].dtype == object:
                assert not df[col].map(lambda v: isinstance(v, (list, dict))).any(), (name, col)

    # The shipped gold, as is (raw / K8 names), and every gold column exists.
    assert _gold_set(view.gold) == _shipped_gold(domain, level)
    for src, col in zip(view.gold["source_dataset"], view.gold["source_column"]):
        assert col in view.sources[src].columns, (src, col)
    assert view.target_schema == json.loads(
        (root / "input" / "schemamatching" / "target_schema.json").read_text(encoding="utf-8")
    )

    # EM gold for the duplicate member: every declared pair, ids in the id
    # space of the view's own record-id column (a prefixed-vs-bare mismatch
    # would resolve 0%; the variant knobs drop some records the EM gold still
    # names, hence a majority rather than all).
    assert view.source_pairs == list(config.source_pairs)
    for src1, src2 in config.source_pairs:
        pos = _positives(view.em_gold[(src1, src2)])
        assert not pos.empty, (src1, src2)
        for src, col in ((src1, "id1"), (src2, "id2")):
            id_col = view.record_id_columns[src]
            assert id_col is not None, src
            known = set(view.sources[src][id_col].astype(str))
            ids = set(pos[col].astype(str))
            assert len(ids & known) / len(ids) > 0.5, (src, id_col, len(ids - known))

    assert len(view.fusion_frames) == 2


@pytest.mark.parametrize("domain", ID_RENAMING_DOMAINS)
def test_base_view_keeps_raw_id_columns(domain: str) -> None:
    """games / companies / music: the bundle renames each id column to
    ``id``; the SM view keeps ``wiki_ref`` / ``entity_uri`` / ... and the gold
    keeps them too, so no matcher gets a free ``id -> id`` correspondence."""
    view = load_sm_view(domain, "baseline", task_dir(domain))
    gold = _gold_set(view.gold)
    for spec in load_domain_config(domain).sources:
        assert spec.id_column and spec.id_column != "id"
        cols = set(view.sources[spec.name].columns)
        assert spec.id_column in cols and "id" not in cols, spec.name
        assert view.record_id_columns[spec.name] == spec.id_column
        assert (spec.name, spec.id_column, "target_schema", "id") in gold
        assert not any(s == spec.name and c == "id" for s, c, _, _ in gold)


def test_papers_base_view_raw_names_no_padding_id_visible() -> None:
    view = load_sm_view("papers", "baseline", task_dir("papers"))
    canonical_only = {"title", "authors", "publication_year", "journal", "keywords"}
    for name, df in view.sources.items():
        cols = set(df.columns)
        # Raw names (dblp publication_title, crossref title_text, ...), none of
        # the loader's canonical names, and no all-NA padded column.
        assert set(_PAPERS_SOURCE_COLUMN_MAP[name]) <= cols, name
        assert not (cols & canonical_only), (name, cols & canonical_only)
        assert not [c for c in df.columns if df[c].isna().all()], name
        # The shipped ``id`` is a schema column now (the bundle hid it).
        assert "id" in get_schema_columns(df)
        assert df["id"].astype(str).str.startswith(f"{name}-").all()
    gold = _gold_set(view.gold)
    assert ("dblp", "publication_title", "target_schema", "title") in gold
    assert ("dblp", "id", "target_schema", "id") in gold


def test_products_base_view_reads_the_shipped_task() -> None:
    """Products base: the public task (not the generator copy), bare ids, no
    cluster_id, gold target ``target_schema``, and no target ``id`` values
    (the shipped fusion gold has only ``source_ids``)."""
    root = task_dir("products")
    view = load_sm_view("products", "baseline", root)
    for n in (1, 2, 3, 4):
        name = f"products_{n}"
        assert view.source_files[name] == root / "input" / "data" / f"dataset_{n}.json"
        df = view.sources[name]
        assert "cluster_id" not in df.columns
        assert not df["id"].astype(str).str.startswith("products_").any()
    assert set(view.gold["target_dataset"]) == {"target_schema"}
    assert len(view.gold) == 100
    target = _target_df_from_schema(
        view.target_schema, view.sources, target_name="target_schema",
        fusion_frames=view.fusion_frames,
    )
    assert "id" in target.columns and target["id"].isna().all()


@pytest.mark.parametrize("level", LEVELS)
def test_products_views_never_carry_target_id_values(level: str) -> None:
    view = load_sm_view("products", level, task_dir("products", _tier(level)))
    for frame in view.fusion_frames:
        assert "id" not in frame.columns or frame["id"].isna().all()
    for df in view.sources.values():
        assert "cluster_id" not in df.columns


def test_music_view_keeps_discogs_zero_durations_as_shipped() -> None:
    """The bundle coalesces discogs ``duration == 0`` to NA (a value change
    the later stages rely on); the SM view reads the file as shipped."""
    view = load_sm_view("music", "baseline", task_dir("music"))
    raw = view.sources["discogs"]["duration"].astype(str).str.strip()
    assert raw.isin({"0", "0.0"}).sum() > 0


# ---------------------------------------------------------------------------
# Where the view comes from: sm_task_root on the bundles
# ---------------------------------------------------------------------------


def test_load_variant_sets_sm_task_root() -> None:
    games = load_variant("games", "baseline")
    assert games.sm_task_root == task_dir("games") == games.variant_root

    # products: the bundle keeps the generator copy for the other stages,
    # schema matching reads the released task.
    products = load_variant("products", "baseline")
    assert products.sm_task_root == task_dir("products")
    assert products.variant_root != products.sm_task_root
    assert "usecases_synthetic" in products.variant_root.parts

    easy = load_variant("games", "easy")
    assert easy.sm_task_root == easy.variant_root == task_dir("games", "easy")

    view = load_sm_view_for_bundle(products)
    assert view is not None and view.task_root == task_dir("products")


def test_hand_built_bundle_takes_legacy_path_with_warning(caplog, tmp_path: Path) -> None:
    src = pd.DataFrame({"id": ["a1", "a2"], "name": ["x", "y"]})
    src.attrs["dataset_name"] = "a"
    gold = pd.DataFrame(
        [("a", "name", "t", "name", 1.0)], columns=[*GOLD_COLS, "score"]
    )
    bundle = VariantBundle(
        domain="companies", level="baseline", sources={"a": src},
        target_schema={"properties": {"name": {}, "id": {}}}, sm_mapping=gold,
        em_gold={}, em_splits={}, fusion_gold=pd.DataFrame(), fusion_validation=None,
        pooled_positives=None, variant_root=tmp_path,
    )
    assert bundle.sm_task_root is None
    assert load_sm_view_for_bundle(bundle) is None
    assert legacy_sm_view(bundle).input_tag == SM_INPUT_TAG_LEGACY
    runner = SMCommitteeRunner(_roster(tmp_path, ("label_jw",)))
    with caplog.at_level(logging.WARNING):
        result = runner.run(bundle)
    assert "legacy path" in caplog.text
    assert result.per_member["label_jw"].notes["sm_input"] == SM_INPUT_TAG_LEGACY


# ---------------------------------------------------------------------------
# The runner reads the view; the bundle (other stages) is untouched
# ---------------------------------------------------------------------------


def _fingerprint(sources: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        k: (list(map(str, v.columns)), pd.util.hash_pandas_object(v.astype(str), index=True).sum())
        for k, v in sources.items()
    }


def test_runner_scores_raw_names_and_leaves_bundle_untouched(tmp_path: Path) -> None:
    bundle = load_variant("games", "baseline")
    # What the other stages read: renamed ids, reconciled gold.
    assert "id" in bundle.sources["dbpedia"].columns
    assert "wiki_ref" not in bundle.sources["dbpedia"].columns
    assert ("dbpedia", "id", "target_schema", "id") in _gold_set(bundle.sm_mapping)
    before = _fingerprint(bundle.sources)
    gold_before = bundle.sm_mapping.copy()

    runner = SMCommitteeRunner(_roster(tmp_path, ("label_jw", "instance_tf_cosine")))
    result = runner.run(bundle)

    raw_cols = {
        name: set(df.columns)
        for name, df in load_sm_view("games", "baseline", task_dir("games")).sources.items()
    }
    for member in result.per_member.values():
        assert member.notes["sm_input"] == SM_INPUT_TAG
        pred = member.predictions
        for src, col in zip(pred["source_dataset"], pred["source_column"]):
            assert col in raw_cols[src], (member.name, src, col)
    # Per-attribute keys are the raw gold's (e.g. dbpedia.wiki_ref, not dbpedia.id).
    assert "dbpedia.wiki_ref" in result.per_attribute
    assert "dbpedia.id" not in result.per_attribute
    # label_jw sees no identical id names any more: no id row is a label hit.
    label = result.per_member["label_jw"].predictions
    assert not ((label["target_column"] == "id") & label["source_column"].isin(
        ["wiki_ref", "mc_id", "rec_id"])).any()

    # The bundle the later stages read is unchanged.
    assert _fingerprint(bundle.sources) == before
    pd.testing.assert_frame_equal(bundle.sm_mapping, gold_before)


def test_bundle_frames_for_other_stages_keep_their_renames() -> None:
    """The fix is SM-only: papers / music / products bundles still carry the
    loader's renames and value fixes the norm / EM / fusion stages expect."""
    papers = load_variant("papers", "baseline")
    dblp = papers.sources["dblp"]
    assert {"title", "authors", "keywords"} <= set(dblp.columns)
    assert "publication_title" not in dblp.columns
    assert dblp["keywords"].isna().all()  # NA padding for the later stages
    assert ("dblp", "title", "target_schema", "title") in _gold_set(papers.sm_mapping)

    music = load_variant("music", "baseline")
    duration = music.sources["discogs"]["duration"].astype(str).str.strip()
    assert not duration.isin({"0", "0.0"}).any()  # coalesced to NA
    assert "id" in music.sources["discogs"].columns

    products = load_variant("products", "baseline")
    assert products.sources["products_1"]["id"].astype(str).str.startswith("products_1_").all()


# ---------------------------------------------------------------------------
# Duplicate-based member: record-id hint
# ---------------------------------------------------------------------------


def _duplicate_matcher():
    from PyDI.schemamatching.duplicate_based import DuplicateBasedSchemaMatcher

    return DuplicateBasedSchemaMatcher(
        vote_aggregation="majority", value_comparison="exact", min_votes=1
    )


def test_record_id_hint_is_required_for_non_id_named_columns() -> None:
    left = pd.DataFrame({"wiki_ref": ["l1", "l2"], "title": ["Alpha", "Beta"]})
    right = pd.DataFrame({"mc_key": ["r1", "r2"], "name": ["Alpha", "Beta"]})
    left.attrs["dataset_name"], right.attrs["dataset_name"] = "left", "right"
    corr = pd.DataFrame({"id1": ["l1", "l2"], "id2": ["r1", "r2"]})
    matcher = _duplicate_matcher()

    assert matcher.match(left, right, correspondences=corr, threshold=0.1).empty
    hinted = matcher.match(
        _with_record_id_hint(left, "wiki_ref"), _with_record_id_hint(right, "mc_key"),
        correspondences=corr, threshold=0.1,
    )
    assert ("title", "name") in set(zip(hinted["source_column"], hinted["target_column"]))
    hinted_left = _with_record_id_hint(left, "wiki_ref")
    assert list(hinted_left.columns) == list(left.columns)
    assert hinted_left.attrs["dataset_name"] == "left"
    assert _with_record_id_hint(left, None) is left


def test_duplicate_member_on_raw_view_equals_renamed_run(tmp_path: Path) -> None:
    """games base: the duplicate member on the raw view predicts the same
    column pairs as on the renamed bundle frames, in raw names."""
    bundle = load_variant("games", "baseline")
    runner = SMCommitteeRunner(_roster(tmp_path, ("duplicate_majority",)))
    spec, matcher = runner._specs[0], runner._matchers[0]

    raw = runner.run(bundle).per_member["duplicate_majority"]
    assert raw.notes["sm_input"] == SM_INPUT_TAG
    assert not raw.predictions.empty

    renamed = runner._run_duplicate_per_pair(
        matcher, spec, legacy_sm_view(bundle), bundle.sm_mapping
    )
    to_raw = {s.name: s.id_column for s in load_domain_config("games").sources}
    renamed_in_raw = {
        (src, to_raw[src] if col == "id" else col, tgt)
        for src, col, tgt in zip(
            renamed["source_dataset"], renamed["source_column"], renamed["target_column"]
        )
    }
    got = set(
        zip(raw.predictions["source_dataset"], raw.predictions["source_column"],
            raw.predictions["target_column"])
    )
    assert got == renamed_in_raw
