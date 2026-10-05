"""EM validation split resolution for P2's stage 3 (blocker selection).

P2 selects blockers (and matchers) on the ``val`` split that
``stage_runners._swap_em_gold`` swaps into ``bundle.em_gold``; with no val
split for any pair it falls back to the test split and records
``val_split_available=false``. The stored games base run did
that because the games val split did not exist yet. The
(metacritic, dbpedia) pair ships its val file in the declared orientation
(``metacritic_2_dbpedia_val.csv``) but its test file reversed
(``dbpedia_2_metacritic_test.csv``); both orientations must load, oriented to
the declared pair.
"""

from __future__ import annotations

from pathlib import Path

from usecases_synthetic.lib.variant_loader import _load_em_splits


def _write(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.write_text("\n".join(",".join(r) for r in rows) + "\n")


def test_em_splits_resolve_either_file_orientation(tmp_path: Path) -> None:
    _write(tmp_path / "metacritic_2_dbpedia_val.csv", [("metacritic_1", "dbpedia_1", "TRUE")])
    _write(tmp_path / "dbpedia_2_metacritic_test.csv", [("dbpedia_2", "metacritic_2", "TRUE")])
    _write(tmp_path / "sales_2_dbpedia_val.csv", [("sales_3", "dbpedia_3", "FALSE")])

    splits = _load_em_splits(tmp_path, [("metacritic", "dbpedia"), ("dbpedia", "sales")])

    md = splits[("metacritic", "dbpedia")]
    assert set(md) == {"val", "test"}
    assert (md["val"].loc[0, "id1"], md["val"].loc[0, "id2"]) == ("metacritic_1", "dbpedia_1")
    # reverse-direction file: swapped into the declared (metacritic, dbpedia) orientation
    assert (md["test"].loc[0, "id1"], md["test"].loc[0, "id2"]) == ("metacritic_2", "dbpedia_2")
    ds = splits[("dbpedia", "sales")]
    assert set(ds) == {"val"}
    assert (ds["val"].loc[0, "id1"], ds["val"].loc[0, "id2"]) == ("dbpedia_3", "sales_3")


def test_games_base_val_split_swapped_in_for_every_pair() -> None:
    from pipelines.lib.bundle import PipelineState, load_pipeline_bundle
    from pipelines.lib.stage_runners import _restore_em_gold, _swap_em_gold

    bundle = load_pipeline_bundle("games")
    state = PipelineState(bundle=bundle)
    pairs = set(bundle.em_gold)
    assert pairs == {("dbpedia", "sales"), ("metacritic", "dbpedia")}
    for pair in pairs:
        assert {"train", "val", "test"} <= set(bundle.em_splits[pair])

    prev = _swap_em_gold(state, split="val")
    try:
        assert prev is not None  # stage 3 then records val_split_available=True
        assert set(state.bundle.em_gold) == pairs
        for pair, gold in state.bundle.em_gold.items():
            assert gold is bundle.em_splits[pair]["val"]
            assert gold["id1"].astype(str).str.startswith(f"{pair[0]}_").all()
            assert gold["id2"].astype(str).str.startswith(f"{pair[1]}_").all()
    finally:
        _restore_em_gold(state, prev)
