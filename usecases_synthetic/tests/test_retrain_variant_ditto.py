"""Smoke tests for the variant Ditto retrain script.

The heavy trainer (``_invoke_ditto_train``, which shells out to
``ditto/train.py``) is monkeypatched to a stub. The wide committee-scope
record builder is the REAL one (wrapped for capture), run on a tiny
fixture shaped like a shipped companies/medium variant: the loader's
``id`` join column on every source, K8-renamed dbpedia columns
(``org_nm``/``ctry``) and the K8 provenance on ``bundle.knob_08_renames``.
What is exercised: pooling + dedup, the ``train.json.gz`` /
``val.json.gz`` build on the *wide* committee field scope (not
knob-02 canonical_schema), the K8 translation of the committee
column_mapping restoring canonical names (non-empty ``<field>_left`` /
``<field>_right``), the degenerate-data guard refusing field-stripped
records, the trainer boundary (signature + the flags it passes to
``ditto/train.py``), and placement of the variant checkpoint symlink.
"""

from __future__ import annotations

import gzip
import inspect
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import usecases_synthetic.scripts.ditto.retrain_variant as rv
from usecases_synthetic.lib.domain_config import load_knob_config
from usecases_synthetic.lib.variant_loader import VariantBundle
from usecases_synthetic.scripts.retrain_variant_cascade import (
    _is_corner_filled_data_gap,
)

# K8 provenance of the fixture, keyed like ``_load_knob_08_renames``
# (``{source: {pre_k8_col: post_k8_col}}``). Mirrors the shipped
# companies/medium ``knob_08_naming.csv``: dbpedia abbreviated, forbes renamed
# up to the descriptive (= canonical) names.
_K8_RENAMES: dict[str, dict[str, str]] = {
    "dbpedia": {"org_name": "org_nm", "nation": "ctry"},
    "forbes": {"company": "name", "region": "country"},
}


def _k8_sources() -> dict[str, pd.DataFrame]:
    return {
        "dbpedia": pd.DataFrame(
            {"id": ["d1", "d2"], "org_nm": ["Acme Corp", "Beta AG"], "ctry": ["US", "DE"]}
        ),
        "forbes": pd.DataFrame(
            {"id": ["f1", "f2"], "name": ["Acme", "Gamma"], "country": ["USA", "France"]}
        ),
    }


def _tiny_bundle(
    level: str,
    variant_root: Path,
    *,
    sources: dict[str, pd.DataFrame] | None = None,
    knob_08_renames: dict[str, dict[str, str]] | None = None,
) -> VariantBundle:
    pair = ("dbpedia", "forbes")
    gold = pd.DataFrame(
        {"id1": ["d1", "d2"], "id2": ["f1", "f2"], "label": ["true", "false"]}
    )
    return VariantBundle(
        domain="companies",
        level=level,
        sources=_k8_sources() if sources is None else sources,
        target_schema={},
        sm_mapping=None,
        em_gold={pair: gold},
        em_splits={},
        em_gold_regenerated={
            pair: {
                "train": {"corner_filled": gold},
                "val": {"corner_filled": gold.head(1)},
            }
        },
        fusion_gold=pd.DataFrame(),
        fusion_validation=None,
        pooled_positives=None,
        variant_root=variant_root,
        knob_08_renames=_K8_RENAMES if knob_08_renames is None else knob_08_renames,
    )


def _fake_best(run_parent: Path) -> Path:
    best = run_parent / "run_test" / "checkpoints" / "best"
    best.mkdir(parents=True)
    (best / "config.json").write_text("{}", encoding="utf-8")
    return best


def _read_json_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestRetrainVariantDittoSmoke:
    def test_builds_data_invokes_trainer_and_places_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = _tiny_bundle("medium", tmp_path / "variant")
        monkeypatch.setattr(
            rv,
            "load_variant",
            lambda domain, level, *, root_override=None: bundle,
        )

        calls: dict[str, Any] = {}
        real_builder = rv.build_ditto_pair_records_committee_scope

        def _capturing_records(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            # Record the builder kwargs so the test can assert the wide
            # field scope + the K8-resolved committee column_mapping flow
            # through, then delegate to the REAL builder.
            calls.setdefault("builder_fields", kwargs.get("fields"))
            calls.setdefault("builder_column_mapping", kwargs.get("column_mapping"))
            calls.setdefault("builder_sources", kwargs.get("sources"))
            return real_builder(*args, **kwargs)

        monkeypatch.setattr(
            rv, "build_ditto_pair_records_committee_scope", _capturing_records
        )
        ckpt_parent = tmp_path / "ckpt"
        monkeypatch.setattr(
            rv,
            "_ditto_variant_dir",
            lambda domain, level: ckpt_parent / domain / f"variant_{level}",
        )

        def fake_train(
            train_json: Path,
            val_json: Path,
            run_parent: Path,
            *,
            fields: str,
            batch_size: int,
            max_len: int,
            max_field_len: int,
            config_path: Path,
        ) -> Path:
            calls.update(
                train_json=train_json,
                val_json=val_json,
                fields=fields,
                batch_size=batch_size,
                max_len=max_len,
                max_field_len=max_field_len,
            )
            return _fake_best(run_parent)

        # The stub must track the real trainer boundary (a stale stub is how
        # these smoke tests broke before): same parameters, same order.
        assert list(inspect.signature(fake_train).parameters) == list(
            inspect.signature(rv._invoke_ditto_train).parameters
        )
        monkeypatch.setattr(rv, "_invoke_ditto_train", fake_train)

        work = tmp_path / "work"
        out = rv.retrain_variant_ditto("companies", "medium", work_dir=work)

        # train.json.gz / val.json.gz built with the pooled records.
        train_recs = _read_json_gz(work / "train.json.gz")
        val_recs = _read_json_gz(work / "val.json.gz")
        assert len(train_recs) == 2  # d1/d2 deduped to 2 rows
        assert len(val_recs) == 1
        assert train_recs[0]["pair_id"] == "d1__f1"
        assert [r["label"] for r in train_recs] == [1, 0]

        # Trainer invoked with the WIDE committee field scope (not
        # knob-02 canonical_schema), and knob-02 still supplies the PLM
        # hyperparameters.
        from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
            committee_ditto_fields,
        )

        knob02 = load_knob_config(2, "companies")
        expected_fields = committee_ditto_fields("companies")
        assert calls["fields"] == ",".join(expected_fields)
        assert calls["builder_fields"] == expected_fields
        assert calls["batch_size"] == int(knob02.get("plm_batch_size", 16))
        assert calls["max_len"] == int(knob02.get("plm_max_len", 256))
        assert calls["max_field_len"] == int(knob02.get("plm_max_field_len", 350))

        # The committee column_mapping, translated through the bundle's K8
        # renames, flows into the builder: the abbreviated dbpedia columns are
        # keyed on their post-K8 names; the raw variant sources pass through.
        resolved = calls["builder_column_mapping"]
        assert resolved["dbpedia"]["org_nm"] == "name"
        assert resolved["dbpedia"]["ctry"] == "country"
        assert "org_name" not in resolved["dbpedia"]
        assert calls["builder_sources"] is bundle.sources

        # Every record carries the full wide field scope on both sides, and
        # the K8-renamed columns come back under their canonical names with
        # their values (the degenerate-data guard's precondition).
        for rec in train_recs + val_recs:
            for f in expected_fields:
                assert f"{f}_left" in rec and f"{f}_right" in rec
        assert train_recs[0]["name_left"] == "Acme Corp"
        assert train_recs[0]["name_right"] == "Acme"
        assert train_recs[0]["country_left"] == "US"
        assert train_recs[0]["country_right"] == "USA"
        assert train_recs[1]["name_left"] == "Beta AG"
        # Fields no fixture source carries serialize as empty strings.
        assert train_recs[0]["city_left"] == "" and train_recs[0]["city_right"] == ""

        # Variant checkpoint placed as a symlink to the produced best dir.
        assert out == ckpt_parent / "companies" / "variant_medium" / "best"
        assert out.is_symlink()
        assert (out / "config.json").exists()

    def test_out_dir_override_routes_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``out_dir`` places the best symlink in the pipeline-isolated tree,
        bypassing ``_ditto_variant_dir`` (committee cache)."""
        bundle = _tiny_bundle("medium", tmp_path / "variant")
        monkeypatch.setattr(
            rv, "load_variant", lambda domain, level, *, root_override=None: bundle
        )

        def _boom(domain: str, level: str) -> Path:  # pragma: no cover
            raise AssertionError("committee cache path must not be used")

        monkeypatch.setattr(rv, "_ditto_variant_dir", _boom)

        def fake_train(
            train_json: Path, val_json: Path, run_parent: Path, **_: Any
        ) -> Path:
            return _fake_best(run_parent)

        monkeypatch.setattr(rv, "_invoke_ditto_train", fake_train)

        isolated = tmp_path / "pipelines" / "companies" / "ckpt" / "variant_medium"
        out = rv.retrain_variant_ditto(
            "companies", "medium", work_dir=isolated, out_dir=isolated
        )
        assert out == isolated / "best"
        assert out.is_symlink()
        # run_* work dirs are isolated under the same out_dir.
        assert (isolated / "runs").exists()
        assert (isolated / "train.json.gz").exists()

    def test_degenerate_training_data_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Variant columns that do not map onto the committee fields (here: K8
        renames on disk but no K8 provenance on the bundle) leave every
        serialized field empty; the guard must refuse to train, before any
        json.gz is written, with an error the cascade does NOT swallow as a
        corner_filled data gap."""
        sources = {
            "dbpedia": pd.DataFrame(
                {"id": ["d1", "d2"], "org_nm": ["Acme Corp", "Beta AG"], "ctry": ["US", "DE"]}
            ),
            "forbes": pd.DataFrame(
                {"id": ["f1", "f2"], "co_nm": ["Acme", "Gamma"], "rgn": ["USA", "France"]}
            ),
        }
        bundle = _tiny_bundle(
            "medium", tmp_path / "variant", sources=sources, knob_08_renames={}
        )
        monkeypatch.setattr(
            rv, "load_variant", lambda domain, level, *, root_override=None: bundle
        )

        def _no_train(*_: Any, **__: Any) -> Path:  # pragma: no cover
            raise AssertionError("trainer must not run on degenerate data")

        monkeypatch.setattr(rv, "_invoke_ditto_train", _no_train)

        work = tmp_path / "work"
        with pytest.raises(RuntimeError, match="Degenerate variant training data") as exc:
            rv.retrain_variant_ditto("companies", "medium", work_dir=work, out_dir=work)
        assert "2/2 train records" in str(exc.value)
        assert not _is_corner_filled_data_gap(exc.value)
        assert not (work / "train.json.gz").exists()

    def test_trainer_boundary_passes_flags_the_trainer_declares(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run the REAL ``_invoke_ditto_train`` with ``subprocess.run``
        stubbed: every ``--flag`` it passes must be declared by
        ``scripts/ditto/train.py``'s argparser, and it returns the newest
        ``run_*/checkpoints/best``."""
        captured: dict[str, Any] = {}

        def _fake_run(cmd: list[str], check: bool = False, **_: Any) -> Any:
            captured["cmd"] = cmd
            captured["check"] = check
            _fake_best(Path(cmd[cmd.index("--output-dir") + 1]))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(rv.subprocess, "run", _fake_run)
        run_parent = tmp_path / "runs"
        best = rv._invoke_ditto_train(
            tmp_path / "train.json.gz",
            tmp_path / "val.json.gz",
            run_parent,
            fields="name,country",
            batch_size=16,
            max_len=256,
            max_field_len=350,
            config_path=rv._DEFAULT_TRAIN_YAML,
        )
        assert best == run_parent / "run_test" / "checkpoints" / "best"
        assert captured["check"] is True
        cmd = captured["cmd"]
        assert cmd[1] == str(rv._DITTO_TRAIN_PY)
        assert cmd[cmd.index("--fields") + 1] == "name,country"
        passed = {tok for tok in cmd if tok.startswith("--")}
        declared = set(
            re.findall(
                r"add_argument\(\s*\"(--[A-Za-z0-9-]+)\"",
                rv._DITTO_TRAIN_PY.read_text(encoding="utf-8"),
            )
        )
        assert passed, cmd
        assert passed <= declared, sorted(passed - declared)

    def test_baseline_level_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="baseline"):
            rv.retrain_variant_ditto("companies", "baseline")


class TestDedupeRecords:
    def test_unordered_pair_dedup(self) -> None:
        records = [
            {"id_left": "a", "id_right": "b"},
            {"id_left": "b", "id_right": "a"},  # same unordered pair
            {"id_left": "a", "id_right": "c"},
        ]
        out = rv._dedupe_records(records)
        assert len(out) == 2
