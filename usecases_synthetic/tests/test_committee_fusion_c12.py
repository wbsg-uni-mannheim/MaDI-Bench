"""Tests for the C12 fusion committee runner.

Covers the coherent-member roster shape (C12). The legacy
per-(attribute, strategy) tests live in
``tests/test_committee_fusion.py`` and stay green via the
:class:`FusionCommitteeRunner.__new__` dispatcher.

Scope of this module:

* YAML parser — good cases + every documented validation error.
* Selection-cache I/O round-trip.
* Per-member ``_selection_attrs_for_member`` (which attributes need
  val-best PyDI for each of the 9 members).
* Dispatcher routing — ``FusionCommitteeRunner`` returns a C12
  instance for ``members:``-shape YAMLs and the legacy instance for
  the per-(attribute, strategy) YAMLs.
* End-to-end execution for the two no-val-selection-needed members
  (``voting_only`` + ``prefer_higher_trust_only``) — they should run
  without a fusion validation set and produce a per-member
  macro_accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from usecases_synthetic.lib.committee_fusion import FusionCommitteeRunner
from usecases_synthetic.lib.committee_fusion_c12 import (
    SUPPORTED_MEMBERS,
    C12FusionCommitteeRunner,
    _NATIVE_TYPES_BY_MEMBER,
    _load_selection_cache,
    _parse_roster,
    _save_selection_cache,
    _selection_attrs_for_member,
    _selection_cache_path,
)
from usecases_synthetic.lib.variant_loader import VariantBundle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_roster_dict() -> dict[str, Any]:
    """Return a minimal but valid C12 roster dict for parser tests."""
    return {
        "seed": 42,
        "fused_id_column": "id",
        "gold_id_column": "id",
        "trust_scores": {"a": 2.0, "b": 1.0},
        "column_mapping": {"a": {}, "b": {}},
        "evaluation_functions": {"name": "exact_match", "revenue": "exact_match"},
        "attribute_types": {"name": "string", "revenue": "numeric"},
        "pydi_candidates": {
            "string": [
                {
                    "name": "voting",
                    "function": "voting",
                    "module": "PyDI.fusion.conflict_resolution.general",
                }
            ],
            "numeric": [
                {
                    "name": "median",
                    "function": "median",
                    "module": "PyDI.fusion.conflict_resolution.numeric",
                }
            ],
        },
        "members": [
            {"name": "voting_only", "params": {}},
            {"name": "prefer_higher_trust_only", "params": {}},
        ],
    }


def _write_yaml(tmp_path: Path, raw: dict[str, Any]) -> Path:
    """Write *raw* to a fresh roster YAML under tmp_path."""
    p = tmp_path / "fusion_committee_test.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return p


def _make_two_source_bundle(
    *,
    domain: str = "test_domain",
    level: str = "baseline",
) -> VariantBundle:
    """Build a minimal two-source bundle with 3 entities for fusion tests.

    Sources ``a`` and ``b`` carry the same 3 ids (mapped via
    correspondences). The gold uses source ``a``'s name + revenue
    values. With trust ``a=2 > b=1`` (in the roster), both
    ``voting_only`` (tied 1-vs-1, falls back to first non-null) and
    ``prefer_higher_trust_only`` (picks ``a``) should fuse to ``a``'s
    values, yielding accuracy 1.0 on ``a``-flavoured gold.
    """
    n = 3
    src_a = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": [f"A Co {i}" for i in range(n)],
            "revenue": [float(100 * i) for i in range(n)],
        }
    )
    src_a.attrs["dataset_name"] = "a"
    src_b = pd.DataFrame(
        {
            "id": [f"b_{i}" for i in range(n)],
            "name": [f"B Co {i}" for i in range(n)],
            "revenue": [float(100 * i + 5) for i in range(n)],
        }
    )
    src_b.attrs["dataset_name"] = "b"

    correspondences = pd.DataFrame(
        [{"id1": f"a_{i}", "id2": f"b_{i}", "score": 1.0} for i in range(n)]
    )
    em_gold = {
        ("a", "b"): correspondences.assign(label=1),
    }
    fusion_gold = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(n)],
            "name": [f"A Co {i}" for i in range(n)],
            "revenue": [float(100 * i) for i in range(n)],
        }
    )

    return VariantBundle(
        domain=domain,
        level=level,
        sources={"a": src_a, "b": src_b},
        target_schema={"properties": {"id": {}, "name": {}, "revenue": {}}},
        sm_mapping=None,
        em_gold=em_gold,
        em_splits={},
        fusion_gold=fusion_gold,
        fusion_validation=None,
        pooled_positives=None,
        variant_root=Path("/tmp/fusion_c12_test"),
    )


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseRoster:
    """Cover the C12 YAML parser's good case + every validation error."""

    def test_parses_minimal_roster(self) -> None:
        raw = _minimal_roster_dict()
        roster = _parse_roster(raw)
        assert [m.name for m in roster.members] == [
            "voting_only",
            "prefer_higher_trust_only",
        ]
        assert roster.attribute_types == {"name": "string", "revenue": "numeric"}
        assert "string" in roster.pydi_candidates_by_type
        assert "numeric" in roster.pydi_candidates_by_type
        assert roster.trust_scores == {"a": 2.0, "b": 1.0}

    def test_parses_all_nine_members(self) -> None:
        raw = _minimal_roster_dict()
        raw["members"] = [{"name": n, "params": {}} for n in sorted(SUPPORTED_MEMBERS)]
        roster = _parse_roster(raw)
        assert {m.name for m in roster.members} == SUPPORTED_MEMBERS

    def test_rejects_unknown_member(self) -> None:
        raw = _minimal_roster_dict()
        raw["members"] = [{"name": "made_up_member"}]
        with pytest.raises(ValueError, match="Unknown member"):
            _parse_roster(raw)

    def test_rejects_missing_attribute_types(self) -> None:
        raw = _minimal_roster_dict()
        del raw["attribute_types"]
        with pytest.raises(ValueError, match="attribute_types"):
            _parse_roster(raw)

    def test_rejects_invalid_attribute_type(self) -> None:
        raw = _minimal_roster_dict()
        raw["attribute_types"]["name"] = "bogus_type"
        with pytest.raises(ValueError, match="bogus_type"):
            _parse_roster(raw)

    def test_rejects_pydi_candidate_missing_keys(self) -> None:
        raw = _minimal_roster_dict()
        raw["pydi_candidates"]["string"][0] = {"name": "voting"}  # no function/module
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_roster(raw)

    def test_rejects_empty_pydi_candidate_list(self) -> None:
        raw = _minimal_roster_dict()
        raw["pydi_candidates"]["string"] = []
        with pytest.raises(ValueError, match="empty"):
            _parse_roster(raw)

    def test_rejects_missing_pydi_candidates_for_used_type(self) -> None:
        raw = _minimal_roster_dict()
        # Remove the numeric candidate but keep ``revenue`` numeric.
        del raw["pydi_candidates"]["numeric"]
        with pytest.raises(ValueError, match="pydi_candidates is missing"):
            _parse_roster(raw)

    def test_rejects_empty_members(self) -> None:
        raw = _minimal_roster_dict()
        raw["members"] = []
        with pytest.raises(ValueError, match="members"):
            _parse_roster(raw)

    def test_parses_music_yaml(self) -> None:
        """The migrated music YAML is the canonical C12 reference."""
        repo_root = Path(__file__).resolve().parents[1]
        yaml_path = repo_root / "config" / "committees" / "fusion_committee_music.yaml"
        with yaml_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        roster = _parse_roster(raw)
        assert {m.name for m in roster.members} == SUPPORTED_MEMBERS
        # Music has 8 attributes spanning every supported type.
        assert set(roster.attribute_types.values()) == {
            "string",
            "categorical",
            "date",
            "numeric",
            "list",
        }


# ---------------------------------------------------------------------------
# Selection-cache I/O tests
# ---------------------------------------------------------------------------


class TestSelectionCache:
    """Round-trip the per-domain val-selection cache."""

    def test_path_under_baselines(self) -> None:
        path = _selection_cache_path("music")
        assert path.name == "fusion_committee_selection.json"
        assert "baselines" in path.parts
        assert "music" in path.parts

    def test_load_returns_empty_when_absent(self, tmp_path: Path) -> None:
        # Point the loader at a non-existent domain.
        cache = _load_selection_cache("__no_such_domain__")
        assert cache == {}

    def test_save_then_load_roundtrip(self, tmp_path: Path, monkeypatch) -> None:
        """Save → load returns an equivalent dict.

        Patches the path-resolver to keep tests isolated from the real
        ``baselines/`` tree (no test should write into the canonical
        baselines directory).
        """
        domain = "fusion_c12_test_domain"
        target = tmp_path / "baselines" / domain / "fusion_committee_selection.json"

        from usecases_synthetic.lib import committee_fusion_c12 as mod

        def fake_path(d: str) -> Path:
            return tmp_path / "baselines" / d / "fusion_committee_selection.json"

        monkeypatch.setattr(mod, "_selection_cache_path", fake_path)

        payload = {
            "pydi_per_attribute_optimal": {
                "name": "voting",
                "revenue": "median",
            },
            "fusionquery_only": {
                "revenue": "median",
                "tracks": "union",
            },
        }
        _save_selection_cache(domain, payload)
        assert target.exists()
        loaded = _load_selection_cache(domain)
        assert loaded == payload


# ---------------------------------------------------------------------------
# Per-member selection-attribute calculation
# ---------------------------------------------------------------------------


class TestSelectionAttrsForMember:
    """Members with universal native sets need no val selection; TD
    members need it on non-native types; pydi_per_attribute_optimal needs
    it on every attribute."""

    @pytest.fixture
    def roster(self):  # type: ignore[no-untyped-def]
        raw = _minimal_roster_dict()
        return _parse_roster(raw)

    def test_voting_only_needs_nothing(self, roster) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        assert (
            _selection_attrs_for_member(_MemberSpec(name="voting_only"), roster) == []
        )

    def test_prefer_higher_trust_only_needs_nothing(self, roster) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        assert (
            _selection_attrs_for_member(
                _MemberSpec(name="prefer_higher_trust_only"), roster
            )
            == []
        )

    def test_llm_only_needs_nothing(self, roster) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        assert _selection_attrs_for_member(_MemberSpec(name="llm_only"), roster) == []

    def test_accusim_only_needs_nothing(self, roster) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        assert (
            _selection_attrs_for_member(_MemberSpec(name="accusim_only"), roster) == []
        )

    def test_pydi_per_attribute_optimal_needs_every_attribute(
        self, roster
    ) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        attrs = _selection_attrs_for_member(
            _MemberSpec(name="pydi_per_attribute_optimal"), roster
        )
        assert set(attrs) == set(roster.attribute_types.keys())

    def test_fusionquery_only_skips_native_string(
        self, roster
    ) -> None:  # type: ignore[no-untyped-def]
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        # roster has ``name`` (string, native) and ``revenue`` (numeric, fallback).
        attrs = _selection_attrs_for_member(
            _MemberSpec(name="fusionquery_only"), roster
        )
        assert attrs == ["revenue"]

    def test_ltm_only_skips_list_and_string_keeps_numeric(self) -> None:
        # LTM is native on string/categorical/date + list; numeric is fallback.
        raw = _minimal_roster_dict()
        raw["attribute_types"] = {
            "name": "string",
            "revenue": "numeric",
            "tags": "list",
        }
        raw["pydi_candidates"]["list"] = [
            {
                "name": "union",
                "function": "union",
                "module": "PyDI.fusion.conflict_resolution.list",
            }
        ]
        roster = _parse_roster(raw)
        from usecases_synthetic.lib.committee_fusion_c12 import _MemberSpec

        attrs = _selection_attrs_for_member(_MemberSpec(name="ltm_only"), roster)
        assert attrs == ["revenue"]


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------


class TestDispatcher:
    """``FusionCommitteeRunner(path)`` routes by YAML shape."""

    def test_routes_members_yaml_to_c12(self, tmp_path: Path) -> None:
        yaml_path = _write_yaml(tmp_path, _minimal_roster_dict())
        runner = FusionCommitteeRunner(yaml_path)
        assert isinstance(runner, C12FusionCommitteeRunner)
        assert runner.roster_names == ["voting_only", "prefer_higher_trust_only"]

    def test_routes_legacy_yaml_to_legacy(self, tmp_path: Path) -> None:
        """A legacy-shape YAML (per-attribute ``strategies:`` blocks)
        stays on the legacy path. All four shipped per-domain YAMLs are
        C12 post-2026-05-25, so this test synthesises a minimal legacy
        roster to keep the dispatcher's negative branch covered."""
        legacy_raw = {
            "seed": 42,
            "fused_id_column": "id",
            "gold_id_column": "id",
            "trust_scores": {"a": 2.0, "b": 1.0},
            "column_mapping": {"a": {}, "b": {}},
            "evaluation_functions": {"name": "exact_match"},
            "attributes": {
                "name": {
                    "attribute_class": "primary",
                    "strategies": [
                        {
                            "name": "voting",
                            "function": "voting",
                            "module": "PyDI.fusion.conflict_resolution.general",
                            "strategy_type": "cell_local",
                            "params": {},
                        }
                    ],
                }
            },
        }
        legacy_yaml = _write_yaml(tmp_path, legacy_raw)
        runner = FusionCommitteeRunner(legacy_yaml)
        # Legacy runner is the base class, not the C12 subclass.
        assert not isinstance(runner, C12FusionCommitteeRunner)
        # Legacy uses per-(attr, strat) names like "name_voting".
        assert any("_" in n for n in runner.roster_names)


# ---------------------------------------------------------------------------
# End-to-end execution (no val-selection needed)
# ---------------------------------------------------------------------------


class TestSimpleMemberExecution:
    """End-to-end runs for the two no-val-selection members.

    These members do not need ``fusion_validation``; they can score
    every level (including baseline) without a val sweep. Both should
    produce a fused DataFrame and a macro_accuracy > 0 on the
    trust-aligned fixture.
    """

    def test_voting_and_prefer_higher_trust_run_end_to_end(
        self, tmp_path: Path
    ) -> None:
        # Build a roster with only the two no-val-selection members so
        # the test doesn't accidentally exercise val selection.
        raw = _minimal_roster_dict()
        raw["members"] = [
            {"name": "voting_only"},
            {"name": "prefer_higher_trust_only"},
        ]
        # Trust scores: a > b so prefer_higher_trust picks a.
        raw["trust_scores"] = {"a": 2.0, "b": 1.0}
        yaml_path = _write_yaml(tmp_path, raw)

        runner = FusionCommitteeRunner(yaml_path)
        bundle = _make_two_source_bundle()
        result = runner.run(bundle)

        assert set(result.per_member.keys()) == {
            "voting_only",
            "prefer_higher_trust_only",
        }
        for member_name, member in result.per_member.items():
            assert "macro_accuracy" in member.metrics, member_name
            assert member.metrics["macro_accuracy"] >= 0.0
            assert "f1" in member.metrics  # promoted from macro_accuracy
            # Selection map should be empty for these members.
            assert member.notes.get("selection_map") == {}

        # prefer_higher_trust should pick source ``a`` cleanly →
        # name + revenue both equal ``a``'s gold values → 1.0.
        pht = result.per_member["prefer_higher_trust_only"]
        assert pht.metrics["macro_accuracy"] == pytest.approx(1.0)

    def test_unknown_member_in_yaml_raises(self, tmp_path: Path) -> None:
        raw = _minimal_roster_dict()
        raw["members"].append({"name": "this_member_does_not_exist"})
        yaml_path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="Unknown member"):
            FusionCommitteeRunner(yaml_path)


# ---------------------------------------------------------------------------
# source_ids-keyed gold (papers at every tier, products task variants)
# ---------------------------------------------------------------------------


def _source_ids_bundle(*, as_list: bool) -> VariantBundle:
    """The two-source fixture with its gold keyed by ``source_ids`` (no
    ``id``), as the papers JSON gold (a list) or the products variant XML
    (a comma-joined string) ships it. Test gold = source ``a``'s values;
    val gold = source ``b``'s revenue (``a``'s + 5)."""
    bundle = _make_two_source_bundle()
    members = [[f"b_{i}", f"a_{i}"] for i in range(3)]
    key = members if as_list else [",".join(m) for m in members]
    gold = bundle.fusion_gold.drop(columns=["id"]).assign(source_ids=key)
    val = gold.assign(revenue=[float(100 * i + 5) for i in range(3)])
    bundle.fusion_gold = gold
    bundle.fusion_validation = val
    return bundle


class TestSourceIdsKeyedGold:
    """A gold keyed by member lists aligns to the fused records and feeds
    val-selection with its member ids."""

    @pytest.mark.parametrize("as_list", [True, False])
    def test_members_align_gold_rows(self, tmp_path: Path, as_list: bool) -> None:
        raw = _minimal_roster_dict()
        raw["gold_id_column"] = "source_ids"
        raw["members"] = [{"name": "prefer_higher_trust_only"}]
        runner = FusionCommitteeRunner(_write_yaml(tmp_path, raw))
        result = runner.run(_source_ids_bundle(as_list=as_list))
        metrics = result.per_member["prefer_higher_trust_only"].metrics
        assert metrics["num_evaluated_records"] == 3
        assert metrics["macro_accuracy"] == pytest.approx(1.0)
        # val surface scored too (val revenue is b's value -> 0.0 on revenue)
        assert metrics["macro_accuracy_val"] == pytest.approx(0.5)

    def test_val_selection_uses_member_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Val correspondences are the EM positives touching the val
        members. With the joined "b_i,a_i" strings taken as ids no pair is
        selected, every record fuses as a singleton, every candidate ties
        and the first one ("minimum") wins; with the member ids the pairs
        group and "maximum" (b's revenue == the val gold) wins."""
        from usecases_synthetic.lib import committee_fusion_c12 as mod

        monkeypatch.setattr(
            mod, "_selection_cache_path",
            lambda d: tmp_path / "baselines" / d / "fusion_committee_selection.json",
        )
        raw = _minimal_roster_dict()
        raw["gold_id_column"] = "source_ids"
        raw["pydi_candidates"]["numeric"] = [
            {"name": "minimum", "function": "minimum",
             "module": "PyDI.fusion.conflict_resolution.numeric"},
            {"name": "maximum", "function": "maximum",
             "module": "PyDI.fusion.conflict_resolution.numeric"},
        ]
        raw["members"] = [{"name": "pydi_per_attribute_optimal"}]
        runner = FusionCommitteeRunner(_write_yaml(tmp_path, raw))
        result = runner.run(_source_ids_bundle(as_list=True))
        selection = result.per_member["pydi_per_attribute_optimal"].notes["selection_map"]
        assert selection["revenue"] == "maximum"


def test_apply_gold_aliases_normalises_source_ids_only_for_source_ids_rosters() -> None:
    from usecases_synthetic.lib.committee_fusion_c12 import _apply_gold_aliases_and_lists

    gold = pd.DataFrame({"source_ids": [["crossref-2", "dblp-1"]], "title": ["t"]})
    source_ids_roster = _parse_roster({**_minimal_roster_dict(), "gold_id_column": "source_ids"})
    out = _apply_gold_aliases_and_lists(source_ids_roster, gold)
    assert out["source_ids"].tolist() == ["dblp-1,crossref-2"]

    id_roster = _parse_roster(_minimal_roster_dict())
    untouched = _apply_gold_aliases_and_lists(id_roster, gold)
    assert untouched is gold


# ---------------------------------------------------------------------------
# Native-type table consistency
# ---------------------------------------------------------------------------


class TestNativeTypesTable:
    """The :data:`_NATIVE_TYPES_BY_MEMBER` table encodes the C12 spec.

    These tests pin the spec so a refactor of the table can't silently
    flip a member's competence profile.
    """

    def test_pydi_per_attribute_optimal_is_empty(self) -> None:
        # Always selects PyDI per attribute; no native methods.
        assert _NATIVE_TYPES_BY_MEMBER["pydi_per_attribute_optimal"] == frozenset()

    def test_llm_only_is_universal(self) -> None:
        # Prompt v2: LLM handles every type natively.
        assert _NATIVE_TYPES_BY_MEMBER["llm_only"] == frozenset(
            {"string", "categorical", "date", "numeric", "list"}
        )

    def test_accusim_only_is_universal(self) -> None:
        # Type-aware similarity hook covers everything.
        assert _NATIVE_TYPES_BY_MEMBER["accusim_only"] == frozenset(
            {"string", "categorical", "date", "numeric", "list"}
        )

    def test_voting_only_and_pht_only_are_universal(self) -> None:
        for member in ("voting_only", "prefer_higher_trust_only"):
            assert _NATIVE_TYPES_BY_MEMBER[member] == frozenset(
                {"string", "categorical", "date", "numeric", "list"}
            ), member

    def test_truthfinder_fusionquery_casefusion_skip_numeric_and_list(self) -> None:
        for member in ("truthfinder_only", "fusionquery_only", "casefusion_only"):
            native = _NATIVE_TYPES_BY_MEMBER[member]
            assert "numeric" not in native, member
            assert "list" not in native, member
            assert native == frozenset({"string", "categorical", "date"}), member

    def test_ltm_only_covers_list_but_not_numeric(self) -> None:
        native = _NATIVE_TYPES_BY_MEMBER["ltm_only"]
        assert "list" in native
        assert "numeric" not in native
        assert native == frozenset({"string", "categorical", "date", "list"})


# ---------------------------------------------------------------------------
# Scoring-side value types
# ---------------------------------------------------------------------------


class TestScoringValueTypes:
    """``scoring_value_types`` brings fused + gold values to one type before
    scoring, so PyDI's lenient comparators compare what they are meant to
    (comparators unchanged, only type/parse fixes)."""

    def test_parser_defaults_to_empty_and_reads_block(self) -> None:
        assert _parse_roster(_minimal_roster_dict()).scoring_value_types == {}
        raw = {
            **_minimal_roster_dict(),
            "scoring_value_types": {"name": "string", "revenue": "year"},
        }
        assert _parse_roster(raw).scoring_value_types == {
            "name": "string",
            "revenue": "year",
        }

    def test_parser_rejects_unknown_kind(self) -> None:
        raw = {**_minimal_roster_dict(), "scoring_value_types": {"name": "date"}}
        with pytest.raises(ValueError, match="scoring_value_types"):
            _parse_roster(raw)

    def test_papers_yaml_declares_the_fixed_attributes(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "config" / "committees" / "fusion_committee_papers.yaml"
        )
        roster = _parse_roster(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert roster.scoring_value_types == {
            "publication_year": "year",
            "authors": "list",
            "volume": "string",
            "issue": "string",
        }

    @pytest.mark.parametrize(
        "domain", ["fusion_committee", "fusion_committee_games",
                   "fusion_committee_music", "fusion_committee_products"],
    )
    def test_other_domains_declare_none(self, domain: str) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "config" / "committees" / f"{domain}.yaml"
        )
        roster = _parse_roster(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert roster.scoring_value_types == {}

    def test_year_value(self) -> None:
        import numpy as np

        from usecases_synthetic.lib.committee_fusion_c12 import _as_year_value

        for value in (2019, np.int64(2019), 2019.0, np.float64(2019.0),
                      "2019", " 2019 ", "2019.0"):
            assert _as_year_value(value) == "2019", value
        # Anything else is left to the comparator exactly as before.
        for value in ("2 019", "wo29", "2019-05-01", "202", True):
            assert _as_year_value(value) == value
        assert _as_year_value(202) == "202"
        nan = _as_year_value(float("nan"))
        assert isinstance(nan, float) and nan != nan

    def test_year_only_match_becomes_meaningful(self) -> None:
        from PyDI.fusion.evaluation import year_only_match

        from usecases_synthetic.lib.committee_fusion_c12 import _as_year_value

        # The bug: pd.to_datetime reads an int as epoch ns (1970).
        assert year_only_match(2019, 2020) is True
        assert year_only_match("2019", 2019) is False
        # After the type fix the years themselves are compared.
        assert year_only_match(_as_year_value(2019), _as_year_value(2020)) is False
        assert year_only_match(_as_year_value(2019), _as_year_value(2019)) is True
        assert year_only_match(_as_year_value("2019"), _as_year_value(2019)) is True
        assert year_only_match(_as_year_value(2019.0), _as_year_value(2019)) is True
        # A corrupted fused year still fails.
        assert year_only_match(_as_year_value("2 019"), _as_year_value(2019)) is False

    def test_string_value(self) -> None:
        from PyDI.fusion.evaluation import tokenized_match

        from usecases_synthetic.lib.committee_fusion_c12 import _as_string_value

        assert _as_string_value(19.0) == "19"
        assert _as_string_value(19) == "19"
        assert _as_string_value("19") == "19"
        assert _as_string_value("CSCW") == "CSCW"
        assert _as_string_value(1.5) == 1.5
        nan = _as_string_value(float("nan"))
        assert isinstance(nan, float) and nan != nan
        # The bug: "19.0" loses its dot -> token "190".
        assert tokenized_match("19", 19.0) is False
        assert tokenized_match(_as_string_value("19"), _as_string_value(19.0)) is True
        assert tokenized_match(_as_string_value("20"), _as_string_value(19.0)) is False

    def test_list_value(self) -> None:
        from PyDI.fusion.evaluation import tokenized_match

        from usecases_synthetic.lib.committee_fusion_c12 import _as_list_value

        assert _as_list_value("['Ilaria Amantea', ' Antonio Di Leva']") == [
            "Ilaria Amantea",
            "Antonio Di Leva",
        ]
        assert _as_list_value([" A ", "B"]) == ["A", "B"]
        assert _as_list_value(("A", "B")) == ["A", "B"]
        prose = "A, B, and C"
        assert _as_list_value(prose) == prose
        nan = _as_list_value(float("nan"))
        assert isinstance(nan, float) and nan != nan
        gold = ["Antonio Di Leva", "Ilaria Amantea"]
        # The bug: a list-literal string is ONE token against the gold list,
        # and a whitespace-padded element never equals the gold element.
        literal = "['Ilaria Amantea', ' Antonio Di Leva']"
        assert tokenized_match(literal, gold) is False
        assert tokenized_match(["Ilaria Amantea", " Antonio Di Leva"], gold) is False
        assert tokenized_match(_as_list_value(literal), _as_list_value(gold)) is True
        assert tokenized_match(_as_list_value(["Ilaria Amantea"]), gold) is False

    def test_apply_is_identity_without_types(self) -> None:
        from usecases_synthetic.lib.committee_fusion_c12 import (
            _apply_scoring_value_types,
        )

        frame = pd.DataFrame({"name": [1.0], "revenue": [2019]})
        roster = _parse_roster(_minimal_roster_dict())
        assert _apply_scoring_value_types(roster, frame) is frame

    def test_apply_converts_a_copy(self) -> None:
        from usecases_synthetic.lib.committee_fusion_c12 import (
            _apply_scoring_value_types,
        )

        roster = _parse_roster(
            {**_minimal_roster_dict(),
             "scoring_value_types": {"revenue": "year", "name": "string"}}
        )
        frame = pd.DataFrame({"name": [19.0, None], "revenue": [2019, 2020],
                              "other": [2019, 2020]})
        out = _apply_scoring_value_types(roster, frame)
        assert out is not frame
        assert out["revenue"].tolist() == ["2019", "2020"]
        assert out["name"].tolist()[0] == "19"
        assert pd.isna(out["name"].tolist()[1])
        assert out["other"].tolist() == [2019, 2020]
        # Input untouched.
        assert frame["revenue"].tolist() == [2019, 2020]
        assert frame["name"].tolist()[0] == 19.0

    def test_gold_prep_applies_types(self) -> None:
        from usecases_synthetic.lib.committee_fusion_c12 import (
            _apply_gold_aliases_and_lists,
        )

        raw = {
            **_minimal_roster_dict(),
            "gold_id_column": "source_ids",
            "gold_list_columns": ["authors"],
            "scoring_value_types": {"authors": "list", "revenue": "year"},
        }
        gold = pd.DataFrame(
            {"source_ids": [["dblp-1", "crossref-2"]],
             "authors": ["['A', ' B']"], "revenue": [2019]}
        )
        out = _apply_gold_aliases_and_lists(_parse_roster(raw), gold)
        assert out["authors"].tolist() == [["A", "B"]]
        assert out["revenue"].tolist() == ["2019"]


def _papers_like_bundle() -> VariantBundle:
    """Two papers-like sources: ``crossref`` (trust 3) ships int years +
    whitespace-padded list-literal authors (like the base JSONL), ``dblp``
    string years (like a CSV variant). Gold (keyed by ``source_ids``) has
    int years, list authors and a float volume (the JSON reader's
    inference). Entity 2's fused year (crossref: 2011) is wrong against
    the gold (2012)."""
    n = 3
    src_a = pd.DataFrame(
        {
            "id": [f"crossref-{i:05d}" for i in range(n)],
            "publication_year": [2010, 2011, 2011],
            "authors": [f"['Ann {i}', ' Bob {i}']" for i in range(n)],
            "volume": [str(10 + i) for i in range(n)],
        }
    )
    src_a.attrs["dataset_name"] = "crossref"
    src_b = pd.DataFrame(
        {
            "id": [f"dblp-{i:05d}" for i in range(n)],
            "publication_year": ["2010", "2011", "2012"],
            "authors": [f"['Ann {i}', 'Bob {i}']" for i in range(n)],
            "volume": [str(10 + i) for i in range(n)],
        }
    )
    src_b.attrs["dataset_name"] = "dblp"
    correspondences = pd.DataFrame(
        [{"id1": f"crossref-{i:05d}", "id2": f"dblp-{i:05d}", "score": 1.0} for i in range(n)]
    )
    gold = pd.DataFrame(
        {
            "source_ids": [[f"crossref-{i:05d}", f"dblp-{i:05d}"] for i in range(n)],
            "publication_year": [2010, 2011, 2012],
            "authors": [[f"Ann {i}", f"Bob {i}"] for i in range(n)],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    return VariantBundle(
        domain="papers",
        level="baseline",
        sources={"crossref": src_a, "dblp": src_b},
        target_schema={"properties": {}},
        sm_mapping=None,
        em_gold={("crossref", "dblp"): correspondences.assign(label=1)},
        em_splits={},
        fusion_gold=gold,
        fusion_validation=None,
        pooled_positives=None,
        variant_root=Path("/tmp/fusion_c12_papers_test"),
    )


def _papers_like_roster(*, with_types: bool) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "seed": 42,
        "fused_id_column": "id",
        "gold_id_column": "source_ids",
        "gold_list_columns": ["authors"],
        "trust_scores": {"crossref": 3.0, "dblp": 1.0},
        "column_mapping": {},
        "evaluation_functions": {
            "publication_year": "year_only_match",
            "authors": "tokenized_match",
            "volume": "tokenized_match",
        },
        "attribute_types": {
            "publication_year": "date",
            "authors": "list",
            "volume": "string",
        },
        "pydi_candidates": {
            t: [{"name": "voting", "function": "voting",
                 "module": "PyDI.fusion.conflict_resolution.general"}]
            for t in ("date", "list", "string")
        },
        "members": [{"name": "prefer_higher_trust_only"}],
    }
    if with_types:
        raw["scoring_value_types"] = {
            "publication_year": "year",
            "authors": "list",
            "volume": "string",
        }
    return raw


class TestPapersScoringEndToEnd:
    """prefer_higher_trust_only fuses source ``crossref`` (int years, padded
    list-literal authors). Before the fix: year 1.0 (any int year matches),
    authors 0.0, volume 0.0; after: year 2/3, authors 1.0, volume 1.0."""

    @pytest.fixture(autouse=True)
    def _no_repo_writes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from usecases_synthetic.lib import committee_fusion_c12 as mod

        monkeypatch.setattr(mod, "_op_log_dir", lambda d, lvl: tmp_path / "oplog")
        monkeypatch.setattr(
            mod, "_selection_cache_path",
            lambda d: tmp_path / "baselines" / d / "fusion_committee_selection.json",
        )

    def test_sources_authors_parsed_to_lists(self) -> None:
        from usecases_synthetic.lib.committee_fusion import _parse_source_list_columns

        bundle = _papers_like_bundle()
        parsed = _parse_source_list_columns(bundle.sources, "papers")
        assert parsed["crossref"]["authors"].tolist()[0] == ["Ann 0", " Bob 0"]
        # Loader-side frames stay strings (SM / EM / Norm need them).
        assert isinstance(bundle.sources["crossref"]["authors"].tolist()[0], str)
        # Other domains keep their authors-like columns untouched.
        other = _parse_source_list_columns(bundle.sources, "test_domain")
        assert isinstance(other["crossref"]["authors"].tolist()[0], str)

    def test_with_scoring_value_types(self, tmp_path: Path) -> None:
        runner = FusionCommitteeRunner(
            _write_yaml(tmp_path, _papers_like_roster(with_types=True))
        )
        result = runner.run(_papers_like_bundle())
        metrics = result.per_member["prefer_higher_trust_only"].metrics
        assert metrics["num_evaluated_records"] == 3
        assert metrics["publication_year_accuracy"] == pytest.approx(2 / 3)
        assert metrics["authors_accuracy"] == pytest.approx(1.0)
        assert metrics["volume_accuracy"] == pytest.approx(1.0)
        # The member's fused output itself is not rewritten.
        fused = result.per_member["prefer_higher_trust_only"].predictions
        assert sorted(fused["publication_year"].tolist()) == [2010, 2011, 2011]
        assert any(v == ["Ann 0", " Bob 0"] for v in fused["authors"].tolist())

    def test_without_scoring_value_types_shows_the_bug(self, tmp_path: Path) -> None:
        runner = FusionCommitteeRunner(
            _write_yaml(tmp_path, _papers_like_roster(with_types=False))
        )
        result = runner.run(_papers_like_bundle())
        metrics = result.per_member["prefer_higher_trust_only"].metrics
        assert metrics["publication_year_accuracy"] == pytest.approx(1.0)
        assert metrics["authors_accuracy"] == pytest.approx(0.0)
        assert metrics["volume_accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Opt-in fused-output dump (fused_dump_dir / MADI_COMMITTEE_FUSED_DUMP_DIR)
# ---------------------------------------------------------------------------


class TestFusedDump:
    """Every member's fused output can be written for re-scoring; off by
    default, and the metrics do not depend on it."""

    @pytest.fixture(autouse=True)
    def _no_repo_writes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from usecases_synthetic.lib import committee_fusion_c12 as mod

        monkeypatch.setattr(mod, "_op_log_dir", lambda d, lvl: tmp_path / "oplog")
        monkeypatch.setattr(
            mod, "_selection_cache_path",
            lambda d: tmp_path / "baselines" / d / "fusion_committee_selection.json",
        )
        monkeypatch.delenv(mod.FUSED_DUMP_ENV, raising=False)

    @staticmethod
    def _runner(tmp_path: Path) -> C12FusionCommitteeRunner:
        raw = _minimal_roster_dict()
        raw["members"] = [{"name": "voting_only"}, {"name": "prefer_higher_trust_only"}]
        return FusionCommitteeRunner(_write_yaml(tmp_path, raw))

    def test_explicit_dir_writes_the_engine_output_per_member(self, tmp_path: Path) -> None:
        import ast

        dump = tmp_path / "dump"
        result = self._runner(tmp_path).run(_make_two_source_bundle(), fused_dump_dir=dump)
        for name, member in result.per_member.items():
            path = dump / f"{name}.csv.gz"
            assert path.is_file(), name
            assert member.notes["fused_dump"] == str(path)
            back = pd.read_csv(path, dtype={"_id": str})
            assert list(back.columns) == list(member.predictions.columns)
            assert sorted(back["_id"]) == sorted(member.predictions["_id"].astype(str))
            assert back["_fusion_sources"].map(ast.literal_eval).map(sorted).tolist() == (
                member.predictions["_fusion_sources"].map(sorted).tolist()
            )
        assert not list(dump.glob(".*.tmp"))

    def test_metrics_do_not_depend_on_the_dump(self, tmp_path: Path) -> None:
        plain = self._runner(tmp_path).run(_make_two_source_bundle())
        dumped = self._runner(tmp_path).run(
            _make_two_source_bundle(), fused_dump_dir=tmp_path / "dump"
        )
        for name in plain.per_member:
            assert plain.per_member[name].metrics == dumped.per_member[name].metrics
            assert "fused_dump" not in plain.per_member[name].notes

    def test_env_var_names_the_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from usecases_synthetic.lib import committee_fusion_c12 as mod

        monkeypatch.setenv(mod.FUSED_DUMP_ENV, str(tmp_path / "root"))
        self._runner(tmp_path).run(_make_two_source_bundle())
        level_dir = tmp_path / "root" / "test_domain" / "baseline"
        assert sorted(p.name for p in level_dir.glob("*.csv.gz")) == [
            "prefer_higher_trust_only.csv.gz",
            "voting_only.csv.gz",
        ]

    def test_default_writes_nothing(self, tmp_path: Path) -> None:
        self._runner(tmp_path).run(_make_two_source_bundle())
        assert not list(tmp_path.rglob("*.csv.gz"))

    def test_dump_is_byte_reproducible(self, tmp_path: Path) -> None:
        self._runner(tmp_path).run(_make_two_source_bundle(), fused_dump_dir=tmp_path / "a")
        self._runner(tmp_path).run(_make_two_source_bundle(), fused_dump_dir=tmp_path / "b")
        files = sorted((tmp_path / "a").glob("*.csv.gz"))
        assert files
        for f in files:
            assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes()

    def test_failed_dump_aborts_the_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from usecases_synthetic.lib import committee_fusion_c12 as mod

        def boom(*args: Any, **kwargs: Any) -> Path:
            raise OSError("disk full")

        monkeypatch.setattr(mod, "_dump_member_fused", boom)
        with pytest.raises(mod._FusedDumpError):
            self._runner(tmp_path).run(_make_two_source_bundle(), fused_dump_dir=tmp_path / "d")
