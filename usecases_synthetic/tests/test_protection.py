"""Tests for protection set construction and the continuous closeness."""

from __future__ import annotations

import pytest

from usecases_synthetic.lib.protection import (
    ToleranceSpec,
    _numeric_readings,
    _parse_float,
    build_drop_corner_protection_set,
    build_expanded_positives,
    cell_has_close_survivor,
    is_close_enough,
    is_protected,
)

_CONT = ToleranceSpec(kind="continuous", threshold=0.03)


class TestContinuousClosenessLocales:
    """The continuous kind reads Knob 5 locale forms correctly.

    ``_parse_float`` drops commas and reads the dot as the decimal point,
    so K5's de_DE / fr_FR forms were misread and the K6 close-survivor
    gate lost every de_DE / fr_FR sibling.
    """

    @pytest.mark.parametrize(
        ("value", "readings"),
        [
            ("2.000,0", (2000.0,)),  # de_DE, was 2.0
            ("500,0", (500.0,)),  # de_DE / fr_FR, was 5000
            ("14 000,0", (14000.0,)),  # fr_FR, was 140000
            ("14\u00a0000,0", (14000.0,)),  # fr_FR NBSP grouping
            ("14\u202f000,0", (14000.0,)),  # fr_FR NNBSP grouping
            ("1.234,56", (1234.56,)),  # de_DE, was 1.23456
            ("1.234.567", (1234567.0,)),  # de_DE, was unparseable
            ("2.048,0", (2048.0,)),  # de_DE, was 2.048
            ("8,000.0", (8000.0,)),  # en_US
            ("1 234", (1234.0,)),  # fr_FR integer
            ("0,500", (0.5,)),  # de_DE / fr_FR; no en_US 500 (zero-led group)
            # Ambiguous forms keep both readings.
            ("2.000", (2.0, 2000.0)),
            ("1,234", (1234.0, 1.234)),
            ("3.134", (3.134, 3134.0)),  # plain 3-decimal: also de_DE 3134
        ],
    )
    def test_locale_readings(self, value: str, readings: tuple[float, ...]) -> None:
        assert _numeric_readings(value) == readings

    @pytest.mark.parametrize(
        ("value", "target", "wrong_target"),
        [
            ("2.000,0", "2000.0", "2.0"),
            ("500,0", "500", "5000"),
            ("14 000,0", "14000.0", "140000"),
            ("14\u00a0000,0", "14000.0", "140000"),
            ("14\u202f000,0", "14000.0", "140000"),
            ("1.234,56", "1234.56", "1.23456"),
            ("1.234.567", "1234567", "1.234567"),
            # The spurious closeness of the old parser: 2048 MB is not 2 GB.
            ("2.048,0", "2048.0", "2.0"),
        ],
    )
    def test_decimal_comma_forms_close_to_true_value_only(
        self, value: str, target: str, wrong_target: str
    ) -> None:
        assert is_close_enough(value, target, _CONT)
        assert not is_close_enough(value, wrong_target, _CONT)

    @pytest.mark.parametrize(
        ("value", "targets"),
        [("2.000", ("2.0", "2000")), ("1,234", ("1234", "1.234"))],
    )
    def test_ambiguous_forms_close_under_either_reading(
        self, value: str, targets: tuple[str, str]
    ) -> None:
        for target in targets:
            assert is_close_enough(value, target, _CONT)
        assert not is_close_enough(value, "20", _CONT)

    def test_k6_gate_counts_de_de_and_fr_fr_survivors(self) -> None:
        gold = ["2000.0"]
        assert cell_has_close_survivor(gold, ["corrupted", "2.000,0"], _CONT)
        assert cell_has_close_survivor(gold, [None, "2 000,0"], _CONT)
        assert cell_has_close_survivor(gold, ["2\u202f000,0"], _CONT)
        assert not cell_has_close_survivor(gold, ["corrupted", "2.500,0"], _CONT)


class TestContinuousClosenessPlainUnchanged:
    """Values today's ``_parse_float`` reads correctly keep that reading."""

    @pytest.mark.parametrize(
        "value",
        [
            "2000.0",
            "2000",
            "0.5",
            "-3.25",
            "+1.5",
            "123",
            "12.5",
            "148700000000",
            "1e5",
            "2.1492E2",
            "-4.5e-3",
            ".5",
            "5.",
            "8,000.0",
            "148,700,000,000",
            "$1,299.99",
            "1 234.5",
            "  42.0  ",
            # Zero-led: never a de_DE thousands group.
            "0.178",
            "0.008",
            "-0.125",
        ],
    )
    def test_single_reading_equals_parse_float(self, value: str) -> None:
        assert _numeric_readings(value) == (_parse_float(value),)

    def test_zero_led_plain_values_keep_their_verdict(self) -> None:
        # Products base weight_g cells in kg next to a gram gold value
        # (products_4): read as de_DE thousands they would turn close.
        assert not is_close_enough("0.178", "178.0", _CONT)
        assert not is_close_enough("0.008", "8.0", _CONT)

    @pytest.mark.parametrize(
        "value", ["", "   ", "abc", "SAR 219", "R5 299,98 incl tax", "1.2.3,4,5"]
    )
    def test_unparseable_stays_unparseable(self, value: str) -> None:
        assert _parse_float(value) is None
        assert _numeric_readings(value) == ()
        assert not is_close_enough(value, "1", _CONT)

    def test_nan_and_inf_never_close(self) -> None:
        assert not is_close_enough("nan", "1", _CONT)
        assert not is_close_enough("inf", "1", _CONT)
        assert not is_close_enough("1", "nan", _CONT)

    @pytest.mark.parametrize(
        ("value", "target", "expected"),
        [
            ("2000.0", "2000", True),
            ("2059.0", "2000", True),  # +2.95%
            ("2061.0", "2000", False),  # +3.05%
            ("1.5", "1.52", True),
            ("1.5", "1.6", False),
            ("148700000000", "148,700,000,000", True),
            ("$1,299.99", "1299.99", True),
            ("2.1492E2", "214.92", True),
            ("8,000.0", "8000", True),
            (2000.0, "2000", True),
            (7, "7.0", True),
            ("0", "0", True),
            ("0.0001", "0", False),
        ],
    )
    def test_plain_verdicts(self, value: object, target: str, expected: bool) -> None:
        assert is_close_enough(value, target, _CONT) is expected

    def test_target_side_parsed_as_before(self) -> None:
        # The target is the canonical fusion value: read by _parse_float,
        # so a canonical "2.000" target stays 2.0.
        assert is_close_enough("2.0", "2.000", _CONT)
        assert not is_close_enough("2000.0", "2.000", _CONT)
        # Empty target is vacuously close (unchanged).
        assert is_close_enough("anything", "", _CONT)


class TestProtectionSet:
    """Tests for expanded_positives and is_protected."""

    def test_is_protected_true(self, mock_protection_set: set[str]) -> None:
        assert is_protected(
            "http://dbpedia.org/resource/Company_0", mock_protection_set
        )
        assert is_protected("fullcontact_0", mock_protection_set)

    def test_is_protected_false(self, mock_protection_set: set[str]) -> None:
        assert not is_protected("unknown_entity_99", mock_protection_set)
        assert not is_protected("fullcontact_999", mock_protection_set)

    def test_build_expanded_positives_companies(self) -> None:
        """expanded_positives for companies contains pooled + EM gold IDs.

        This test requires the actual data files to be present (LFS).
        """
        positives = build_expanded_positives("companies")

        # Must contain pooled positive IDs (2803 pairs → many entity IDs)
        # At minimum we expect some dbpedia and forbes IDs
        dbpedia_ids = {eid for eid in positives if "dbpedia.org" in eid}
        forbes_ids = {eid for eid in positives if "forbes.com" in eid}

        assert len(dbpedia_ids) > 0, "No dbpedia IDs in expanded_positives"
        assert len(forbes_ids) > 0, "No forbes IDs in expanded_positives"

        # Total should be substantial (EM gold + fusion gold + pool)
        assert (
            len(positives) > 100
        ), f"expanded_positives only has {len(positives)} IDs — expected >100"

    def test_build_expanded_positives_includes_pool(self) -> None:
        """Spot-check: a known pooled pair's IDs must appear."""
        positives = build_expanded_positives("companies")

        # From the pooled_positives.csv first data row
        assert "http://dbpedia.org/resource/A2A" in positives
        assert "http://www.forbes.com/companies/a2a/" in positives


class TestDropCornerProtectionSet:
    """Drop-corner protection set for K2.

    On ``pool_quality: live`` domains (products) BOTH pool and EM gold
    are coextensive with the full record set, so any protection scheme
    that includes EM gold leaves zero droppable entities. The
    drop-corner protection set is therefore fusion val/test only
    under default ``gold``; the EM regeneration handles dropped
    EM-gold members downstream by pruning Set 1 and corner-mining
    Set 2 from the surviving pool. ``silver`` widens back to include
    pool (the silver-standard semantics).
    """

    def test_gold_protection_is_fusion_only(self) -> None:
        """Under ``protection_source="gold"``, the drop-corner set is
        exactly the fusion val/test gold — narrower than
        ``build_expanded_positives`` which also includes EM gold +
        pool. On ``pool_quality: live`` domains (products), the broader
        set was coextensive with the full pool and made drop-corner a
        noop; the narrowing is what unblocks the operator."""
        from usecases_synthetic.lib.protection import _load_fusion_protected_ids

        gold = build_drop_corner_protection_set("companies", protection_source="gold")
        fusion_only = _load_fusion_protected_ids("companies")
        assert gold == fusion_only, (
            "drop-corner gold protection must equal fusion val/test only "
            "(no EM gold, no pool)"
        )

    def test_gold_protection_is_strict_subset_of_expanded_positives(self) -> None:
        """``gold`` (fusion-only) must be a strict subset of
        ``expanded_positives`` (em ∪ fusion ∪ pool). Confirms the
        narrowing actually drops entities from the protection set,
        which is the whole point of the helper."""
        full = build_expanded_positives("companies")
        gold = build_drop_corner_protection_set("companies", protection_source="gold")
        assert gold.issubset(full)
        assert len(gold) < len(full), (
            f"gold set ({len(gold)}) must be strictly smaller than "
            f"expanded_positives ({len(full)}) — otherwise pool + EM gold "
            "aren't being excluded."
        )

    def test_silver_protection_includes_pool(self) -> None:
        """Under ``protection_source="silver"``, the drop-corner set
        widens to fusion ∪ pool (silver-standard semantics: every
        pool-cluster member is fusion-recoverable, therefore
        protected). EM gold is still NOT included on its own — it
        only contributes via overlap with pool members."""
        from usecases_synthetic.lib.protection import (
            _load_fusion_protected_ids,
            _load_pooled_positive_ids,
        )

        silver = build_drop_corner_protection_set(
            "companies", protection_source="silver"
        )
        expected = _load_fusion_protected_ids("companies") | _load_pooled_positive_ids(
            "companies"
        )
        assert silver == expected

    def test_em_gold_members_droppable_under_gold(self) -> None:
        """EM gold positive ids that are NOT also in fusion val/test
        gold must NOT be in the gold protection set — i.e. drop-corner
        is allowed to drop them. The EM regeneration rebuilds the EM splits
        from the surviving pool after the drops land."""
        from usecases_synthetic.lib.protection import (
            _load_em_gold_ids,
            _load_fusion_protected_ids,
        )

        em_only = _load_em_gold_ids("companies") - _load_fusion_protected_ids(
            "companies"
        )
        gold = build_drop_corner_protection_set("companies", protection_source="gold")
        # EM-only ids (not in fusion val/test) must be absent from gold.
        # If the test data has no em-only ids, this is vacuously true —
        # but for companies this set should be nonempty (companies has
        # an EM gold of pairs that doesn't fully overlap fusion val/test).
        intersection = em_only & gold
        assert intersection == set(), (
            f"{len(intersection)} EM-gold-only ids leaked into gold protection — "
            "they should be droppable under the EM-regen contract."
        )

    def test_gold_protection_default_when_param_omitted(self) -> None:
        """The default protection_source value is 'gold' — omitting the
        param must produce the narrow set, not the wide one."""
        default = build_drop_corner_protection_set("companies")
        gold = build_drop_corner_protection_set("companies", protection_source="gold")
        assert default == gold
