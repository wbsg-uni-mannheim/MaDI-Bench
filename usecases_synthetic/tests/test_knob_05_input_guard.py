"""Knob 05: value-preservation guard on the INPUT of number rewrites.

The input parser ``_parse_number`` (a locale-agnostic heuristic, kept
unchanged) reads a decimal comma as an en_US thousands separator whenever the
digits after the comma are not a group of three, so K5 used to rewrite e.g.
"€385,00" as 38,500 and log it as a format-only ``reformat_number`` row. The
number operators now parse their input with ``_parse_input_number``: the
heuristic value is accepted only if it equals one of the strict readings of
the input (en_US, de_DE, fr_FR, plain incl. scientific, after stripping the
currency symbols / codes the heuristic strips). A rejected cell is left
unchanged and the K5 dispatcher logs it as ``unparseable_value``
(``SKIP_UNPARSEABLE``), distinct from ``roundtrip_parse_fail``.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from usecases_synthetic.lib.format_operators import (
    _parse_input_number,
    _parse_number,
    _parse_number_in_locale,
    _strict_number_readings,
    _strip_currency_affixes,
    reconvert_currency,
    reconvert_unit,
    reformat_number,
    reformat_number_suffix,
)
from usecases_synthetic.scripts.apply_knob_05_format import (
    SKIP_ROUNDTRIP,
    SKIP_UNPARSEABLE,
    apply_knob_05,
)

_LOCALES = ("en_US", "de_DE", "fr_FR", "plain")

# The ten products price cells written with a decimal comma (raw sources,
# (source, entity id, column, raw value, heuristic misreading,
# true value).
PRODUCTS_DECIMAL_COMMA_CELLS: list[tuple[str, str, str, str, str, str]] = [
    ("products_1", "products_1_9830450", "list_price", "6,25", "625", "6.25"),
    ("products_1", "products_1_13127008", "list_price", "€385,00", "38500", "385.00"),
    ("products_1", "products_1_1829549", "list_price", "71,74 €", "7174", "71.74"),
    ("products_1", "products_1_1993958", "list_price", "€ 249,99", "24999", "249.99"),
    ("products_2", "products_2_23109514", "priceAmount", "€ 399,99", "39999", "399.99"),
    ("products_3", "products_3_3463117", "Price", "€1.229,00", "1.229", "1229.00"),
    ("products_3", "products_3_37417537", "Price", "294,90 €", "29490", "294.90"),
    ("products_3", "products_3_66798457", "Price", "€155,00", "15500", "155.00"),
    ("products_4", "products_4_56844385", "amt", "387,01 €", "38701", "387.01"),
    ("products_4", "products_4_52397662", "amt", "€109,00", "10900", "109.00"),
]

_CELL_PARAMS = [
    pytest.param(*cell, id=f"{cell[1]}-{cell[3]}") for cell in PRODUCTS_DECIMAL_COMMA_CELLS
]


class TestProductsDecimalCommaCells:
    """The ten products cells: misread by the heuristic, rejected by the guard,
    and never rewritten by any number operator."""

    @pytest.mark.parametrize("source,entity_id,column,raw,misread,true_value", _CELL_PARAMS)
    def test_heuristic_misreads(
        self, source: str, entity_id: str, column: str, raw: str, misread: str, true_value: str
    ) -> None:
        """Documents the defect the guard exists for (``_parse_number`` is
        unchanged)."""
        assert _parse_number(raw) == Decimal(misread)
        assert _strict_number_readings(raw) == {Decimal(true_value)}

    @pytest.mark.parametrize("source,entity_id,column,raw,misread,true_value", _CELL_PARAMS)
    def test_guard_rejects(
        self, source: str, entity_id: str, column: str, raw: str, misread: str, true_value: str
    ) -> None:
        assert _parse_input_number(raw) is None

    @pytest.mark.parametrize("source,entity_id,column,raw,misread,true_value", _CELL_PARAMS)
    def test_no_operator_rewrites(
        self, source: str, entity_id: str, column: str, raw: str, misread: str, true_value: str
    ) -> None:
        for locale in _LOCALES:
            assert reformat_number(raw, locale) is None, locale
        assert reformat_number_suffix(raw, "K") is None
        assert reconvert_unit(raw, "magnitude", "raw", "millions") is None
        assert reconvert_unit(raw, "file_size", "GB", "MB") is None
        assert reconvert_currency(raw, "USD", "EUR") is None


class TestParseInputNumber:
    """Inputs the guard must keep accepting, and the ones it rejects."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("2000.0", "2000.0"),  # plain
            ("64", "64"),
            ("-12.5", "-12.5"),
            ("0.99", "0.99"),
            ("2.1492E2", "214.92"),  # scientific (products sources)
            ("1.75E3", "1750"),
            ("1,234.5", "1234.5"),  # en_US
            ("1,234,567.89", "1234567.89"),
            ("1,234", "1234"),  # ambiguous: heuristic = the en_US reading
            ("104.988", "104.988"),  # ambiguous: heuristic = the plain reading
            ("1.234.567,5", "1234567.5"),  # de_DE, heuristic reads it right
            ("1 234,5", "1234.5"),  # fr_FR, heuristic reads it right
            ("$1,299.99", "1299.99"),  # currency symbol stripped
            ("1299.99 €", "1299.99"),
            ("USD 12.50", "12.50"),
            ("1.5B", "1.5E9"),  # suffix notation
            ("200M", "2E8"),
            ("1,234.5K", "1234500"),
        ],
    )
    def test_accepted(self, value: str, expected: str) -> None:
        parsed = _parse_input_number(value)
        assert parsed == Decimal(expected)
        # Accepting never changes what the heuristic reads.
        assert parsed == _parse_number(value)

    @pytest.mark.parametrize(
        "value",
        [
            "1.234,56",  # de_DE decimal comma, heuristic 1.23456
            "500,0",  # heuristic 5000
            "2.000,0",  # heuristic 2.0
            "12,34,567",  # en_IN grouping: no strict reading
        ],
    )
    def test_rejected_misreading(self, value: str) -> None:
        assert _parse_number(value) is not None
        assert _parse_input_number(value) is None

    @pytest.mark.parametrize(
        "value",
        ["SAR 219", "R 921,73 excl tax", "N35,999.00", "e1005925", "1:1", "", "null", "abc"],
    )
    def test_rejected_unreadable(self, value: str) -> None:
        assert _parse_number(value) is None
        assert _parse_input_number(value) is None

    @pytest.mark.parametrize("value", ["Infinity", "-inf", "sNaN", "NaN"])
    def test_rejected_non_finite(self, value: str) -> None:
        assert _parse_input_number(value) is None

    def test_non_string_input(self) -> None:
        assert _parse_input_number(64.0) == Decimal("64.0")  # type: ignore[arg-type]


class TestStrictNumberReadings:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1,234", {"1234", "1.234"}),
            ("2.000", {"2.000", "2000"}),
            ("€385,00", {"385.00"}),
            ("2.1492E2", {"214.92"}),
            ("1.5B", {"1.5E9"}),
            ("1 234,5", {"1234.5"}),
            ("SAR 219", set()),
        ],
    )
    def test_readings(self, value: str, expected: set[str]) -> None:
        assert _strict_number_readings(value) == {Decimal(x) for x in expected}

    @pytest.mark.parametrize(
        "value",
        ["€385,00", "71,74 €", "€ 249,99", "$1,299.99", "1299.99 $", "USD 12.50", "12.50 EUR", "£5", "₹1,000"],
    )
    def test_strips_like_the_heuristic(self, value: str) -> None:
        """_strip_currency_affixes mirrors the stripping inside _parse_number."""
        assert _parse_number(_strip_currency_affixes(value)) == _parse_number(value)


class TestOperatorsStillAccept:
    """Regression: valid inputs are rewritten exactly as before."""

    @pytest.mark.parametrize(
        "value,locale,expected",
        [
            ("2000.0", "de_DE", "2.000,0"),
            ("2.1492E2", "fr_FR", "214,92"),
            ("1,234.5", "plain", "1234.5"),
            ("1,234", "de_DE", "1.234"),
            ("€1,299.99", "de_DE", "1.299,99"),
        ],
    )
    def test_reformat_number(self, value: str, locale: str, expected: str) -> None:
        result = reformat_number(value, locale)
        assert result is not None
        assert result[0] == expected

    def test_other_operators(self) -> None:
        assert reformat_number_suffix("1500000000", "B")[0] == "1.50B"
        assert reconvert_unit("8", "file_size", "GB", "MB")[0] == "8000"
        assert reconvert_unit("65170000000", "magnitude", "raw", "billions")[0] == "65.17"
        assert reconvert_currency("1000.00", "USD", "EUR")[0] == "921.50"


def _base_config(family: str, pools: dict[str, list[str]], unit_pool: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": "products",
        "attribute_classes": {"s1": {"v": family}},
        "id_columns": {"s1": "id"},
        "format_pools_per_level": {lv: {family: pools[lv]} for lv in ("easy", "medium", "hard")},
        "within_source_consistency": {"easy": "source", "medium": "source", "hard": "row"},
        "unit_pool_per_level": {lv: unit_pool for lv in ("easy", "medium", "hard")},
        "source_magnitude_context": {},
    }


def _sources(values: list[str]) -> dict[str, pd.DataFrame]:
    df = pd.DataFrame({"id": [f"s1-{i}" for i in range(len(values))], "v": values})
    df.attrs["dataset_name"] = "s1"
    return {"s1": df}


class TestApplyKnob05Guard:
    """End to end through apply_knob_05: guarded cells stay unchanged, get no
    provenance row and exactly one ``unparseable_value`` skip row."""

    _VALID = ["2000.0", "2.1492E2", "1,234.5", "1,234", "64", "219.99", "1.75E3", "0.5"]
    _BAD = [cell[3] for cell in PRODUCTS_DECIMAL_COMMA_CELLS] + ["SAR 219"]

    @pytest.fixture
    def number_config(self) -> dict[str, Any]:
        return _base_config(
            "number",
            {"easy": ["en_US", "plain"], "medium": ["en_US", "de_DE", "plain"], "hard": list(_LOCALES)},
            {},
        )

    @pytest.mark.parametrize("level", ["easy", "medium", "hard"])
    def test_number_family(self, number_config: dict[str, Any], level: str) -> None:
        values = self._VALID + self._BAD
        sources = _sources(values)
        bad_ids = {f"s1-{i}" for i in range(len(self._VALID), len(values))}
        for seed in range(4):
            out, prov_df, skipped_df = apply_knob_05("products", level, sources, number_config, seed=seed)
            # Guarded cells: unchanged, no provenance, one unparseable row each.
            for i in range(len(self._VALID), len(values)):
                assert out["s1"]["v"].iloc[i] == values[i]
            assert not set(prov_df["entity_id"]) & bad_ids
            assert sorted(skipped_df["entity_id"]) == sorted(bad_ids)
            assert set(skipped_df["reason"]) == {SKIP_UNPARSEABLE}
            assert (skipped_df["reason"] == SKIP_ROUNDTRIP).sum() == 0
            # Valid cells: every rewrite preserves the value exactly.
            for _, row in prov_df.iterrows():
                locale = json.loads(row["transform_params"])["to_locale"]
                assert _parse_number_in_locale(row["new_value"], locale) == _parse_number(
                    row["original_value"]
                ), (row["original_value"], row["new_value"], locale)

    def test_money_family_one_skip_row_no_partial_provenance(self) -> None:
        """A money cell with magnitude + currency rotation: the guard fires
        before any step, so there is one skip row and no reconvert_* row."""
        config = _base_config(
            "money",
            {"easy": ["de_DE"], "medium": ["de_DE"], "hard": ["de_DE"]},
            {"money": {"currencies": ["EUR"], "magnitude": ["millions"]}},
        )
        config["source_magnitude_context"] = {
            "s1": {"columns": ["v"], "implicit_currency": "USD", "implicit_magnitude": "raw"}
        }
        sources = _sources(["1500000000", "€385,00"])
        out, prov_df, skipped_df = apply_knob_05("products", "medium", sources, config, seed=0)
        assert out["s1"]["v"].iloc[1] == "€385,00"
        assert "s1-1" not in set(prov_df["entity_id"])
        assert len(skipped_df) == 1
        row = skipped_df.iloc[0]
        assert (row["entity_id"], row["reason"], row["attempted_format"]) == (
            "s1-1",
            SKIP_UNPARSEABLE,
            "locale:de_DE",
        )
        # The valid cell went through all three steps.
        assert set(prov_df.loc[prov_df["entity_id"] == "s1-0", "transform_fn"]) == {
            "reconvert_unit",
            "reconvert_currency",
            "reformat_number",
        }

    @pytest.mark.parametrize(
        "units,fmt,expected_attempt",
        [(["MB"], "en_US", "file_size:GB->MB"), (["GB"], "en_US", "locale:en_US")],
    )
    def test_file_size_family(self, units: list[str], fmt: str, expected_attempt: str) -> None:
        config = _base_config("file_size", {lv: [fmt] for lv in ("easy", "medium", "hard")}, {"file_size": {"units": units}})
        config["source_magnitude_context"] = {"s1": {"columns": {"v": {"implicit_unit": "GB"}}}}
        sources = _sources(["8", "6,25"])
        out, prov_df, skipped_df = apply_knob_05("products", "medium", sources, config, seed=0)
        assert out["s1"]["v"].iloc[1] == "6,25"
        assert "s1-1" not in set(prov_df["entity_id"])
        assert list(skipped_df["reason"]) == [SKIP_UNPARSEABLE]
        assert list(skipped_df["attempted_format"]) == [expected_attempt]
        assert out["s1"]["v"].iloc[0] in ("8,000 MB", "8 GB")

    def test_file_size_bare_same_unit_untouched(self) -> None:
        """``bare`` without a unit change reads nothing, so nothing is logged."""
        config = _base_config("file_size", {lv: ["bare"] for lv in ("easy", "medium", "hard")}, {"file_size": {"units": ["GB"]}})
        config["source_magnitude_context"] = {"s1": {"columns": {"v": {"implicit_unit": "GB"}}}}
        out, prov_df, skipped_df = apply_knob_05("products", "medium", _sources(["8", "6,25"]), config, seed=0)
        assert list(out["s1"]["v"]) == ["8", "6,25"]
        assert len(prov_df) == 0 and len(skipped_df) == 0
