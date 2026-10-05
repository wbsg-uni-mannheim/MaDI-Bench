"""Knob 05: locale-aware round-trip verification of number rewrites.

``reformat_number`` re-parses every string it writes to verify the rewrite.
The re-parse used to be the locale-agnostic heuristic ``_parse_number``
(plain Decimal, then en_US, then de_DE, then fr_FR), which misreads de_DE /
fr_FR output below 1,000,000: "2.000" -> 2.0, "2.000,0" -> 2.0,
"1.234,56" -> 1.23456, "500,0" -> 5000. Those cells were skipped as
``roundtrip_parse_fail`` (products medium/hard: ~3,000 cells per level). The
round trip now parses in the locale just written
(``_parse_number_in_locale``, separators from
``config/knob_05_format/_tables/number_locales.yaml``) and falls back to the
heuristic only when that fails. ``_parse_number`` itself (input parsing) is
unchanged. The date round trip had the same defect for day-first patterns
(``eu_dot`` "05.06.1985" read back month-first as 6 May) and now re-parses
day-first when the written pattern puts the day first.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from usecases_synthetic.lib.format_operators import (
    _parse_number,
    _parse_number_in_locale,
    _roundtrip_parse_date,
    _roundtrip_parse_number,
    reconvert_currency,
    reconvert_unit,
    reformat_date,
    reformat_number,
    reformat_number_suffix,
)
from usecases_synthetic.lib.rate_tables import get_date_format
from usecases_synthetic.scripts.apply_knob_05_format import (
    SKIP_ROUNDTRIP,
    apply_knob_05,
)

# (input, {locale: expected written string}). Covers integers and floats
# below 1,000, between 1,000 and 999,999, at or above 1,000,000, negative
# values, many decimals and scientific input.
_CASES: list[tuple[str, dict[str, str]]] = [
    ("5", {"en_US": "5", "de_DE": "5", "fr_FR": "5", "plain": "5"}),
    ("500", {"en_US": "500", "de_DE": "500", "fr_FR": "500", "plain": "500"}),
    ("500.0", {"en_US": "500.0", "de_DE": "500,0", "fr_FR": "500,0", "plain": "500.0"}),
    ("999.99", {"en_US": "999.99", "de_DE": "999,99", "fr_FR": "999,99", "plain": "999.99"}),
    ("0.5", {"en_US": "0.5", "de_DE": "0,5", "fr_FR": "0,5", "plain": "0.5"}),
    ("2000", {"en_US": "2,000", "de_DE": "2.000", "fr_FR": "2 000", "plain": "2000"}),
    ("2000.0", {"en_US": "2,000.0", "de_DE": "2.000,0", "fr_FR": "2 000,0", "plain": "2000.0"}),
    ("1234.56", {"en_US": "1,234.56", "de_DE": "1.234,56", "fr_FR": "1 234,56", "plain": "1234.56"}),
    ("999999", {"en_US": "999,999", "de_DE": "999.999", "fr_FR": "999 999", "plain": "999999"}),
    (
        "123456.789",
        {"en_US": "123,456.789", "de_DE": "123.456,789", "fr_FR": "123 456,789", "plain": "123456.789"},
    ),
    ("1000000", {"en_US": "1,000,000", "de_DE": "1.000.000", "fr_FR": "1 000 000", "plain": "1000000"}),
    (
        "1234567.891",
        {"en_US": "1,234,567.891", "de_DE": "1.234.567,891", "fr_FR": "1 234 567,891", "plain": "1234567.891"},
    ),
    (
        "240560000000",
        {
            "en_US": "240,560,000,000",
            "de_DE": "240.560.000.000",
            "fr_FR": "240 560 000 000",
            "plain": "240560000000",
        },
    ),
    ("-5", {"en_US": "-5", "de_DE": "-5", "fr_FR": "-5", "plain": "-5"}),
    ("-0.25", {"en_US": "-0.25", "de_DE": "-0,25", "fr_FR": "-0,25", "plain": "-0.25"}),
    ("-2000", {"en_US": "-2,000", "de_DE": "-2.000", "fr_FR": "-2 000", "plain": "-2000"}),
    ("-1234.5", {"en_US": "-1,234.5", "de_DE": "-1.234,5", "fr_FR": "-1 234,5", "plain": "-1234.5"}),
    (
        "-1234567.5",
        {"en_US": "-1,234,567.5", "de_DE": "-1.234.567,5", "fr_FR": "-1 234 567,5", "plain": "-1234567.5"},
    ),
    (
        "3.14159265358979",
        {
            "en_US": "3.14159265358979",
            "de_DE": "3,14159265358979",
            "fr_FR": "3,14159265358979",
            "plain": "3.14159265358979",
        },
    ),
    (
        "4321.123456789",
        {
            "en_US": "4,321.123456789",
            "de_DE": "4.321,123456789",
            "fr_FR": "4 321,123456789",
            "plain": "4321.123456789",
        },
    ),
    # Scientific input (the products sources carry e.g. "2.1492E2", "1.75E3").
    ("2.1492E2", {"en_US": "214.92", "de_DE": "214,92", "fr_FR": "214,92", "plain": "214.92"}),
    ("1.75E3", {"en_US": "1,750", "de_DE": "1.750", "fr_FR": "1 750", "plain": "1750"}),
    ("1.5E6", {"en_US": "1,500,000", "de_DE": "1.500.000", "fr_FR": "1 500 000", "plain": "1500000"}),
]

_LOCALES = ("en_US", "de_DE", "fr_FR", "plain")

_PARAMS = [
    pytest.param(value, locale, expected[locale], id=f"{value}-{locale}")
    for value, expected in _CASES
    for locale in _LOCALES
]


class TestReformatNumberRoundtrip:
    """reformat_number accepts every locale rewrite and it round-trips."""

    @pytest.mark.parametrize("value,locale,expected", _PARAMS)
    def test_rewrite_accepted_and_exact(
        self, value: str, locale: str, expected: str
    ) -> None:
        result = reformat_number(value, locale)
        assert result is not None, f"{value!r} -> {locale} rejected"
        new_value, params = result
        assert new_value == expected
        assert params["to_locale"] == locale
        # Parsed in the written locale, the string is exactly the input value.
        assert _parse_number_in_locale(new_value, locale) == Decimal(value)

    def test_single_group_de_integer_is_2000(self) -> None:
        """de_DE "2.000" is 2000 once the written locale is known."""
        result = reformat_number("2000", "de_DE")
        assert result is not None
        assert result[0] == "2.000"
        assert _parse_number_in_locale("2.000", "de_DE") == Decimal("2000")
        assert _roundtrip_parse_number("2.000", "de_DE") == Decimal("2000")

    @pytest.mark.parametrize(
        "value", ["2000", "2000.0", "1234.56", "500.0", "999999", "-1234.5"]
    )
    def test_heuristic_alone_would_misread(self, value: str) -> None:
        """Regression guard: these de_DE rewrites were rejected before the fix
        because the heuristic misreads them; they must now be accepted."""
        new_value, _ = reformat_number(value, "de_DE")
        heuristic = _parse_number(new_value)
        assert heuristic is None or heuristic != Decimal(value)
        assert _parse_number_in_locale(new_value, "de_DE") == Decimal(value)


class TestParseNumberInLocale:
    """Strict locale-aware parser."""

    @pytest.mark.parametrize(
        "value,locale,expected",
        [
            ("2.000", "de_DE", "2000"),
            ("2.000", "en_US", "2.000"),
            ("2,000", "en_US", "2000"),
            ("2,000", "de_DE", "2.000"),
            ("500,0", "fr_FR", "500.0"),
            ("1 234 567,5", "fr_FR", "1234567.5"),
            ("-1.234,5", "de_DE", "-1234.5"),
            ("+12", "en_US", "12"),
            ("1234567.5", "plain", "1234567.5"),
            ("  1.000  ", "de_DE", "1000"),
        ],
    )
    def test_parses(self, value: str, locale: str, expected: str) -> None:
        assert _parse_number_in_locale(value, locale) == Decimal(expected)

    @pytest.mark.parametrize(
        "value,locale",
        [
            ("1.2345", "de_DE"),  # malformed group
            ("1,234.5", "de_DE"),  # en_US shape
            ("1.234,5", "en_US"),  # de_DE shape
            ("1 234.5", "fr_FR"),  # dot decimal in fr_FR
            ("1,234", "plain"),  # separators in plain
            ("12,34,567", "en_US"),  # non-3 grouping
            ("2.1492E2", "plain"),  # scientific is not a written form
            ("SAR 219", "en_US"),
            ("", "en_US"),
            ("1.000", "xx_XX"),  # unknown locale
        ],
    )
    def test_rejects(self, value: str, locale: str) -> None:
        assert _parse_number_in_locale(value, locale) is None

    def test_roundtrip_parse_falls_back_to_heuristic(self) -> None:
        """A string not in the written locale's shape falls back to
        _parse_number."""
        assert _parse_number_in_locale("1234", "en_US") is None
        assert _roundtrip_parse_number("1234", "en_US") == Decimal("1234")

    def test_parse_number_semantics_unchanged(self) -> None:
        """Input parsing keeps the locale-agnostic heuristic (other callers
        rely on it); only the round trip is locale-aware."""
        assert _parse_number("2.000") == Decimal("2")
        assert _parse_number("500,0") == Decimal("5000")
        assert _parse_number("1.234,56") == Decimal("1.23456")
        assert _parse_number("1.234.567,5") == Decimal("1234567.5")
        assert _parse_number("1,234.5") == Decimal("1234.5")


class TestOtherOperatorsRoundtrip:
    """The other number writers emit en_US / plain; their locale-aware round
    trip must keep accepting the same rewrites."""

    @pytest.mark.parametrize(
        "value,suffix,expected",
        [
            ("1500000000", "B", "1.50B"),
            ("2000000000000", "K", "2,000,000,000K"),
            ("3000000", "M", "3M"),
            ("-2500", "K", "-2.50K"),
            ("1234000000", "K", "1,234,000K"),
        ],
    )
    def test_suffix(self, value: str, suffix: str, expected: str) -> None:
        result = reformat_number_suffix(value, suffix)
        assert result is not None
        assert result[0] == expected

    def test_reconvert_unit_plain(self) -> None:
        assert reconvert_unit("8", "file_size", "GB", "MB")[0] == "8000"
        assert reconvert_unit("65170000000", "magnitude", "raw", "billions")[0] == "65.17"
        assert reconvert_unit("1500000", "magnitude", "raw", "millions")[0] == "1.50"
        # The tolerance logic is unchanged: 2-decimal rounding that loses
        # more than 1e-4 relative is still rejected.
        assert reconvert_unit("1234567", "magnitude", "raw", "millions") is None

    def test_reconvert_currency_plain(self) -> None:
        assert reconvert_currency("1000.00", "USD", "EUR")[0] == "921.50"


class TestApplyKnob05NumberFamily:
    """End to end: a number-family column below 1e6 at medium / hard has no
    round-trip skips and every rewrite parses back in its written locale."""

    @pytest.fixture
    def config(self) -> dict[str, Any]:
        return {
            "domain": "products",
            "attribute_classes": {"s1": {"cap": "number", "price": "number"}},
            "id_columns": {"s1": "id"},
            "format_pools_per_level": {
                "easy": {"number": ["en_US", "plain"]},
                "medium": {"number": ["en_US", "de_DE", "plain"]},
                "hard": {"number": ["en_US", "de_DE", "fr_FR", "plain"]},
            },
            "within_source_consistency": {"easy": "source", "medium": "source", "hard": "row"},
            "unit_pool_per_level": {"easy": {}, "medium": {}, "hard": {}},
            "source_magnitude_context": {},
        }

    @pytest.fixture
    def sources(self) -> dict[str, pd.DataFrame]:
        cap = ["2000.0", "500.0", "64", "1000", "3840.0", "2.1492E2", "1.75E3", "999999", "1000000", "-12.5"]
        price = ["219.99", "1299", "59.9", "1234.56", "0.99", "25349", "4900.0", "12", "150000.5", "7"]
        df = pd.DataFrame({"id": [f"s1-{i}" for i in range(len(cap))], "cap": cap, "price": price})
        df.attrs["dataset_name"] = "s1"
        return {"s1": df}

    @pytest.mark.parametrize("level", ["medium", "hard"])
    def test_no_roundtrip_skips(
        self, config: dict[str, Any], sources: dict[str, pd.DataFrame], level: str
    ) -> None:
        for seed in range(8):
            _, prov_df, skipped_df = apply_knob_05(
                "products", level, sources, config, seed=seed
            )
            assert (skipped_df["reason"] == SKIP_ROUNDTRIP).sum() == 0, skipped_df
            for _, row in prov_df.iterrows():
                assert row["transform_fn"] == "reformat_number"
                locale = json.loads(row["transform_params"])["to_locale"]
                assert _parse_number_in_locale(row["new_value"], locale) == _parse_number(
                    row["original_value"]
                ), (row["original_value"], row["new_value"], locale)

    def test_hard_emits_de_and_fr_below_one_million(
        self, config: dict[str, Any], sources: dict[str, pd.DataFrame]
    ) -> None:
        locales: set[str] = set()
        for seed in range(8):
            _, prov_df, _ = apply_knob_05("products", "hard", sources, config, seed=seed)
            for _, row in prov_df.iterrows():
                if abs(_parse_number(row["original_value"])) < 1_000_000:
                    locales.add(json.loads(row["transform_params"])["to_locale"])
        assert {"de_DE", "fr_FR"} <= locales


class TestReformatDateDayFirstRoundtrip:
    """reformat_date re-parses day-first output (eu_dot, two_digit_year_dot)
    day-first; the month-first heuristic swapped day and month for day <= 12
    and rejected the rewrite."""

    @pytest.mark.parametrize(
        "value,fmt,expected",
        [
            ("1985-06-05", "eu_dot", "05.06.1985"),
            ("2010-03-01", "eu_dot", "01.03.2010"),
            ("1985-06-15", "eu_dot", "15.06.1985"),
            ("1985-06-06", "eu_dot", "06.06.1985"),
            ("1985-06-05", "two_digit_year_dot", "05.06.85"),
            ("1985-06-05", "us_slash", "06/05/1985"),
            ("1985-06-05", "two_digit_year_us", "06/05/85"),
            ("1985-06-05", "long_english", "June 05, 1985"),
            ("1985-06-05", "compact", "19850605"),
            ("1985-06-05", "iso", "1985-06-05"),
        ],
    )
    def test_accepted(self, value: str, fmt: str, expected: str) -> None:
        result = reformat_date(value, fmt)
        assert result is not None, f"{value} -> {fmt} rejected"
        assert result[0] == expected
        assert _roundtrip_parse_date(result[0], get_date_format(fmt)["pattern"]) == date.fromisoformat(value)

    def test_century_loss_still_rejected(self) -> None:
        """Two-digit years keep dateutil's century window: 1908 -> "08"
        reads back as 2008 and is rejected, as before."""
        assert reformat_date("1908-01-01", "two_digit_year_dot") is None
        assert reformat_date("1908-01-01", "two_digit_year_us") is None
