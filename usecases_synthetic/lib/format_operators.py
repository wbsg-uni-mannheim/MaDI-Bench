"""Format and unit transformation operators for Knob 05.

Each operator returns ``(new_value, params_dict)`` on success or
``None`` on round-trip verification failure (caller falls back to
identity). The number operators also return ``None`` when the input fails
the value-preservation guard :func:`_parse_input_number`. All operators are
pure functions with no side effects.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import dateutil.parser as du_parser

from .rate_tables import (
    get_date_format,
    get_fx_rate,
    get_fx_rate_date,
    get_locale_config,
    get_suffix_scales,
    get_unit_factor,
)

# Floating-point tolerance for round-trip verification on unit conversions.
_FP_TOLERANCE = 1e-6

# Relative tolerance for large-magnitude financial values.
_REL_TOLERANCE = 1e-4


# ---- Date operators ---------------------------------------------------------


def _parse_date_flexible(value: str) -> date | None:
    """Try to parse a date string into a ``datetime.date``.

    Handles ISO, US, EU, long-form, compact, and year-only formats.
    Returns ``None`` on parse failure.
    """
    value = value.strip()
    if not value or value.lower() in ("null", "nan", "none", ""):
        return None

    # Year-only (e.g. "1908", "2023")
    if re.fullmatch(r"\d{4}", value):
        return date(int(value), 1, 1)

    # Year-month (e.g. "2024-03")
    if re.fullmatch(r"\d{4}-\d{2}", value):
        parts = value.split("-")
        return date(int(parts[0]), int(parts[1]), 1)

    # Compact (YYYYMMDD)
    if re.fullmatch(r"\d{8}", value):
        return datetime.strptime(value, "%Y%m%d").date()

    try:
        return du_parser.parse(value, dayfirst=False).date()
    except (ValueError, OverflowError):
        pass

    # Try dayfirst=True for EU formats
    try:
        return du_parser.parse(value, dayfirst=True).date()
    except (ValueError, OverflowError):
        return None


def _roundtrip_parse_date(value: str, pattern: str) -> date | None:
    """Re-parse a date string this module just wrote with strftime *pattern*.

    :func:`_parse_date_flexible` reads month-first (dateutil
    ``dayfirst=False``) and tries day-first only when that raises, so a
    day-first string written here comes back with day and month swapped
    whenever the day is <= 12 ("05.06.1985" from ``%d.%m.%Y`` reads as
    6 May). When the written pattern puts the day before the month the
    string is parsed day-first; everything else, and any failure, falls back
    to :func:`_parse_date_flexible`. Two-digit years keep dateutil's century
    window either way.
    """
    d_pos, m_pos = pattern.find("%d"), pattern.find("%m")
    if 0 <= d_pos < m_pos:
        try:
            return du_parser.parse(value, dayfirst=True).date()
        except (ValueError, OverflowError):
            pass
    return _parse_date_flexible(value)


def reformat_date(
    value: str,
    to_format_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """Reformat a date string to a different format.

    Parameters
    ----------
    value : str
        Original date string.
    to_format_id : str
        Target format ID from ``date_formats.yaml``.

    Returns
    -------
    tuple of (str, dict) or None
        ``(reformatted_value, params)`` on success, ``None`` on failure.
        ``params`` has keys ``from_format``, ``to_format``, ``direction``.
    """
    parsed = _parse_date_flexible(value)
    if parsed is None:
        return None

    fmt_spec = get_date_format(to_format_id)
    pattern = fmt_spec["pattern"]

    # For precision-downgrade formats, we lose information.
    # Year-only: only keep year.
    if to_format_id == "precision_year":
        new_value = str(parsed.year)
    elif to_format_id == "precision_year_month":
        new_value = f"{parsed.year:04d}-{parsed.month:02d}"
    else:
        try:
            new_value = parsed.strftime(pattern)
        except ValueError:
            return None

    # Round-trip verification: parse back in the day/month order just
    # written and compare.
    roundtrip = _roundtrip_parse_date(new_value, pattern)
    if roundtrip is None:
        return None

    # For precision downgrades, verify at the emitted precision level.
    if to_format_id == "precision_year":
        if roundtrip.year != parsed.year:
            return None
    elif to_format_id == "precision_year_month":
        if roundtrip.year != parsed.year or roundtrip.month != parsed.month:
            return None
    else:
        if roundtrip != parsed:
            return None

    # Determine direction.
    direction = "identity" if new_value.strip() == value.strip() else "up"

    params = {
        "from_format": "detected",
        "to_format": to_format_id,
        "direction": direction,
    }
    return new_value, params


# ---- Number operators -------------------------------------------------------


def _parse_number(value: str) -> Decimal | None:
    """Parse a number string (possibly with locale separators or suffixes).

    Handles: plain, en_US (1,234.56), de_DE (1.234,56),
    fr_FR (1 234,56), K/M/B suffixes.
    """
    if not isinstance(value, str):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None

    value = value.strip()
    if not value or value.lower() in ("null", "nan", "none", ""):
        return None

    # Strip currency symbols/codes.
    value = re.sub(r"^[£€$¥₹]+\s*", "", value)
    value = re.sub(r"\s*[£€$¥₹]+$", "", value)
    value = re.sub(r"^(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY)\s*", "", value)
    value = re.sub(r"\s*(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY)$", "", value)
    value = value.strip()

    # Check for suffix notation (e.g. "1.5B", "200M", "3.2K").
    suffix_match = re.fullmatch(
        r"([+-]?\d[\d\s.,]*\d?)\s*([KMBT])\b", value, re.IGNORECASE
    )
    if suffix_match:
        num_str = suffix_match.group(1)
        suffix = suffix_match.group(2).upper()
        scales = get_suffix_scales()
        if suffix in scales:
            base = _parse_number_bare(num_str)
            if base is not None:
                return base * Decimal(str(scales[suffix]))

    return _parse_number_bare(value)


def _parse_number_bare(value: str) -> Decimal | None:
    """Parse a bare number without suffix, handling locale separators."""
    value = value.strip()
    if not value:
        return None

    # Try plain parse first.
    try:
        return Decimal(value)
    except InvalidOperation:
        pass

    # en_US: commas as thousands, dot as decimal.
    cleaned = value.replace(",", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        pass

    # de_DE: dots as thousands, comma as decimal.
    # Heuristic: if there's a comma and dots before it, it's de_DE.
    if "," in value:
        cleaned = value.replace(".", "").replace(",", ".")
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            pass

    # fr_FR: spaces as thousands, comma as decimal.
    if " " in value:
        cleaned = value.replace(" ", "").replace(",", ".")
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            pass

    return None


def _parse_number_in_locale(value: str, locale_id: str) -> Decimal | None:
    """Parse a number string written in a KNOWN locale (strict).

    Round-trip verification counterpart of :func:`_format_number_with_locale`.
    The writer knows the locale it just emitted, so the separators are
    unambiguous: ``"2.000"`` in ``de_DE`` is 2000, ``"500,0"`` in ``fr_FR``
    is 500.0. The locale-agnostic :func:`_parse_number` cannot know this and
    misreads those strings (it tries plain, then en_US, then de_DE).

    Accepted shape, with the separators of *locale_id* from
    ``number_locales.yaml``: an optional sign, an integer part grouped in
    blocks of ``grouping`` digits joined by ``thousands_sep`` (bare digits
    when the locale has no grouping), and an optional ``decimal_sep``
    followed by digits.

    Parameters
    ----------
    value : str
        The formatted number string.
    locale_id : str
        Locale ID from ``number_locales.yaml`` the string was written in.

    Returns
    -------
    Decimal or None
        ``None`` when the string does not have that locale's shape (or the
        locale is unknown); callers then fall back to :func:`_parse_number`.
    """
    if not isinstance(value, str):
        return None
    try:
        cfg = get_locale_config(locale_id)
    except KeyError:
        return None
    decimal_sep: str = cfg["decimal_sep"]
    thousands_sep: str = cfg["thousands_sep"]
    grouping: int = cfg["grouping"]

    if grouping and thousands_sep:
        int_re = (
            rf"\d{{1,{grouping}}}(?:{re.escape(thousands_sep)}\d{{{grouping}}})*"
        )
    else:
        int_re = r"\d+"
    m = re.fullmatch(
        rf"([+-]?)({int_re})(?:{re.escape(decimal_sep)}(\d+))?", value.strip()
    )
    if m is None:
        return None
    sign, int_part, frac_part = m.groups()
    if thousands_sep:
        int_part = int_part.replace(thousands_sep, "")
    canonical = f"{sign}{int_part}" + (f".{frac_part}" if frac_part else "")
    try:
        return Decimal(canonical)
    except InvalidOperation:
        return None


def _roundtrip_parse_number(value: str, locale_id: str) -> Decimal | None:
    """Re-parse a string this module just wrote in *locale_id*.

    Tries the strict locale-aware :func:`_parse_number_in_locale` first and
    falls back to the heuristic :func:`_parse_number` only if that fails.
    """
    parsed = _parse_number_in_locale(value, locale_id)
    if parsed is not None:
        return parsed
    return _parse_number(value)


# ---- Input guard ------------------------------------------------------------

# Locales whose strict shapes count as legitimate readings of an INPUT cell.
_STRICT_INPUT_LOCALES: tuple[str, ...] = ("en_US", "de_DE", "fr_FR", "plain")

# Suffix notation exactly as :func:`_parse_number` recognises it.
_SUFFIX_RE = re.compile(r"([+-]?\d[\d\s.,]*\d?)\s*([KMBT])\b", re.IGNORECASE)


def _strip_currency_affixes(value: str) -> str:
    """Strip the currency symbols / codes :func:`_parse_number` strips.

    Same four patterns in the same order (kept separate so that
    ``_parse_number`` itself stays unchanged).
    """
    value = value.strip()
    value = re.sub(r"^[£€$¥₹]+\s*", "", value)
    value = re.sub(r"\s*[£€$¥₹]+$", "", value)
    value = re.sub(r"^(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY)\s*", "", value)
    value = re.sub(r"\s*(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY)$", "", value)
    return value.strip()


def _strict_bare_readings(value: str) -> set[Decimal]:
    """Strict readings of a bare number string (no currency, no suffix).

    The plain Decimal literal (finite values only; includes scientific
    notation such as ``"2.1492E2"``, which ``_parse_number`` also accepts
    first) plus the strict :func:`_parse_number_in_locale` reading in each
    of ``_STRICT_INPUT_LOCALES``.
    """
    value = value.strip()
    readings: set[Decimal] = set()
    try:
        plain = Decimal(value)
    except (InvalidOperation, ValueError):
        plain = None
    if plain is not None and plain.is_finite():
        readings.add(plain)
    for locale_id in _STRICT_INPUT_LOCALES:
        parsed = _parse_number_in_locale(value, locale_id)
        if parsed is not None:
            readings.add(parsed)
    return readings


def _strict_number_readings(value: str) -> set[Decimal]:
    """Every value *value* denotes under a strict locale reading.

    After stripping the currency symbols / codes the heuristic strips:
    the readings of :func:`_strict_bare_readings`, and for K/M/B/T suffix
    notation (as :func:`_parse_number` accepts it) the strict readings of
    the number part times the suffix scale. ``"1,234"`` has two readings
    (1234 en_US, 1.234 de_DE / fr_FR); ``"€385,00"`` has one (385.00);
    ``"SAR 219"`` has none.
    """
    if not isinstance(value, str):
        value = str(value)
    bare = _strip_currency_affixes(value)
    readings = _strict_bare_readings(bare)
    suffix_match = _SUFFIX_RE.fullmatch(bare)
    if suffix_match:
        scales = get_suffix_scales()
        suffix = suffix_match.group(2).upper()
        if suffix in scales:
            scale = Decimal(str(scales[suffix]))
            readings |= {r * scale for r in _strict_bare_readings(suffix_match.group(1))}
    return readings


def _parse_input_number(value: str) -> Decimal | None:
    """Parse the INPUT of a K5 number rewrite; ``None`` if it is not safe.

    Knob 5 must be value-preserving. The heuristic :func:`_parse_number`
    (kept unchanged; other callers rely on it) reads a decimal comma as an
    en_US thousands separator: ``"6,25"`` -> 625, ``"€385,00"`` -> 38500,
    ``"€1.229,00"`` -> 1.229, so a rewrite of such a cell would change its
    value while being logged as a format-only change. The heuristic value is
    therefore accepted only if it equals one of the strict readings of the
    input (:func:`_strict_number_readings`). Ambiguous inputs keep the
    heuristic's choice when it is one of their strict readings (``"1,234"``
    -> 1234, the en_US reading). Returns ``None`` when the heuristic fails,
    gives a non-finite value, or matches no strict reading; callers leave
    the cell unchanged (the K5 dispatcher logs it as ``unparseable_value``).
    """
    if not isinstance(value, str):
        value = str(value)
    parsed = _parse_number(value)
    if parsed is None or not parsed.is_finite():
        return None
    if parsed not in _strict_number_readings(value):
        return None
    return parsed


def _format_number_with_locale(
    value: Decimal,
    locale_id: str,
    precision: int | None = None,
) -> str:
    """Format a Decimal with the given locale's separator conventions.

    Parameters
    ----------
    value : Decimal
        The number to format.
    locale_id : str
        Locale ID from ``number_locales.yaml``.
    precision : int or None
        Decimal places. None = auto (use the value's natural precision).

    Returns
    -------
    str
        Formatted number string.
    """
    cfg = get_locale_config(locale_id)
    decimal_sep: str = cfg["decimal_sep"]
    thousands_sep: str = cfg["thousands_sep"]
    grouping: int = cfg["grouping"]

    # Determine precision.
    if precision is None:
        # Use the natural precision of the Decimal.
        sign, digits, exponent = value.as_tuple()
        if exponent < 0:
            precision = -exponent
        else:
            precision = 0

    # Format with Python's string formatting (always uses . as decimal).
    raw = f"{value:,.{precision}f}" if grouping else f"{value:.{precision}f}"

    # Replace Python's default separators (comma=thousands, dot=decimal)
    # with the locale's separators.
    if grouping and thousands_sep != ",":
        # First protect the decimal point.
        raw = raw.replace(".", "\x00")
        raw = raw.replace(",", thousands_sep)
        raw = raw.replace("\x00", decimal_sep)
    elif decimal_sep != ".":
        # No thousands grouping or already correct.
        raw = raw.replace(",", "")
        raw = raw.replace(".", decimal_sep)
    elif not grouping:
        raw = raw.replace(",", "")

    return raw


def reformat_number(
    value: str,
    to_locale_id: str,
    precision: int | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Reformat a number string to a different locale convention.

    Parameters
    ----------
    value : str
        Original number string.
    to_locale_id : str
        Target locale ID from ``number_locales.yaml``.
    precision : int or None
        Decimal places (None = preserve original).

    Returns
    -------
    tuple of (str, dict) or None
        ``(reformatted_value, params)`` or ``None`` on failure.
    """
    # Guarded input parse: a heuristic misreading (decimal comma read as a
    # thousands separator) is rejected, see _parse_input_number.
    parsed = _parse_input_number(str(value))
    if parsed is None:
        return None

    new_value = _format_number_with_locale(parsed, to_locale_id, precision)

    # Round-trip verification in the locale just written (the heuristic
    # _parse_number misreads e.g. de_DE "2.000" as 2.0).
    roundtrip = _roundtrip_parse_number(new_value, to_locale_id)
    if roundtrip is None:
        return None
    if abs(roundtrip - parsed) > Decimal(str(_FP_TOLERANCE)):
        # Try relative tolerance for large values.
        if parsed != 0 and abs(roundtrip - parsed) / abs(parsed) > Decimal(
            str(_REL_TOLERANCE)
        ):
            return None

    params = {
        "from_locale": "detected",
        "to_locale": to_locale_id,
        "precision": precision,
    }
    return new_value, params


def reformat_number_suffix(
    value: str,
    suffix: str,
) -> tuple[str, dict[str, Any]] | None:
    """Reformat a number using K/M/B suffix notation (hard-only).

    Parameters
    ----------
    value : str
        Original number string.
    suffix : str
        Target suffix (``"K"``, ``"M"``, ``"B"``, ``"T"``).

    Returns
    -------
    tuple of (str, dict) or None
        ``(reformatted_value, params)`` or ``None`` on failure.
    """
    parsed = _parse_input_number(str(value))  # guarded, see reformat_number
    if parsed is None:
        return None

    scales = get_suffix_scales()
    if suffix not in scales:
        return None

    scale = Decimal(str(scales[suffix]))
    scaled = parsed / scale

    # Format with reasonable precision.
    if scaled == scaled.to_integral_value():
        new_value = f"{scaled:,.0f}{suffix}"
        written_locale = "en_US"  # comma thousands grouping
    else:
        new_value = f"{float(scaled):.2f}{suffix}"
        written_locale = "plain"

    # Round-trip verification in the locale just written (number part),
    # heuristic fallback on the whole suffixed string.
    base = _parse_number_in_locale(new_value[: -len(suffix)], written_locale)
    roundtrip = base * scale if base is not None else _parse_number(new_value)
    if roundtrip is None:
        return None
    diff = abs(roundtrip - parsed)
    if parsed != 0 and diff / abs(parsed) > Decimal(str(_REL_TOLERANCE)):
        return None

    params = {
        "from_locale": "detected",
        "to_locale": f"suffix_{suffix}",
        "precision": None,
        "suffix": suffix,
    }
    return new_value, params


# ---- Unit conversion operators ----------------------------------------------


def reconvert_unit(
    value: str,
    group: str,
    from_unit: str,
    to_unit: str,
) -> tuple[str, dict[str, Any]] | None:
    """Convert a numeric value between units in the same group.

    Parameters
    ----------
    value : str
        Original numeric string.
    group : str
        Unit group (e.g. ``"magnitude"``, ``"duration"``).
    from_unit : str
        Source unit.
    to_unit : str
        Target unit.

    Returns
    -------
    tuple of (str, dict) or None
        ``(converted_value, params)`` or ``None`` on failure.
    """
    parsed = _parse_input_number(str(value))  # guarded, see reformat_number
    if parsed is None:
        return None

    factor = Decimal(str(get_unit_factor(group, from_unit, to_unit)))
    converted = parsed * factor

    # Format: preserve reasonable precision.
    if converted == converted.to_integral_value():
        new_value = str(int(converted))
    else:
        # Up to 2 decimal places for readability.
        new_value = f"{float(converted):.2f}"

    # Round-trip verification: convert back and compare. ``new_value`` is
    # written without separators (the ``plain`` locale).
    roundtrip_parsed = _roundtrip_parse_number(new_value, "plain")
    if roundtrip_parsed is None:
        return None
    reverse_factor = Decimal(str(get_unit_factor(group, to_unit, from_unit)))
    roundtrip_original = roundtrip_parsed * reverse_factor

    if parsed != 0:
        rel_diff = abs(roundtrip_original - parsed) / abs(parsed)
        if rel_diff > Decimal(str(_REL_TOLERANCE)):
            return None
    elif roundtrip_original != 0:
        return None

    rate_date = ""
    if group == "magnitude":
        rate_date = ""
    # For FX conversions, the rate_date comes from the FX table.
    # The caller handles currency conversion separately.

    params = {
        "from_unit": from_unit,
        "to_unit": to_unit,
        "rate": float(factor),
        "rate_date": rate_date,
        "magnitude_scale": "",
    }
    return new_value, params


# ---- Duration operators -----------------------------------------------------

# Supported duration target formats (Knob 05 music duration column).
# All forms preserve the canonical seconds value under round-trip parse.
_DURATION_FORMATS = frozenset({"seconds_int", "mm_ss", "hh_mm_ss", "human_xm_ys"})

_RE_MM_SS = re.compile(r"^(\d{1,4}):([0-5]\d)$")
_RE_HH_MM_SS = re.compile(r"^(\d{1,4}):([0-5]\d):([0-5]\d)$")
_RE_HUMAN_XM_YS = re.compile(r"^(?:(\d+)h\s+)?(?:(\d+)m\s+)?(\d+)s$")


def parse_duration(value: str) -> int | None:
    """Parse a duration string into integer seconds.

    Accepts every form emitted by :func:`format_duration` (``seconds_int``,
    ``mm_ss``, ``hh_mm_ss``, ``human_xm_ys``) plus a bare integer or float
    of seconds. Returns ``None`` on parse failure.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("null", "nan", "none", ""):
        return None

    # Bare integer / float seconds.
    try:
        return int(round(float(s)))
    except (ValueError, TypeError):
        pass

    m = _RE_HH_MM_SS.match(s)
    if m is not None:
        h, mm, ss = (int(g) for g in m.groups())
        return h * 3600 + mm * 60 + ss

    m = _RE_MM_SS.match(s)
    if m is not None:
        mm, ss = (int(g) for g in m.groups())
        return mm * 60 + ss

    m = _RE_HUMAN_XM_YS.match(s)
    if m is not None:
        h_raw, mm_raw, ss_raw = m.groups()
        h = int(h_raw) if h_raw is not None else 0
        mm = int(mm_raw) if mm_raw is not None else 0
        ss = int(ss_raw)
        return h * 3600 + mm * 60 + ss

    return None


def format_duration(
    value: str,
    target_format: str,
) -> tuple[str, dict[str, Any]] | None:
    """Reformat a duration value into a target string form.

    Parameters
    ----------
    value : str
        Original duration expressed as bare seconds (int/float) or any of
        the supported string forms.
    target_format : str
        One of ``seconds_int``, ``mm_ss``, ``hh_mm_ss``, ``human_xm_ys``.

    Returns
    -------
    tuple of (str, dict) or None
        ``(new_value, params)`` on success — ``params`` carries
        ``from_unit``, ``to_unit``, ``rate=1.0``, ``rate_date=""``,
        ``magnitude_scale=""`` (matches the ``reconvert_unit`` schema so
        the dispatcher can emit a uniform ``transform_fn=reconvert_unit``
        provenance row). ``None`` on parse failure or round-trip failure.
    """
    if target_format not in _DURATION_FORMATS:
        return None

    seconds = parse_duration(value)
    if seconds is None or seconds < 0:
        return None

    if target_format == "seconds_int":
        new_value = str(seconds)
    elif target_format == "mm_ss":
        mm, ss = divmod(seconds, 60)
        new_value = f"{mm}:{ss:02d}"
    elif target_format == "hh_mm_ss":
        hh, rem = divmod(seconds, 3600)
        mm, ss = divmod(rem, 60)
        new_value = f"{hh}:{mm:02d}:{ss:02d}"
    else:  # human_xm_ys
        hh, rem = divmod(seconds, 3600)
        mm, ss = divmod(rem, 60)
        parts = []
        if hh:
            parts.append(f"{hh}h")
        if mm:
            parts.append(f"{mm}m")
        parts.append(f"{ss}s")
        new_value = " ".join(parts)

    # Round-trip verification: parse the emitted string back and compare
    # against the canonical seconds.
    parsed_back = parse_duration(new_value)
    if parsed_back != seconds:
        return None

    params = {
        "from_unit": "seconds",
        "to_unit": target_format,
        "rate": 1.0,
        "rate_date": "",
        "magnitude_scale": "",
    }
    return new_value, params


def reconvert_currency(
    value: str,
    from_ccy: str,
    to_ccy: str,
) -> tuple[str, dict[str, Any]] | None:
    """Convert a monetary value between currencies using the FX table.

    Parameters
    ----------
    value : str
        Original numeric string (without currency symbol).
    from_ccy : str
        Source currency ISO code.
    to_ccy : str
        Target currency ISO code.

    Returns
    -------
    tuple of (str, dict) or None
        ``(converted_value, params)`` or ``None`` on failure.
    """
    parsed = _parse_input_number(str(value))  # guarded, see reformat_number
    if parsed is None:
        return None

    rate = Decimal(str(get_fx_rate(from_ccy, to_ccy)))
    converted = parsed * rate

    # Format with 2 decimal places for currency.
    new_value = f"{float(converted):.2f}"

    # Round-trip verification (``new_value`` is in the ``plain`` locale).
    roundtrip = _roundtrip_parse_number(new_value, "plain")
    if roundtrip is None:
        return None
    reverse_rate = Decimal(str(get_fx_rate(to_ccy, from_ccy)))
    roundtrip_original = roundtrip * reverse_rate

    if parsed != 0:
        rel_diff = abs(roundtrip_original - parsed) / abs(parsed)
        if rel_diff > Decimal(str(_REL_TOLERANCE)):
            return None
    elif roundtrip_original != 0:
        return None

    params = {
        "from_unit": from_ccy,
        "to_unit": to_ccy,
        "rate": float(rate),
        "rate_date": get_fx_rate_date(),
        "magnitude_scale": "",
    }
    return new_value, params
