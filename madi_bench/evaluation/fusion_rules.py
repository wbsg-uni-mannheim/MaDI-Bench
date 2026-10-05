"""The MaDI-Bench fusion comparators and the per-domain rule tables.

One comparator per gold attribute decides whether a fused value is correct
against the gold value. The tables below are the STRICT rules the benchmark
reports: a comparator tolerates how a value is written
(rounding, units, casing, list serialization, a rating on a coarser scale),
never disagreement about the fact. Everything a system is expected to
repair in its normalization stage (annotation suffixes, partial dates,
typography, taxonomy mapping) is NOT forgiven here.

The five base comparators mirror PyDI's ``PyDI.fusion.evaluation`` functions
(``exact_match``, ``tokenized_match``, ``year_only_match``,
``numeric_tolerance_match``, ``set_equality_match``) so that the package
scores exactly as the evaluation code that produced the reference numbers;
the others are the comparators PyDI lacks. This module depends on pandas
only.
"""

from __future__ import annotations

import ast
import json
import re
import string
import unicodedata
from datetime import date, datetime
from typing import Any, Callable

import pandas as pd

Rule = tuple[str, Callable[..., bool], dict[str, Any]]

# --------------------------------------------------------------- missing

def is_missing(value: Any) -> bool:
    """PyDI's notion of a missing cell: NaN/None, an empty list or array. The
    empty string is NOT missing here (it reaches the comparator)."""
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.size == 0 or bool(np.all(pd.isna(value)))
    except Exception:  # noqa: BLE001
        pass
    return False


def _blank(value: Any) -> bool:
    """The stricter test the newer comparators use: also '', 'nan', 'none'."""
    try:
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        if isinstance(value, str) and value.strip().lower() in ("", "nan", "none"):
            return True
    except (TypeError, ValueError):
        return False
    return False


_LIST_SEP_RE = re.compile(r"[|;]")


def parse_list_literal(value: Any) -> Any:
    """A JSON / Python list literal becomes a list of strings, a '|' or ';'
    separated string with several parts becomes a list; anything else passes
    through. Array-typed attributes may be submitted in any of these forms."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
        if _LIST_SEP_RE.search(text):
            parts = [p.strip() for p in _LIST_SEP_RE.split(text) if p.strip()]
            if len(parts) > 1:
                return parts
    return value


# ------------------------------------------------ the five PyDI comparators

def exact_match(fused_value: Any, expected_value: Any) -> bool:
    return fused_value == expected_value


def _jaccard(a: set, b: set) -> float:
    """|a & b| / |a | b|; two empty sets are identical (as textdistance has it)."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def tokenized_match(fused_value: Any, expected_value: Any, threshold: float = 1.0) -> bool:
    """Jaccard >= threshold over token sets (lists: the elements; strings:
    lower-cased words with punctuation stripped)."""
    if is_missing(fused_value) and is_missing(expected_value):
        return True
    if is_missing(fused_value) or is_missing(expected_value):
        return False
    if isinstance(fused_value, list) and isinstance(expected_value, list):
        return _jaccard(set(fused_value), set(expected_value)) >= threshold
    if isinstance(fused_value, list) or isinstance(expected_value, list):
        fused_list = fused_value if isinstance(fused_value, list) else [str(fused_value)]
        expected_list = expected_value if isinstance(expected_value, list) else [str(expected_value)]
        return _jaccard(set(fused_list), set(expected_list)) >= threshold

    def clean_tokens(text: Any) -> set[str]:
        out = set()
        for word in str(text).lower().split():
            cleaned = word.translate(str.maketrans("", "", string.punctuation))
            if cleaned:
                out.add(cleaned)
        return out

    fused_tokens, expected_tokens = clean_tokens(fused_value), clean_tokens(expected_value)
    if not fused_tokens and not expected_tokens:
        return True
    if not fused_tokens or not expected_tokens:
        return False
    return _jaccard(fused_tokens, expected_tokens) >= threshold


def year_only_match(fused_value: Any, expected_value: Any) -> bool:
    """The year of two dates (parsed with pandas) is the same."""
    if is_missing(fused_value) and is_missing(expected_value):
        return True
    if is_missing(fused_value) or is_missing(expected_value):
        return False

    def to_date(value: Any) -> date | None:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            if isinstance(value, pd.Timestamp):
                return value.date()
        except Exception:  # noqa: BLE001
            pass
        try:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.date()
        except Exception:  # noqa: BLE001
            return None

    d1, d2 = to_date(fused_value), to_date(expected_value)
    if d1 is None or d2 is None:
        return False
    return d1.year == d2.year


def numeric_tolerance_match(fused_value: Any, expected_value: Any, tolerance: float = 0.01) -> bool:
    """|fused - expected| <= tolerance (an ABSOLUTE band)."""
    if is_missing(fused_value) and is_missing(expected_value):
        return True
    if is_missing(fused_value) or is_missing(expected_value):
        return False
    try:
        return abs(float(fused_value) - float(expected_value)) <= tolerance
    except (ValueError, TypeError):
        return str(fused_value).strip() == str(expected_value).strip()


def set_equality_match(fused_value: Any, expected_value: Any) -> bool:
    if is_missing(fused_value) and is_missing(expected_value):
        return True
    if is_missing(fused_value) or is_missing(expected_value):
        return False
    try:
        if isinstance(fused_value, (list, tuple, set)) and isinstance(expected_value, (list, tuple, set)):
            return set(fused_value) == set(expected_value)
        return fused_value == expected_value
    except (TypeError, ValueError):
        return str(fused_value) == str(expected_value)


# ------------------------------------------------ the comparators PyDI lacks

def relative_tolerance_match(fused_value: Any, expected_value: Any, tolerance: float = 0.1) -> bool:
    """|fused - expected| <= tolerance * max(|fused|, |expected|)."""
    if _blank(fused_value) and _blank(expected_value):
        return True
    if _blank(fused_value) or _blank(expected_value):
        return False
    try:
        a, b = float(fused_value), float(expected_value)
    except (TypeError, ValueError):
        return str(fused_value).strip() == str(expected_value).strip()
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return a == b
    return abs(a - b) <= tolerance * scale


def epsilon_tolerance_match(fused_value: Any, expected_value: Any, *, tolerance: float) -> bool:
    """numeric_tolerance_match with the IEEE-754 boundary repaired: a
    difference EQUAL to the tolerance passes (|7.8 - 8.0| is 0.2000...18)."""
    return numeric_tolerance_match(fused_value, expected_value, tolerance=tolerance + 1e-9)


def hardware_strict_spec_match(fused_value: Any, expected_value: Any) -> bool:
    """Hardware strings match when their digit sequences agree and one
    alphanumeric skeleton contains the other."""
    if pd.isna(fused_value) or pd.isna(expected_value):
        return False
    fused, expected = str(fused_value).lower(), str(expected_value).lower()
    if re.findall(r"\d+", fused) != re.findall(r"\d+", expected):
        return False
    fused_clean = re.sub(r"[^a-z0-9]", "", fused)
    expected_clean = re.sub(r"[^a-z0-9]", "", expected)
    return expected_clean in fused_clean or fused_clean in expected_clean


_USB_IF_RENAMES = (
    (re.compile(r"\busb\s*3\.0\b", re.IGNORECASE), "USB 3.2 Gen 1"),
    (re.compile(r"\busb\s*3\.1\s*gen\s*1\b", re.IGNORECASE), "USB 3.2 Gen 1"),
    (re.compile(r"\busb\s*3\.1\s*gen\s*2\b", re.IGNORECASE), "USB 3.2 Gen 2"),
    (re.compile(r"\busb\s*3\.1\b(?!\s*gen)", re.IGNORECASE), "USB 3.2 Gen 2"),
)


def usb_if_hardware_strict(fused_value: Any, expected_value: Any) -> bool:
    """hardware_strict_spec_match after folding the USB-IF renames of the
    same standard (USB 3.0 = USB 3.1 Gen 1 = USB 3.2 Gen 1; USB 3.1 =
    USB 3.1 Gen 2 = USB 3.2 Gen 2) on both sides."""
    if _blank(fused_value) or _blank(expected_value):
        return False

    def fold(v: Any) -> str:
        out = str(v)
        for pattern, canonical in _USB_IF_RENAMES:
            out = pattern.sub(canonical, out)
        return out

    return hardware_strict_spec_match(fold(fused_value), fold(expected_value))


def punct_collapse_match(fused_value: Any, expected_value: Any) -> bool:
    """Equality after casefold with non-alphanumeric runs collapsed to one
    space ('Ace - A80' == 'Ace A80')."""
    if _blank(fused_value) or _blank(expected_value):
        return False

    def fold(v: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(v).casefold()).strip()

    return fold(fused_value) == fold(expected_value)


def casefold_match(fused_value: Any, expected_value: Any) -> bool:
    """Trimmed, case-insensitive equality."""
    if _blank(fused_value) or _blank(expected_value):
        return False
    return str(fused_value).strip().casefold() == str(expected_value).strip().casefold()


def authors_tokenset_equality(fused_value: Any, expected_value: Any) -> bool:
    """Author-SET equality where each name is a set of accent-folded
    alphanumeric tokens: 'j.' vs 'j' and 'Last, First' order are forgiven,
    a missing or extra name or initial is not."""
    if _blank(fused_value) or _blank(expected_value):
        return False

    def name_key(name: str) -> frozenset[str]:
        folded = unicodedata.normalize("NFKD", name)
        folded = "".join(c for c in folded if not unicodedata.combining(c))
        return frozenset(re.findall(r"[a-z0-9]+", folded.casefold()))

    def names(v: Any) -> frozenset[frozenset[str]]:
        v = parse_list_literal(v)
        items = v if isinstance(v, (list, tuple, set)) else re.split(r"[|;]", str(v))
        return frozenset(k for k in (name_key(str(i)) for i in items if str(i).strip()) if k)

    f, e = names(fused_value), names(expected_value)
    return bool(f) and f == e


_LATIN_LETTER_FOLD = str.maketrans({
    "ø": "o", "æ": "ae", "œ": "oe", "đ": "d", "ð": "d", "þ": "th", "ł": "l", "ı": "i",
})


def _fold_person_name(name: Any) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold().translate(_LATIN_LETTER_FOLD).strip()


def folded_person_set_equality(fused_value: Any, expected_value: Any) -> bool:
    """Set equality over accent-folded person names (companies keypeople).
    A missing, extra or different name fails; an empty gold set never passes."""
    if _blank(fused_value) or _blank(expected_value):
        return False

    def names(value: Any) -> frozenset[str]:
        value = parse_list_literal(value)
        items = value if isinstance(value, (list, tuple, set)) else [value]
        return frozenset(key for key in (_fold_person_name(i) for i in items) if key)

    fused, expected = names(fused_value), names(expected_value)
    return bool(expected) and fused == expected


def gold_elements_present(fused_value: Any, expected_value: Any) -> bool:
    """Every gold label is present in the fused list (case-insensitive);
    extra labels are tolerated (games genres, music label). Labels are the
    parsed list elements, still-string values split on commas; nothing is
    split on '/' and nothing is mapped onto a taxonomy."""
    if _blank(fused_value) or _blank(expected_value):
        return False

    def labels(value: Any) -> frozenset[str]:
        value = parse_list_literal(value)
        items = value if isinstance(value, (list, tuple, set)) else [value]
        return frozenset(
            label.casefold()
            for item in items
            for label in (part.strip() for part in str(item).split(","))
            if label
        )

    fused, expected = labels(fused_value), labels(expected_value)
    return bool(expected) and expected <= fused


def full_date_match(fused_value: Any, expected_value: Any) -> bool:
    """The whole calendar date: separators normalised, a time part dropped;
    a value that stops at the year or the month is not the same date."""
    if _blank(fused_value) or _blank(expected_value):
        return False

    def fold(v: Any) -> str:
        text = str(v).strip().replace("/", "-")
        return re.split(r"[ T]", text, maxsplit=1)[0]

    left, right = fold(fused_value), fold(expected_value)
    return bool(left) and left == right


def case_folded_set_equality(fused_value: Any, expected_value: Any) -> bool:
    """Set equality over list elements, case-insensitive only (music
    tracks): typography and punctuation are not folded."""
    def members(value: Any) -> frozenset[str]:
        parsed = parse_list_literal(value)
        items = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
        return frozenset(str(item).strip().casefold() for item in items if str(item).strip())

    expected = members(expected_value)
    return bool(expected) and members(fused_value) == expected


_UNICODE_DASHES = dict.fromkeys([0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212], "-")


def unicode_fold_tokens(fused_value: Any, expected_value: Any) -> bool:
    """Whitespace tokens equal after NFKC + dash + case folding (papers
    title); punctuation is not stripped."""
    if _blank(fused_value) or _blank(expected_value):
        return False

    def tokens(value: Any) -> frozenset[str]:
        text = unicodedata.normalize("NFKC", str(value)).translate(_UNICODE_DASHES)
        return frozenset(part for part in text.casefold().split() if part)

    expected = tokens(expected_value)
    return bool(expected) and tokens(fused_value) == expected


def count_drift_match(fused_value: Any, expected_value: Any, *, floor: float = 2, rate: float = 0.10) -> bool:
    """Counts that grow between crawls (papers cited_by_count): within
    ``floor`` or ``rate`` of the expected value, whichever is larger."""
    try:
        fused = float(str(fused_value).strip())
        expected = float(str(expected_value).strip())
    except (TypeError, ValueError):
        return False
    return abs(fused - expected) <= max(floor, abs(expected) * rate)


# ------------------------------------------------------- the strict tables

_PRODUCT_NUMERICS = ("vram_gb", "storage_gb", "read_speed_mb_s", "write_speed_mb_s",
                     "width_mm", "length_mm", "height_mm", "weight_g")

STRICT_RULES: dict[str, tuple[Rule, ...]] = {
    "companies": (
        ("name", tokenized_match, {}),
        ("revenue", relative_tolerance_match, {"tolerance": 0.02}),
        ("assets", relative_tolerance_match, {"tolerance": 0.02}),
        ("keypeople", folded_person_set_equality, {}),
        ("founded", year_only_match, {}),
        ("country", tokenized_match, {}),
        ("city", tokenized_match, {}),
        ("industry", exact_match, {}),
    ),
    "music": (
        ("name", tokenized_match, {}),
        ("artist", tokenized_match, {}),
        ("duration", numeric_tolerance_match, {"tolerance": 5}),
        ("release-date", full_date_match, {}),
        ("release-country", tokenized_match, {}),
        ("label", gold_elements_present, {}),
        ("tracks", case_folded_set_equality, {}),
        ("genre", exact_match, {}),
    ),
    "games": (
        ("name", tokenized_match, {}),
        ("platform", exact_match, {}),
        ("developer", exact_match, {}),
        ("releaseYear", year_only_match, {}),
        ("ESRB", exact_match, {}),
        ("criticScore", epsilon_tolerance_match, {"tolerance": 1}),
        ("userScore", epsilon_tolerance_match, {"tolerance": 0.1}),
        ("genres", gold_elements_present, {}),
        ("publisher", tokenized_match, {}),
    ),
    "products": (
        ("brand", casefold_match, {}),
        ("product_type", casefold_match, {}),
        *((attribute, relative_tolerance_match, {"tolerance": 0.02}) for attribute in _PRODUCT_NUMERICS),
        ("chipset_name", hardware_strict_spec_match, {}),
        ("bus_type", usb_if_hardware_strict, {}),
        ("interface_type", hardware_strict_spec_match, {}),
        ("memory_type", hardware_strict_spec_match, {}),
        ("model", punct_collapse_match, {}),
        ("model_number", punct_collapse_match, {}),
        ("storage_connection_type", tokenized_match, {}),
        ("color", casefold_match, {}),
        ("form_factor", exact_match, {}),
    ),
    "papers": (
        ("type", exact_match, {}),
        ("title", unicode_fold_tokens, {}),
        ("authors", authors_tokenset_equality, {}),
        ("publication_year", year_only_match, {}),
        ("journal", tokenized_match, {}),
        ("volume", exact_match, {}),
        ("issue", exact_match, {}),
        ("first_page", exact_match, {}),
        ("last_page", exact_match, {}),
        ("referenced_works_count", exact_match, {}),
        ("cited_by_count", count_drift_match, {"floor": 2, "rate": 0.10}),
    ),
}

DOMAINS = tuple(STRICT_RULES)


def rules_for(domain: str) -> tuple[Rule, ...]:
    try:
        return STRICT_RULES[domain]
    except KeyError:
        raise ValueError(f"unknown domain {domain!r}; known: {DOMAINS}") from None


def comparator_for(domain: str, attribute: str) -> Callable[[Any, Any], bool]:
    """The bound comparator for one attribute (exact_match when the attribute
    has no rule, as PyDI's evaluator defaults)."""
    for name, function, kwargs in rules_for(domain):
        if name == attribute:
            return lambda a, b, _f=function, _k=kwargs: _f(a, b, **_k)
    return exact_match


def describe(domain: str) -> list[dict[str, Any]]:
    """The rule table in a serialisable form (for READMEs and manifests)."""
    return [{"attribute": a, "comparator": f.__name__, **k} for a, f, k in rules_for(domain)]
