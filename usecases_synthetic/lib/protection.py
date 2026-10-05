"""Protection set construction + closeness contract for the synthetic generation pipeline.

Implements ``knobs/cross_cutting.md`` §"Gold standard incompleteness and
pooling":

    expanded_positives = test_gold ∪ train_gold ∪ val_gold ∪ pooled_positives

The protection set constrains the generator (never the evaluator).
Protected entity IDs must not be dropped (K2 easy), used as distractor
seeds (K2 hard), or mined as hard negatives (K1/K2).

Closeness contract
-------------------------------------------
For every fusion val + test (entity, attribute) cell, ≥1 surviving
record across all sources must remain "close enough" to the fusion
target value so a lenient fusion strategy can recover the truth.
"Close enough" replaces the prior exact-match guarantee. The thresholds
are fixed and
implemented below via :func:`is_close_enough` + :class:`ToleranceSpec`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .domain_config import (
    POOLS_DIR,
    USECASES_DIR,
    data_root_for_domain,
    load_domain_config,
    task_dir,
)
from .format_operators import _parse_number_in_locale
from .fusion_gold_keys import (
    read_json_records,
    record_key,
    sniff_fusion_format,
    xml_record_key,
)
from .loaders import read_em_gold_csv
from .niche_metrics import _levenshtein_ratio, lexical_extended_jaccard
from .rate_tables import get_locale_config

# ---------------------------------------------------------------------------
# source_ids-keyed fusion gold (papers at every tier, products task variants)
# ---------------------------------------------------------------------------
#
# Pre-2026 domains ship fusion gold as XML keyed by a cluster-anchor source
# record id (e.g. ``mbrainz_974``) with per-attribute provenance. The papers
# domain ships fusion gold as JSON-lines records with a ``source_ids`` list of
# their member record ids (no ``id``, no DOI); the products task variants ship
# XML records with a ``<source_ids>`` element and no ``<id>``. The protection +
# closeness machinery needs one real source record id per fusion entity (the
# consumer finds it as a cluster member via
# ``next(mid for _, mid in members if mid in fusion_gold_ids)``), so such a
# record is keyed on its anchor (:func:`fusion_gold_keys.record_key` /
# :func:`fusion_gold_keys.xml_record_key`: the first primary-source member,
# else the smallest id of the next source present) -- the id the public
# scorer keys the record on -- and that anchor is used consistently as BOTH
# the protected id and the ``load_fusion_target_values`` key. Record ids are
# never perturbed, so the anchor is stable across variant levels.

# Bookkeeping fields of a JSON gold record: never a fusion target.
_JSON_KEY_FIELDS: frozenset[str] = frozenset({"id", "source_ids"})
# publisher + cited_by_count get no fuser in the papers silver strategy
# (fusion_silver_standard._build_papers_strategy), so they carry no
# closeness contract.
_PAPERS_NON_TARGET_ATTRS: frozenset[str] = frozenset(
    {"publisher", "cited_by_count"}
)


def _read_fusion_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines or JSON-array fusion file, dispatching on content
    (the papers variants carry JSON lines under ``*.xml`` names)."""
    return read_json_records(path)


def _json_target_attrs(record: dict[str, Any]) -> dict[str, list[str]]:
    """``{attribute: [value, ...]}`` of a JSON gold record, bookkeeping and
    non-target fields dropped, empty cells skipped."""
    attrs: dict[str, list[str]] = {}
    for key, value in record.items():
        if key in _JSON_KEY_FIELDS or key in _PAPERS_NON_TARGET_ATTRS:
            continue
        vals = _coerce_target_values(value)
        if vals:
            attrs[key] = vals
    return attrs


def _coerce_target_values(value: Any) -> list[str]:
    """Coerce a fusion-gold cell to a list of non-empty string target values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return out
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return []
    return [text]


# ---------------------------------------------------------------------------
# Entity-ID protection sets
# ---------------------------------------------------------------------------


def _load_em_gold_ids(domain: str) -> set[str]:
    """Load all entity IDs from EM gold CSVs (train + val + test)."""
    root = data_root_for_domain(domain) or USECASES_DIR
    em_dir = task_dir(domain, root=root) / "input" / "entitymatching"
    ids: set[str] = set()

    for split in ("_train.csv", "_val.csv", "_test.csv"):
        for csv_path in sorted(em_dir.glob(f"*{split}")):
            df = read_em_gold_csv(csv_path)
            ids.update(df["id1"].astype(str))
            ids.update(df["id2"].astype(str))

    return ids


def _load_fusion_protected_ids(domain: str) -> set[str]:
    """Load entity IDs from the fusion validation + test set files.

    Reads both fusion files declared by the domain config's
    ``fusion_files`` block (defaults: ``validation_set.xml`` and
    ``test_set.xml``) so that any entity referenced by either fusion
    split is protected. An id-keyed XML record contributes its ``<id>``
    (every ``<id>`` element, as always); a source_ids-keyed record (JSON
    lines -- papers --, or an XML record without ``<id>``) contributes its
    anchor.
    """
    cfg = load_domain_config(domain)
    ids: set[str] = set()

    for path in cfg.fusion_paths():
        if not path.exists():
            continue
        if sniff_fusion_format(path) in ("jsonl", "json"):
            for record in _read_fusion_records(path):
                key = record_key(record, domain)
                if key is not None:
                    ids.add(key)
            continue
        tree = ET.parse(path)
        root = tree.getroot()
        for id_elem in root.iter("id"):
            if id_elem.text:
                ids.add(id_elem.text.strip())
        for entity_elem in root:
            if entity_elem.find("id") is None:
                key, _ = xml_record_key(entity_elem, domain)
                if key is not None:
                    ids.add(key)

    return ids


# Back-compat alias (older callers in K3/K4 still import this name).
_load_fusion_gold_ids = _load_fusion_protected_ids


def _load_pooled_positive_ids(domain: str) -> set[str]:
    """Load entity IDs from the pooled positives CSV."""
    pool_path = POOLS_DIR / domain / "pooled_positives.csv"
    ids: set[str] = set()

    if not pool_path.exists():
        return ids

    df = pd.read_csv(pool_path)
    ids.update(df["id1"].astype(str))
    ids.update(df["id2"].astype(str))

    return ids


def build_expanded_positives(domain: str) -> set[str]:
    """Build the expanded positives protection set for a domain.

    Computes::

        expanded_positives = EM_gold_ids ∪ fusion_protected_ids ∪ pooled_positive_ids
    """
    em_ids = _load_em_gold_ids(domain)
    fusion_ids = _load_fusion_protected_ids(domain)
    pool_ids = _load_pooled_positive_ids(domain)
    return em_ids | fusion_ids | pool_ids


def build_drop_corner_protection_set(
    domain: str, protection_source: str = "gold"
) -> set[str]:
    """Protection set for K2's drop-corner-touching operator.

    The drop-corner operator removes existing canonical entities to
    reduce the corner-pair ratio. It must respect a narrower protection
    set than ``build_expanded_positives``: the fusion val/test gold is
    the only universally-required protection (it is the fixed
    cross-level evaluation surface for fusion accuracy). EM gold
    positives are NOT protected here — the corner-filled
    ``regenerate_em_splits`` already handles the case where EM gold
    members get dropped (pruning Set 1 and corner-mining Set 2 from the
    surviving pool). Without this narrowing, ``pool_quality: live``
    domains (products) have EM gold coextensive with the full record
    set, so every drop candidate is "protected" and the operator
    becomes a noop.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"products"``).
    protection_source : str
        ``"gold"`` (default): fusion val/test gold only. EM gold and
        pool members may be dropped. ``"silver"``: also includes every
        pool member, matching the silver-standard "every cluster
        member is fusion-recoverable, therefore protected" semantics.
        On pool-live domains under ``"silver"`` the operator becomes a
        noop (every entity protected); the caller should expect that.

    Returns
    -------
    set of str
        Protected record IDs. Caller compares canonical entity members
        against this set.
    """
    fusion_ids = _load_fusion_protected_ids(domain)
    if protection_source == "silver":
        pool_ids = _load_pooled_positive_ids(domain)
        return fusion_ids | pool_ids
    return fusion_ids


def is_protected(entity_id: str, expanded_positives: set[str]) -> bool:
    """Check whether an entity ID is in the protection set."""
    return entity_id in expanded_positives


# ---------------------------------------------------------------------------
# Closeness contract — tolerance specification + per-cell test
# ---------------------------------------------------------------------------


_VALID_KINDS = (
    "continuous",
    "year",
    "date",
    "nominal",
    "long_string",
    "free_text",
    "list",
)


@dataclass(frozen=True)
class ToleranceSpec:
    """Per-attribute tolerance specification for the closeness contract.

    Parameters
    ----------
    kind : str
        One of ``continuous`` (numeric, relative band), ``year``
        (calendar year, ±1 absolute), ``date`` (calendar day delta ≤ N
        days), ``nominal`` (Levenshtein ratio on short strings),
        ``long_string`` (extended Jaccard on tokenised long strings),
        ``free_text`` (extended Jaccard, looser threshold), ``list``
        (extended Jaccard over flattened set tokens).
    threshold : float
        Numeric tolerance (relative for ``continuous``; absolute years
        for ``year``; absolute calendar-days for ``date``; ratio in
        ``[0, 1]`` for string kinds).
    inner_token_threshold : float
        Inner-Levenshtein gate for the extended-Jaccard kinds.
    """

    kind: str
    threshold: float
    inner_token_threshold: float = 0.8


_DEFAULT_TOLERANCE_BY_KIND: dict[str, ToleranceSpec] = {
    "continuous": ToleranceSpec(kind="continuous", threshold=0.03),
    "year": ToleranceSpec(kind="year", threshold=1.0),
    "date": ToleranceSpec(kind="date", threshold=1.0),
    "nominal": ToleranceSpec(kind="nominal", threshold=0.85),
    "long_string": ToleranceSpec(
        kind="long_string", threshold=0.6, inner_token_threshold=0.8
    ),
    "free_text": ToleranceSpec(
        kind="free_text", threshold=0.5, inner_token_threshold=0.8
    ),
    "list": ToleranceSpec(kind="list", threshold=0.5, inner_token_threshold=0.8),
}


# Per-domain canonical-attribute → kind map, shared by K1/K5/K6.
# Per-attribute overrides may be authored
# in ``config/knob_06_noise/<domain>.yaml::fusion_protection_tolerance``.
_DEFAULT_KIND_BY_DOMAIN_ATTR: dict[str, dict[str, str]] = {
    "companies": {
        "name": "long_string",
        "country": "nominal",
        "city": "nominal",
        "industry": "nominal",
        "founded": "year",
        "keypeople": "long_string",
        "assets": "continuous",
        "revenue": "continuous",
    },
    "games": {
        "name": "long_string",
        "platform": "nominal",
        "ESRB": "nominal",
        "releaseYear": "year",
        "developer": "nominal",
        "publisher": "nominal",
        "genres": "list",
        "criticScore": "continuous",
        "userScore": "continuous",
        "globalSales": "continuous",
        "series": "nominal",
    },
    "music": {
        "name": "long_string",
        "artist": "long_string",
        "release-country": "nominal",
        "genre": "nominal",
        "release-date": "date",
        "duration": "continuous",
        "label": "nominal",
    },
    # Products widened 5 -> 24 attrs (score/perturb
    # the FULL target_schema, incl. the 5 sparse dims color/width_mm/length_mm/
    # height_mm/weight_g). url + id excluded. Kinds mirror the target_schema
    # JSON types (number -> continuous; enum/open_taxonomy/short string ->
    # nominal; long identifier strings -> long_string; description ->
    # free_text). The derived title_description column of the old notebook is
    # gone with the as-extracted sources.
    "products": {
        "title": "long_string",
        "brand": "nominal",
        "description": "free_text",
        "price": "continuous",
        "priceCurrency": "nominal",
        "title_description": "free_text",
        "model": "long_string",
        "model_number": "long_string",
        "product_type": "nominal",
        "chipset_name": "long_string",
        "vram_gb": "continuous",
        "storage_gb": "continuous",
        "read_speed_mb_s": "continuous",
        "write_speed_mb_s": "continuous",
        "bus_type": "nominal",
        "interface_type": "nominal",
        "memory_type": "nominal",
        "storage_connection_type": "nominal",
        "form_factor": "nominal",
        "width_mm": "continuous",
        "length_mm": "continuous",
        "height_mm": "continuous",
        "weight_g": "continuous",
        "color": "nominal",
    },
}


def kind_map_for_domain(domain: str) -> dict[str, str]:
    """Return the per-attribute kind map for ``domain``, alias-aware.

    Aliased domains (e.g. ``music-small`` with ``knob_config_alias: music``
    in its domain YAML) inherit the source domain's kind map.
    """
    from .domain_config import _resolve_knob_config_alias

    direct = _DEFAULT_KIND_BY_DOMAIN_ATTR.get(domain)
    if direct:
        return direct
    alias = _resolve_knob_config_alias(domain)
    if alias:
        return _DEFAULT_KIND_BY_DOMAIN_ATTR.get(alias, {})
    return {}


def fusion_cell_tolerance(
    domain: str,
    canonical_attribute: str,
    config_overrides: dict[str, dict[str, float | str]] | None = None,
) -> ToleranceSpec:
    """Resolve the tolerance spec for a (domain, canonical attribute) cell.

    Resolution order:

    1. ``config_overrides[canonical_attribute]`` — per-attribute override
       authored in ``config/knob_06_noise/<domain>.yaml`` under the
       ``fusion_protection_tolerance`` block. Each entry may carry ``kind``
       and/or ``threshold`` and/or ``inner_token_threshold``.
    2. The per-domain default kind from
       :data:`_DEFAULT_KIND_BY_DOMAIN_ATTR`, with default thresholds from
       :data:`_DEFAULT_TOLERANCE_BY_KIND`.
    3. Fallback to ``long_string`` (a safe, moderately strict default).

    Parameters
    ----------
    domain : str
        Domain name (``"companies"``, ``"games"``, ``"music"``).
    canonical_attribute : str
        Canonical attribute name as authored in
        ``config/domains/<domain>.yaml`` (e.g. ``"name"``, ``"country"``).
    config_overrides : dict or None
        Optional per-domain override block.

    Returns
    -------
    ToleranceSpec
    """
    domain_kinds = kind_map_for_domain(domain)
    default_kind = domain_kinds.get(canonical_attribute, "long_string")
    base = _DEFAULT_TOLERANCE_BY_KIND[default_kind]

    if not config_overrides:
        return base

    override = config_overrides.get(canonical_attribute)
    if not override:
        return base

    kind = str(override.get("kind", base.kind))
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"Unknown tolerance kind {kind!r} for "
            f"{domain}.{canonical_attribute}; valid: {_VALID_KINDS}"
        )

    threshold = override.get("threshold", _DEFAULT_TOLERANCE_BY_KIND[kind].threshold)
    inner = override.get(
        "inner_token_threshold",
        _DEFAULT_TOLERANCE_BY_KIND[kind].inner_token_threshold,
    )
    return ToleranceSpec(
        kind=kind,
        threshold=float(threshold),
        inner_token_threshold=float(inner),
    )


# ---- Parsers used by is_close_enough --------------------------------------

_YEAR_RE = re.compile(r"\b(\d{4})\b")
_LIST_SPLIT_RE = re.compile(r"[,;|]|\s/\s|\s\|\s")


def _parse_float(value: str) -> float | None:
    s = value.strip()
    if not s:
        return None
    # Handle scientific notation, currency punctuation. Drop common
    # thousands separators / spaces / currency markers; keep decimal
    # point and exponent.
    s2 = re.sub(r"[\s,$€£¥₩]", "", s)
    try:
        return float(s2)
    except ValueError:
        return None


# Strict number shapes tried by the continuous-kind closeness: the Knob 5
# number locales of ``config/knob_05_format/_tables/number_locales.yaml``.
_CLOSENESS_NUMBER_LOCALES: tuple[str, ...] = ("plain", "en_US", "de_DE", "fr_FR")
# fr_FR digit grouping as other writers emit it: NBSP (U+00A0) and narrow
# NBSP (U+202F) next to the plain space number_locales.yaml uses.
_FR_GROUP_SPACE_RE = re.compile("[\u00a0\u202f]")
_SCIENTIFIC_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+")


def _has_zero_led_grouping(value: str, locale_id: str) -> bool:
    """True if *value*'s integer part carries *locale_id*'s group separator
    but starts with a zero.

    No writer groups a zero-led integer part (Knob 5 formats from a
    Decimal), yet the strict :func:`format_operators._parse_number_in_locale`
    accepts the shape: ``"0.178"`` would read as de_DE 178 and ``"0,500"``
    as en_US 500. Such readings are dropped.
    """
    cfg = get_locale_config(locale_id)
    thousands_sep: str = cfg["thousands_sep"]
    if not thousands_sep:
        return False
    int_part = value.strip().lstrip("+-").split(cfg["decimal_sep"], 1)[0]
    return thousands_sep in int_part and int_part.startswith("0")


@lru_cache(maxsize=65536)
def _numeric_readings(value: str) -> tuple[float, ...]:
    """Every distinct number *value* can denote, for the closeness check.

    :func:`_parse_float` drops commas and reads the dot as the decimal
    point, so every decimal-comma form Knob 5 writes is misread
    (``"2.000,0"`` -> 2.0, ``"500,0"`` -> 5000, ``"14 000,0"`` -> 140000).
    This returns the strict reading of *value* under each shape of
    :data:`_CLOSENESS_NUMBER_LOCALES` (via
    :func:`format_operators._parse_number_in_locale`; fr_FR also with NBSP /
    NNBSP grouping; no grouped zero-led integer part, see
    :func:`_has_zero_led_grouping`) plus scientific notation, deduplicated
    in that order. An ambiguous string keeps all its readings (``"2.000"``:
    2.0 plain, 2000 de_DE; ``"1,234"``: 1234 en_US, 1.234 de_DE / fr_FR).

    Only when no shape matches does it fall back to :func:`_parse_float`
    (currency markers, ``"1 234.5"``, ``".5"``, ``"nan"``, ...), so every
    value today's parser reads correctly keeps that reading: a plain
    decimal (``"2000.0"``, ``"-3.25"``, ``"0.178"``, ``"1e5"``) has exactly
    the :func:`_parse_float` reading, except a plain value with exactly
    three decimals and a non-zero integer part of at most three digits
    (``"3.134"``), which also has the de_DE thousands reading (3134).
    Returns an empty tuple for an empty or unparseable value.
    """
    s = value.strip()
    if not s:
        return ()
    readings: list[float] = []

    def _add(reading: float) -> None:
        if reading not in readings:
            readings.append(reading)

    for locale_id in _CLOSENESS_NUMBER_LOCALES:
        candidate = _FR_GROUP_SPACE_RE.sub(" ", s) if locale_id == "fr_FR" else s
        parsed = _parse_number_in_locale(candidate, locale_id)
        if parsed is not None and not _has_zero_led_grouping(candidate, locale_id):
            _add(float(parsed))
    if _SCIENTIFIC_RE.fullmatch(s):
        _add(float(s))
    if not readings:
        fallback = _parse_float(s)
        if fallback is not None:
            _add(fallback)
    return tuple(readings)


def _parse_year(value: str) -> int | None:
    s = value.strip()
    if not s:
        return None
    # Try ISO-style first.
    m = re.match(r"^(\d{4})", s)
    if m:
        try:
            year = int(m.group(1))
            if 1500 <= year <= 2200:
                return year
        except ValueError:
            return None
    # Fallback: look for any 4-digit run.
    m = _YEAR_RE.search(s)
    if m:
        try:
            year = int(m.group(1))
            if 1500 <= year <= 2200:
                return year
        except ValueError:
            return None
    return None


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
)


def _parse_date(value: str) -> datetime | None:
    s = value.strip()
    if not s:
        return None
    # Strip trailing timezone offsets like ``+01:00`` / ``Z``.
    s2 = re.sub(r"([+-]\d{2}:?\d{2}|Z)\s*$", "", s)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s2, fmt)
        except ValueError:
            continue
    return None


def _split_list_tokens(value: str) -> list[str]:
    """Split a list-like cell into tokens.

    Handles ``"['a', 'b']"`` JSON-ish forms, comma / semicolon /
    pipe-separated forms, and falls back to whitespace tokenisation.
    """
    s = value.strip()
    if not s:
        return []
    # JSON-ish list: ``['a', 'b']`` → strip brackets + quotes + commas.
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
        return [p for p in parts if p]
    parts = _LIST_SPLIT_RE.split(s)
    return [p.strip() for p in parts if p.strip()]


# ---- Closeness predicate --------------------------------------------------


def is_close_enough(
    value: str | float | int | None,
    target: str | float | int | None,
    tolerance: ToleranceSpec,
) -> bool:
    """Test whether *value* is close enough to *target* under *tolerance*.

    Returns False on any unparseable input under numeric / date kinds —
    the closeness contract requires evidence of closeness, not absence
    of evidence to the contrary.

    Under the ``continuous`` kind the value side is locale-tolerant: it is
    close if ANY of its :func:`_numeric_readings` (plain / en_US / de_DE /
    fr_FR / scientific, falling back to :func:`_parse_float`) is within the
    relative band of the target, which :func:`_parse_float` reads.

    Parameters
    ----------
    value : str, float, int, or None
        Source-side cell value.
    target : str, float, int, or None
        Fusion target value.
    tolerance : ToleranceSpec
        Resolved tolerance for this attribute.
    """
    if value is None or target is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(target, float) and pd.isna(target):
        return False

    v = str(value).strip()
    t = str(target).strip()
    if not t:
        # If target has no value, contract is vacuously true.
        return True
    if not v:
        return False

    kind = tolerance.kind

    if kind == "continuous":
        # Locale-tolerant on the value side (a source cell may carry a Knob 5
        # de_DE / fr_FR form): close if ANY strict reading is within the
        # band. The target is the canonical fusion value, read as before.
        ft = _parse_float(t)
        if ft is None:
            return False
        denom = max(abs(ft), 1e-9)
        return any(
            abs(fv - ft) / denom <= tolerance.threshold
            for fv in _numeric_readings(v)
        )

    if kind == "year":
        yv = _parse_year(v)
        yt = _parse_year(t)
        if yv is None or yt is None:
            return False
        return abs(yv - yt) <= int(tolerance.threshold)

    if kind == "date":
        dv = _parse_date(v)
        dt = _parse_date(t)
        if dv is None or dt is None:
            # Fall back to year-level if either side parses only as year.
            yv = _parse_year(v)
            yt = _parse_year(t)
            if yv is not None and yt is not None:
                # 1-day tolerance translates loosely to "same year" here.
                return yv == yt
            return False
        return abs((dv - dt).days) <= int(tolerance.threshold)

    if kind == "nominal":
        return _levenshtein_ratio(v.casefold(), t.casefold()) >= tolerance.threshold

    if kind in ("long_string", "free_text"):
        sim = lexical_extended_jaccard(
            v, t, inner_token_threshold=tolerance.inner_token_threshold
        )
        return sim >= tolerance.threshold

    if kind == "list":
        toks_v = _split_list_tokens(v)
        toks_t = _split_list_tokens(t)
        if not toks_v and not toks_t:
            return True
        if not toks_v or not toks_t:
            return False
        # Flatten back to a string so lexical_extended_jaccard can tokenise.
        sim = lexical_extended_jaccard(
            " ".join(toks_v),
            " ".join(toks_t),
            inner_token_threshold=tolerance.inner_token_threshold,
        )
        return sim >= tolerance.threshold

    raise ValueError(f"Unknown tolerance kind: {kind!r}")


# ---------------------------------------------------------------------------
# Fusion target value loader
# ---------------------------------------------------------------------------


def load_fusion_target_values(domain: str) -> dict[str, dict[str, list[str]]]:
    """Load per-(entity, attribute) target values from fusion val + test.

    Reads both fusion files declared by the domain config's
    ``fusion_files`` block (defaults: ``validation_set.xml`` and
    ``test_set.xml``). For multi-valued attributes (e.g. games'
    ``<genres><genre>...</genre></genres>``, companies'
    ``<keypeople><name>...</name></keypeople>``), the inner text values
    are aggregated into a list. For scalar attributes the list contains
    the single text value. Empty / null cells are skipped.

    Parameters
    ----------
    domain : str
        Domain name.

    Returns
    -------
    dict
        ``{entity_id: {attribute: [value, ...]}}``. Test-set wins on
        conflicting entity IDs (val is read first, then test overrides).
    """
    cfg = load_domain_config(domain)
    out: dict[str, dict[str, list[str]]] = {}

    for path in cfg.fusion_paths():
        if not path.exists():
            continue
        if sniff_fusion_format(path) in ("jsonl", "json"):
            # Papers: key targets by the same anchor used by
            # _load_fusion_protected_ids so the closeness consumer's member
            # lookup resolves. source_ids (bookkeeping) and publisher /
            # cited_by_count (not fused) carry no fusion target.
            for record in _read_fusion_records(path):
                record_id = record_key(record, domain)
                if record_id is None:
                    continue
                json_attrs = _json_target_attrs(record)
                if json_attrs:
                    out[record_id] = json_attrs
            continue
        tree = ET.parse(path)
        root = tree.getroot()
        for entity_elem in root:
            # <id> when present (the legacy parse, unchanged), else the
            # anchor of <source_ids> (the products task variants).
            eid, key_tag = xml_record_key(entity_elem, domain)
            if eid is None:
                continue
            attrs: dict[str, list[str]] = {}
            for child in entity_elem:
                if child.tag == key_tag:
                    continue
                # Multi-valued (has child elements).
                inner_values: list[str] = []
                if list(child):
                    for sub in child:
                        if sub.text and sub.text.strip():
                            inner_values.append(sub.text.strip())
                elif child.text and child.text.strip():
                    inner_values.append(child.text.strip())
                if inner_values:
                    attrs[child.tag] = inner_values
            if attrs:
                out[eid] = attrs

    return out


# ---------------------------------------------------------------------------
# Per-cell closeness gate (used by K1/K3/K5/K6/K10 dispatchers)
# ---------------------------------------------------------------------------


def cell_has_close_survivor(
    target_values: list[str],
    surviving_values: Iterable[str | None],
    tolerance: ToleranceSpec,
) -> bool:
    """Return True if any surviving value is close to any target value.

    Used as the per-cell closeness gate: after a candidate mutation, the
    caller passes the post-commit values for all sources mapped to the
    same canonical attribute (the candidate value for the source being
    mutated, the current values for others). This function returns
    True iff ≥1 of those values is within tolerance of ≥1 target value.

    Parameters
    ----------
    target_values : list[str]
        Fusion target value(s) for this (entity, canonical attribute).
        For multi-valued list attributes, callers may pass the joined
        list (recommended) or one entry per gold value (less strict).
    surviving_values : iterable of str or None
        Post-commit values across all sources mapped to the same
        canonical attribute. Empty / None entries are ignored.
    tolerance : ToleranceSpec
        Resolved tolerance for the canonical attribute.

    Returns
    -------
    bool
    """
    if not target_values:
        # No target authored → contract is vacuously true.
        return True
    for sv in surviving_values:
        if sv is None:
            continue
        if isinstance(sv, float) and pd.isna(sv):
            continue
        s = str(sv).strip()
        if not s:
            continue
        for tv in target_values:
            if is_close_enough(s, tv, tolerance):
                return True
    return False
