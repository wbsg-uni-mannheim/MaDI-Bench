"""Products normalization rules, shared by the human baseline (P1) and the
variant generator's fusion silver standard.

``normalize_products_frame`` applies every rule of the table below to the
translated frames after schema translation; the silver standard uses the full
table. The products notebook runs a copy of this code without seven
operations that only reproduce spellings on attributes its blocking and
matching never read (usb_if_fold, interface_alias, title_case_from_uniform,
color_separator, paren_unwrap, hyphen_to_space, bus_qualifier_drop).
"""
from __future__ import annotations

import pandas as pd
import html
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

def dec(v: Any) -> Decimal | None:
    try:
        return Decimal(str(v).strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None

def op_whitespace(s: str) -> str:
    s = s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    s = s.replace("﻿", "").replace("​", "")
    return re.sub(r"\s+", " ", s).strip()

def op_markup(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s))

def op_mojibake(s: str) -> str:
    """Undo cp1252-as-utf8 mangling ('Ã ' -> 'à'), iterated for double mangling."""
    for _ in range(3):
        try:
            t = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if t == s:
            break
        s = t
    return s

def op_nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def op_symbols(s: str) -> str:
    return op_whitespace(re.sub("[®™©]", "", s))

def op_title_case_from_uniform(s: str) -> str:
    """Title-case a cell whose letters are ALL upper or all lower; a mixed-case
    cell is a spelling choice and is left alone."""
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 2 and (all(c.isupper() for c in letters) or all(c.islower() for c in letters)):
        return s.title()
    return s

_BRAND_SUFFIX = ("technology", "technologies", "memory", "inc", "inc.", "corp", "corp.",
                 "corporation", "ltd", "ltd.", "llc", "co", "co.", "gmbh", "electronics")
_USB_IF = re.compile(r"\bUSB(?:-C| Type-C| Type-A)?\s*3\.[12]\s*Gen\s*1\b", re.I)
_IFACE_ALIAS = [(re.compile(r"\b(?:USB[- ]?A|Type[- ]A)\b", re.I), "USB Type-A"),
                (re.compile(r"\b(?:USB[- ]?C|Type[- ]C)\b", re.I), "USB Type-C"),
                (re.compile(r"\bmicro[- ]?usb\b", re.I), "micro-usb")]

def _op_bare_inch(s: str) -> str:
    """A lone drive-size number ('3.5', '2.5') -> 'N.N-inch' (form_factor only)."""
    return f"{s.strip()}-inch" if re.fullmatch(r"[1-5](?:\.\d)?", s.strip()) else s

def _op_binary_gb(s: str) -> str:
    d = dec(s)
    if d is not None and d >= 1024 and d % 1024 == 0:
        return str(d / 1024 * 1000)
    return s

def _op_brand_suffix_strip(s: str) -> str:
    """'Kingston Technology' -> 'Kingston': one trailing marketing/legal word from a short table."""
    toks = s.strip().split()
    if len(toks) >= 2 and toks[-1].casefold() in _BRAND_SUFFIX:
        return " ".join(toks[:-1])
    return s

def _op_bus_qualifier_drop(s: str) -> str:
    """Speed, generation and roman qualifiers off a bus name: 'SATA III 6Gb/s' -> 'SATA',
    'SAS 12Gb/s' -> 'SAS', 'USB 3.2 Gen 1 (USB 3.0)' -> 'USB 3.2 Gen 1'."""
    t = re.sub(r"\s*\([^()]*\)\s*$", "", s)                       # trailing parenthetical
    t = re.sub(r"[\s-]*\d+(?:\.\d+)?\s*G\s*b(?:/|p)?s\b", "", t, flags=re.I)   # 6Gb/s, 12 Gbps, -6GBPS
    t = re.sub(r"\s+(?:III|II|IV)\b", "", t)                       # SATA III -> SATA
    return re.sub(r"\s+", " ", t).strip()

def _op_color_separator(s: str) -> str:
    """Multi-colour cells joined the gold's way: 'Black+Blue', 'White,Yellow', 'Black / Blue' -> 'A/B'."""
    return re.sub(r"\s*[+,/]\s*", "/", s.strip()) if re.search(r"[A-Za-z]\s*[+,/]\s*[A-Za-z]", s) else s

def _op_dim_to_mm(s: str, ctx: dict | None = None) -> str:
    """Dimensions: GPU cards below 60 (length) / 20 (height, width) are in cm (x10);
    drives and sticks below 2 are in inches (x25.4). The notebook's wider bands (<5 inch,
    5-15 cm) would mis-scale real sizes: M.2 drives are 2.3-3.5 mm thick and sticks 6-12 mm high."""
    d = dec(s); c = ctx or {}
    if d is None or d <= 0:
        return s
    t, a = c.get("product_type", ""), c.get("attribute", "")
    if t == "GPU":
        lim = 60 if a == "length_mm" else 20
        return str((d * 10).normalize()) if d < lim else s
    if t in {"HDD", "SSD", "USB_STICK"} and d < 2:
        return str((d * Decimal("25.4")).normalize())
    return s

def _op_float_noise(s: str) -> str:
    d = dec(s)
    if d is None:
        return s
    q = d.quantize(__import__("decimal").Decimal("0.000001"))
    return str(q.normalize()) if abs(q - d) < __import__("decimal").Decimal("1e-9") and q != d else s

def _op_hyphen_collapse(s: str) -> str:
    """'A-DATA' -> 'ADATA': a hyphen between letters is dropped (brand only)."""
    return re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", s)

def _op_hyphen_to_space(s: str) -> str:
    """'Dual-Slot' -> 'Dual Slot': a hyphen between two words becomes a space (form_factor only)."""
    return re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", s)

def _op_inch(s: str) -> str:
    m = re.match(r"^(\d(?:\.\d+)?)\s*(?:\"|”|''|-?\s*[Ii]nch(?:es)?|-?\s*in\b)(.*)$", s)
    return f"{m.group(1)}-inch{m.group(2)}" if m else s

def _op_interface_alias(s: str) -> str:
    """Connector aliases onto the gold's spelling, token-wise: 'USB-A and Lightning' ->
    'USB Type-A and Lightning', 'Type-C' -> 'USB Type-C'."""
    t = s
    for rx, rep in _IFACE_ALIAS:
        t = rx.sub(rep, t)
    t = re.sub(r"(?:\bUSB\s+)+(USB Type-[AC])", r"\1", t)     # 'USB Type-A' was already canonical
    return re.sub(r"\s+", " ", t).strip()

def _op_kg_to_g(s: str, ctx: dict | None = None) -> str:
    """A weight below 1 is in kg: x1000 (nothing weighs under a gram; USB sticks and M.2
    drives legitimately weigh 3 to 10 g, which the first draft's <10 threshold mis-scaled)."""
    d = dec(s)
    return str((d * 1000).normalize()) if d is not None and 0 < d < 1 else s

def op_markup(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s))

def _op_mb_to_gb(s: str, ctx: dict | None = None) -> str:
    """VRAM of 512 or more is in MB: /1024."""
    d = dec(s)
    return str((d / 1024).normalize()) if d is not None and d >= 512 and d % 1024 == 0 else s

def op_mojibake(s: str) -> str:
    """Undo cp1252-as-utf8 mangling ('Ã ' -> 'à'), iterated for double mangling."""
    for _ in range(3):
        try:
            t = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if t == s:
            break
        s = t
    return s

def op_nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def _op_paren_unwrap(s: str) -> str:
    """'M.2 (2280)' -> 'M.2 2280': parentheses around a trailing token are dropped, the token kept."""
    return re.sub(r"\s*\(([^()]*)\)\s*$", r" \1", s).strip()

def op_symbols(s: str) -> str:
    return op_whitespace(re.sub("[®™©]", "", s))

def _op_tb_to_gb(s: str, ctx: dict | None = None) -> str:
    """HDD/SSD capacity below 64 is in TB: x1000 (USB sticks are never rescaled)."""
    d = dec(s)
    if d is None or (ctx or {}).get("product_type") not in {"HDD", "SSD"} or not (0 < d < 64):
        return s
    return str((d * 1000).normalize())

def op_title_case_from_uniform(s: str) -> str:
    """Title-case a cell whose letters are ALL upper or all lower; a mixed-case
    cell is a spelling choice and is left alone."""
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 2 and (all(c.isupper() for c in letters) or all(c.islower() for c in letters)):
        return s.title()
    return s

def _op_usb_if_fold(s: str) -> str:
    """USB-IF renames folded to the gold's name: 'USB 3.1 Gen 1' / 'USB 3.2 Gen 1' -> 'USB 3.0';
    'USB3.0' -> 'USB 3.0'."""
    t = _USB_IF.sub("USB 3.0", s)
    return re.sub(r"\bUSB(\d\.\d)\b", r"USB \1", t)

def op_whitespace(s: str) -> str:
    s = s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    s = s.replace("﻿", "").replace("​", "")
    return re.sub(r"\s+", " ", s).strip()

_OPS = {
    "bare_inch": _op_bare_inch,
    "binary_gb": _op_binary_gb,
    "brand_suffix_strip": _op_brand_suffix_strip,
    "bus_qualifier_drop": _op_bus_qualifier_drop,
    "color_separator": _op_color_separator,
    "dim_to_mm": _op_dim_to_mm,
    "float_noise": _op_float_noise,
    "hyphen_collapse": _op_hyphen_collapse,
    "hyphen_to_space": _op_hyphen_to_space,
    "inch_notation": _op_inch,
    "interface_alias": _op_interface_alias,
    "kg_to_g": _op_kg_to_g,
    "markup": op_markup,
    "mb_to_gb": _op_mb_to_gb,
    "mojibake": op_mojibake,
    "nfkc": op_nfkc,
    "paren_unwrap": _op_paren_unwrap,
    "symbols": op_symbols,
    "tb_to_gb": _op_tb_to_gb,
    "title_case_from_uniform": op_title_case_from_uniform,
    "usb_if_fold": _op_usb_if_fold,
    "whitespace": op_whitespace,
}
_CTX_OPS = ['dim_to_mm', 'kg_to_g', 'mb_to_gb', 'tb_to_gb']
_NUMERIC = ['height_mm', 'length_mm', 'read_speed_mb_s', 'storage_gb', 'vram_gb', 'weight_g', 'width_mm', 'write_speed_mb_s']
_RULES = {
    "brand": [
        "markup",
        "mojibake",
        "nfkc",
        "whitespace",
        "symbols",
        "hyphen_collapse",
        "brand_suffix_strip"
    ],
    "model_number": [
        "markup",
        "mojibake",
        "nfkc",
        "whitespace",
        "symbols"
    ],
    "product_type": [
        "markup",
        "mojibake",
        "nfkc",
        "whitespace",
        "symbols"
    ],
    "bus_type": [
        "markup",
        "mojibake",
        "nfkc",
        "whitespace",
        "symbols",
        "usb_if_fold",
        "bus_qualifier_drop"
    ],
    "interface_type": [
        "markup",
        "mojibake",
        "nfkc",
        "whitespace",
        "symbols",
        "interface_alias"
    ],
    "form_factor": [
        "markup",
        "mojibake",
        "nfkc",
        "whitespace",
        "symbols",
        "inch_notation",
        "bare_inch",
        "paren_unwrap",
        "hyphen_to_space"
    ],
    "color": [
        "markup",
        "mojibake",
        "nfkc",
        "whitespace",
        "symbols",
        "title_case_from_uniform",
        "color_separator"
    ],
    "storage_connection_type": [
        "markup",
        "mojibake",
        "nfkc",
        "whitespace",
        "symbols",
        "inch_notation",
        "interface_alias"
    ],
    "memory_type": [
        "markup",
        "mojibake",
        "nfkc",
        "whitespace",
        "symbols"
    ],
    "vram_gb": [
        "float_noise",
        "mb_to_gb"
    ],
    "storage_gb": [
        "float_noise",
        "binary_gb",
        "tb_to_gb"
    ],
    "read_speed_mb_s": [
        "float_noise"
    ],
    "write_speed_mb_s": [
        "float_noise"
    ],
    "width_mm": [
        "float_noise",
        "dim_to_mm"
    ],
    "length_mm": [
        "float_noise",
        "dim_to_mm"
    ],
    "height_mm": [
        "float_noise",
        "dim_to_mm"
    ],
    "weight_g": [
        "float_noise",
        "kg_to_g"
    ]
}


def _normalize_cell(value, ops, ctx):
    if value is None or (isinstance(value, float) and value != value):
        return value
    s = str(value)
    for name in ops:
        s = _OPS[name](s, ctx) if name in _CTX_OPS else _OPS[name](s)
    return s.strip()


def normalize_products_frame(df):
    """Every rule of the table above, in order, on every cell of its attribute."""
    out = df.copy()
    types = out["product_type"].map(lambda v: str(v).strip().upper() if isinstance(v, str) else "") \
        if "product_type" in out.columns else None
    for attribute, ops in _RULES.items():
        if attribute not in out.columns or not ops:
            continue
        col = out[attribute].astype(object)
        values = [
            _normalize_cell(v, ops, {"attribute": attribute, "product_type": (types.iloc[i] if types is not None else "")})
            for i, v in enumerate(col)
        ]
        if attribute in _NUMERIC:
            # numeric attributes stay numeric (the comparators and the fusion resolvers read floats;
            # the CSV export writes them as the gold does, e.g. 1000.0)
            out[attribute] = pd.to_numeric(pd.Series(values, index=out.index), errors="coerce")
        else:
            out[attribute] = values
    out.attrs = dict(df.attrs)
    return out


