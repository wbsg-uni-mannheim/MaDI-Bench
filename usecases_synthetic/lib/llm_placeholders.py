"""Placeholder strings an LLM writes for "no value".

The K2 refill and interpolation clients asked the model for a record and
wrote its answers verbatim; the model fills unknown fields with "N/A", "NA",
"null" and the like, where the real records of a source leave the cell empty,
which made synthetic records recognisable. ``sanitize_llm_record`` turns such
scalar strings into ``None`` (an empty cell downstream). The string "None" is
left alone: some sources (papers) use it in their real records.
"""
from __future__ import annotations

from typing import Any

PLACEHOLDERS = frozenset({"n/a", "na", "null", "unknown", "-", "?", "nan", ""})


def is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in PLACEHOLDERS


def sanitize_llm_record(record: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``record`` with placeholder strings replaced by ``None``;
    lists keep their non-placeholder items; other values are unchanged."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        if is_placeholder(value):
            out[key] = None
        elif isinstance(value, list):
            out[key] = [v for v in value if not is_placeholder(v)]
        else:
            out[key] = value
    return out
