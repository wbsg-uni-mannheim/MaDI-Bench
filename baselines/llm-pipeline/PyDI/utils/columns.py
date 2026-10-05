"""Column filtering helpers shared by pipeline stages."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def normalize_column_name(column: object) -> str:
    """Normalize a column name for policy comparisons."""
    return str(column).strip().lower().replace("-", "_").replace(" ", "_")


def is_excluded_column(column: object, excluded_columns: Iterable[str] | None) -> bool:
    """Return True when a column should be excluded by a manifest/runtime policy."""
    excluded = [normalize_column_name(c) for c in (excluded_columns or []) if str(c).strip()]
    if not excluded:
        return False

    normalized = normalize_column_name(column)
    parts = [part for part in normalized.split("_") if part]
    return any(
        normalized == excluded_name
        or normalized.startswith(f"{excluded_name}_")
        or normalized.endswith(f"_{excluded_name}")
        or excluded_name in parts
        for excluded_name in excluded
    )


def drop_excluded_columns(
    df: pd.DataFrame,
    excluded_columns: Iterable[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop excluded columns from a DataFrame and return (filtered_df, dropped_columns)."""
    drop_cols = [col for col in df.columns if is_excluded_column(col, excluded_columns)]
    if not drop_cols:
        return df, []
    return df.drop(columns=drop_cols), drop_cols


def filter_target_schema(
    target_schema: dict,
    excluded_columns: Iterable[str] | None,
) -> tuple[dict, list[str]]:
    """Return a copy of target_schema with excluded properties removed."""
    properties = target_schema.get("properties")
    if not isinstance(properties, dict):
        return target_schema, []

    filtered_properties = {
        name: spec
        for name, spec in properties.items()
        if not is_excluded_column(name, excluded_columns)
    }
    removed = [name for name in properties if name not in filtered_properties]
    if not removed:
        return target_schema, []

    filtered = dict(target_schema)
    filtered["properties"] = filtered_properties
    if isinstance(filtered.get("required"), list):
        filtered["required"] = [
            name
            for name in filtered["required"]
            if not is_excluded_column(name, excluded_columns)
        ]
    return filtered, removed
