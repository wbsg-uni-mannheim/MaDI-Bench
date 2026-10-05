#!/usr/bin/env python3
"""Schema-matching gold for the prepared base tasks.

prepare_inputs.py renames a few inputs, so the gold files are rewritten to the
prepared source dataset and column names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


CASES = ("products", "papers", "companies", "music", "games")
PRODUCT_GOLD_DATASET_ALIASES = {
    "products_1": "prod1",
    "products_2": "prod2",
    "products_3": "prod3",
    "products_4": "prod4",
}


def read_source_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [str(c) for c in pd.read_csv(path, nrows=0).columns]
    if suffix == ".json":
        return [str(c) for c in pd.read_json(path).columns]
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return [str(c) for c in json.loads(line).keys()]
    return []


def resolve_source_file(schema_dir: Path, relative_path: str) -> Path:
    return (schema_dir / relative_path).resolve()


def build_column_aliases(
    *,
    source_schema_dir: Path,
    local_data_dir: Path,
    source_files: dict[str, str],
    dataset_aliases: dict[str, str],
) -> dict[tuple[str, str], str]:
    aliases: dict[tuple[str, str], str] = {}
    for source_dataset, relative_path in source_files.items():
        local_dataset = dataset_aliases.get(source_dataset, source_dataset)
        source_path = resolve_source_file(source_schema_dir, relative_path)
        local_path = local_data_dir / f"{local_dataset}.csv"
        if not source_path.exists() or not local_path.exists():
            continue

        source_columns = read_source_columns(source_path)
        local_columns = [str(c) for c in pd.read_csv(local_path, nrows=0).columns]
        if not source_columns or not local_columns:
            continue

        first_source = source_columns[0]
        first_local = local_columns[0]
        if first_local == "id" and first_source != "id" and first_source not in local_columns:
            aliases[(source_dataset, first_source)] = "id"
            aliases[(local_dataset, first_source)] = "id"
    return aliases


def normalize_record(
    record: dict[str, Any],
    *,
    dataset_aliases: dict[str, str],
    column_aliases: dict[tuple[str, str], str],
) -> dict[str, Any]:
    out = dict(record)
    source_dataset = str(out.get("source_dataset", ""))
    source_column = out.get("source_column")
    local_dataset = dataset_aliases.get(source_dataset, source_dataset)
    out["source_dataset"] = local_dataset
    if source_column is not None:
        out["source_column"] = column_aliases.get(
            (source_dataset, str(source_column)),
            column_aliases.get((local_dataset, str(source_column)), source_column),
        )
    return out


def normalize_gold(case: str, source_path: Path, target_path: Path) -> None:
    data = json.loads(source_path.read_text(encoding="utf-8"))
    dataset_aliases = PRODUCT_GOLD_DATASET_ALIASES if case == "products" else {}
    source_files = data.get("source_files") if isinstance(data.get("source_files"), dict) else {}
    column_aliases = build_column_aliases(
        source_schema_dir=source_path.parent,
        local_data_dir=target_path.parents[1] / "data",
        source_files=source_files,
        dataset_aliases=dataset_aliases,
    )

    data["mappings"] = [
        normalize_record(
            record,
            dataset_aliases=dataset_aliases,
            column_aliases=column_aliases,
        )
        for record in data.get("mappings", [])
    ]
    unmapped = data.get("unmapped_source_columns")
    if isinstance(unmapped, list):
        data["unmapped_source_columns"] = [
            normalize_record(
                record,
                dataset_aliases=dataset_aliases,
                column_aliases=column_aliases,
            )
            for record in unmapped
        ]

    if isinstance(source_files, dict):
        normalized_source_files = {}
        for source_dataset, _relative_path in source_files.items():
            local_dataset = dataset_aliases.get(source_dataset, source_dataset)
            normalized_source_files[local_dataset] = f"../data/{local_dataset}.csv"
        data["source_files"] = normalized_source_files

    data.setdefault("notes", [])
    if isinstance(data["notes"], list):
        data["notes"].append(
            "Normalized for DI-Bench local dataset/identifier names by sync_schema_gold.py."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
