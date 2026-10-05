#!/usr/bin/env python3
"""Prepare the five MaDI-Bench base tasks for the P3 pipeline.

run_pipeline.py expects CSV source files, XML fusion validation and test files,
and entity-matching labels named ``left_2_right_split.csv``. The Papers and
Products base tasks use JSON/JSONL sources and other file names, so this script
copies the base task inputs from ``use cases/<domain>/base/input`` to
``DI-Bench/usecases/<domain>/input`` and converts them. The base manifests in
``manifests/`` read this folder; the variant manifests read ``use cases/``
directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from sync_schema_gold import normalize_gold


CASES = ("products", "papers", "companies", "music", "games")
CSV_CASES = ("companies", "music", "games")

PAIR_MANIFESTS: dict[str, list[dict[str, str]]] = {
    "products": [
        {"left": "prod1", "right": "prod2"},
        {"left": "prod1", "right": "prod3"},
        {"left": "prod1", "right": "prod4"},
    ],
    "papers": [
        {"left": "dblp", "right": "crossref"},
        {"left": "dblp", "right": "open_alex"},
    ],
    "companies": [
        {"left": "forbes", "right": "dbpedia"},
        {"left": "forbes", "right": "fullcontact"},
    ],
    "music": [
        {"left": "musicbrainz", "right": "discogs"},
        {"left": "musicbrainz", "right": "lastfm"},
    ],
    "games": [
        {"left": "dbpedia", "right": "metacritic"},
        {"left": "dbpedia", "right": "sales"},
    ],
}

PRODUCT_DATASETS = {
    "dataset_1": "prod1",
    "dataset_2": "prod2",
    "dataset_3": "prod3",
    "dataset_4": "prod4",
}

PRODUCT_LABEL_PREFIXES = {
    "prod1_to_prod2": "prod1_2_prod2",
    "prod1_to_prod3": "prod1_2_prod3",
    "prod1_to_prod4": "prod1_2_prod4",
}

PAPER_LABEL_PREFIXES = {
    "dblp_crossref": "dblp_2_crossref",
    "dblp_openalex": "dblp_2_open_alex",
}


def reset_output(out: Path, force: bool) -> None:
    usecases_out = out / "usecases"
    if usecases_out.exists():
        if not force:
            raise FileExistsError(f"{usecases_out} exists; pass --force to replace it")
        shutil.rmtree(usecases_out)
    usecases_out.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def scalarize(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].map(scalarize)
    df.to_csv(path, index=False)


def convert_product_data(src_input: Path, dst_input: Path) -> None:
    raw_dir = dst_input / "source_raw" / "data"
    data_dir = dst_input / "data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in PRODUCT_DATASETS.items():
        json_path = src_input / "data" / f"{src_name}.json"
        metadata_path = src_input / "data" / f"{src_name}_metadata.json"
        shutil.copy2(json_path, raw_dir / json_path.name)
        if metadata_path.exists():
            shutil.copy2(metadata_path, data_dir / f"{dst_name}_metadata.json")
            shutil.copy2(metadata_path, raw_dir / metadata_path.name)
        df = pd.read_json(json_path)
        write_csv(df, data_dir / f"{dst_name}.csv")


def convert_paper_data(src_input: Path, dst_input: Path) -> dict[str, dict[str, str]]:
    raw_dir = dst_input / "source_raw" / "data"
    data_dir = dst_input / "data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    doi_to_ids: dict[str, dict[str, str]] = {}
    for name in ("dblp", "crossref", "open_alex"):
        jsonl_path = src_input / "data" / f"{name}.jsonl"
        metadata_path = src_input / "data" / f"{name}_metadata.json"
        shutil.copy2(jsonl_path, raw_dir / jsonl_path.name)
        if metadata_path.exists():
            shutil.copy2(metadata_path, data_dir / metadata_path.name)
            shutil.copy2(metadata_path, raw_dir / metadata_path.name)

        records: list[dict[str, Any]] = []
        with jsonl_path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = f"{name}-{idx:05d}"
                record = {"id": record_id, **record}
                doi = str(record.get("doi", "")).strip()
                if doi:
                    doi_to_ids.setdefault(doi.lower(), {})[name] = record_id
                records.append(record)
        write_csv(pd.DataFrame(records), data_dir / f"{name}.csv")
    return doi_to_ids


def normalize_label_value(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y"}:
        return "TRUE"
    if text in {"0", "0.0", "false", "f", "no", "n"}:
        return "FALSE"
    return str(value).strip().upper()


def normalize_label_file(src: Path, dst: Path, *, left_header: str = "id1", right_header: str = "id2") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return

    header = [h.strip() for h in rows[0]]
    has_header = any(h.lower() in {"id1", "label"} or h.lower().startswith("id_") for h in header)
    data_rows = rows[1:] if has_header else rows

    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        extra_header = header[3:] if has_header and len(header) > 3 else []
        writer.writerow([left_header, right_header, "label", *extra_header])
        for row in data_rows:
            if len(row) < 3:
                continue
            writer.writerow([row[0], row[1], normalize_label_value(row[2]), *row[3:]])


def copy_product_labels(src_input: Path, dst_input: Path) -> None:
    dst_dir = dst_input / "entitymatching"
    src_dir = src_input / "entitymatching"
    if not src_dir.exists():
        src_dir = src_input / "entity_matching_gt"
    for src_path in sorted(src_dir.glob("*.csv")):
        name = src_path.stem
        dst_name = None
        for old_prefix, new_prefix in PRODUCT_LABEL_PREFIXES.items():
            if name.startswith(old_prefix):
                dst_name = new_prefix + name[len(old_prefix):] + ".csv"
                break
        if dst_name is None:
            dst_name = src_path.name
        normalize_label_file(src_path, dst_dir / dst_name)


def copy_paper_labels(src_input: Path, dst_input: Path) -> None:
    dst_dir = dst_input / "entitymatching"
    src_dir = src_input / "entitymatching"
    for src_path in sorted(src_dir.glob("*.csv")):
        name = src_path.stem
        dst_name = None
        for old_prefix, new_prefix in PAPER_LABEL_PREFIXES.items():
            if name.startswith(old_prefix):
                dst_name = new_prefix + name[len(old_prefix):] + ".csv"
                break
        if dst_name is None:
            dst_name = src_path.name
        normalize_label_file(src_path, dst_dir / dst_name)


def copy_standard_labels(src_input: Path, dst_input: Path) -> None:
    src_dir = src_input / "entitymatching"
    dst_dir = dst_input / "entitymatching"
    for src_path in sorted(src_dir.glob("*.csv")):
        normalize_label_file(src_path, dst_dir / src_path.name)
    copy_tree(src_dir / "old", dst_dir / "old")


def copy_standard_data(src_input: Path, dst_input: Path) -> None:
    src_dir = src_input / "data"
    dst_dir = dst_input / "data"
    raw_dir = dst_input / "source_raw" / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for src_path in sorted(src_dir.iterdir()):
        if src_path.is_dir():
            copy_tree(src_path, raw_dir / src_path.name)
            continue
        shutil.copy2(src_path, raw_dir / src_path.name)
        if src_path.suffix.lower() == ".csv":
            df = pd.read_csv(src_path)
            if "id" not in df.columns and len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: "id"})
            write_csv(df, dst_dir / src_path.name)
        else:
            shutil.copy2(src_path, dst_dir / src_path.name)


def valid_xml_tag(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)
    if not out or out[0].isdigit() or out[0] in {"-", "."}:
        out = f"field_{out}"
    return out


def row_to_xml(parent: ET.Element, row: dict[str, Any], fields: list[str]) -> None:
    for field in fields:
        value = row.get(field, "")
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        text = scalarize(value)
        if str(text).strip() == "":
            continue
        child = ET.SubElement(parent, valid_xml_tag(field))
        child.text = str(text)


def write_xml(records: list[dict[str, Any]], path: Path, *, root_tag: str, record_tag: str, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element(root_tag)
    for row in records:
        rec = ET.SubElement(root, record_tag)
        row_to_xml(rec, row, fields)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def schema_fields(schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    props = schema.get("properties", {})
    return [field for field in props.keys() if field != "id"]


def rewrite_schema_taxonomy_paths(schema_path: Path) -> None:
    """Make PyDI taxonomy references relative to the copied schema file."""
    if not schema_path.exists():
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    changed = False

    def visit(node: Any) -> None:
        nonlocal changed
        if isinstance(node, dict):
            taxonomy_path = node.get("x-pydi-taxonomy")
            if isinstance(taxonomy_path, str):
                path = Path(taxonomy_path)
                if not path.is_absolute() and taxonomy_path.startswith("input/schemamatching/"):
                    node["x-pydi-taxonomy"] = path.name
                    changed = True
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    if changed:
        schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


def convert_product_fusion(src_input: Path, dst_input: Path) -> None:
    fields = schema_fields(dst_input / "schemamatching" / "target_schema.json")
    fusion_dir = dst_input / "fusion"
    raw_dir = dst_input / "source_raw" / "fusion"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (
        ("fusion_validation_set.csv", "validation_set.xml"),
        ("fusion_test_set.csv", "test_set.xml"),
    ):
        src_path = src_input / "fusion" / src_name
        if not src_path.exists():
            continue
        shutil.copy2(src_path, raw_dir / src_name)
        df = pd.read_csv(src_path)
        records = []
        for row in df.to_dict(orient="records"):
            record = {"id": row.get("id_left", "")}
            record.update(row)
            records.append(record)
        write_xml(records, fusion_dir / dst_name, root_tag="products", record_tag="product", fields=["id", *fields])


def convert_paper_fusion(src_input: Path, dst_input: Path, doi_to_ids: dict[str, dict[str, str]]) -> None:
    fields = schema_fields(dst_input / "schemamatching" / "target_schema.json")
    fusion_dir = dst_input / "fusion"
    raw_dir = dst_input / "source_raw" / "fusion"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (
        ("fusion_val.jsonl", "validation_set.xml"),
        ("fusion_test.jsonl", "test_set.xml"),
    ):
        src_path = src_input / "fusion" / src_name
        if not src_path.exists():
            continue
        shutil.copy2(src_path, raw_dir / src_name)
        records = []
        with src_path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                doi = str(row.get("doi", "")).strip().lower()
                ids = doi_to_ids.get(doi, {})
                record_id = ids.get("dblp") or ids.get("crossref") or ids.get("open_alex") or f"paper-gold-{idx:05d}"
                record = {"id": record_id}
                record.update(row)
                records.append(record)
        write_xml(records, fusion_dir / dst_name, root_tag="papers", record_tag="paper", fields=["id", *fields])


def copy_standard_case(src_input: Path, dst_input: Path) -> None:
    copy_standard_data(src_input, dst_input)
    copy_tree(src_input / "schemamatching", dst_input / "schemamatching")
    rewrite_schema_taxonomy_paths(dst_input / "schemamatching" / "target_schema.json")
    copy_tree(src_input / "fusion", dst_input / "fusion")
    copy_standard_labels(src_input, dst_input)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=REPO_ROOT.parents[1] / "use cases",
        help="MaDI-Bench task folder (default: the repository's use cases/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "DI-Bench",
        help="Output folder (default: baselines/llm-pipeline/DI-Bench).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing <out>/usecases content.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = args.tasks.resolve()
    out = args.out.resolve()
    reset_output(out, args.force)

    for case in CASES:
        src_input = tasks / case / "base" / "input"
        dst_input = out / "usecases" / case / "input"
        dst_input.mkdir(parents=True, exist_ok=True)

        if case in CSV_CASES:
            copy_standard_case(src_input, dst_input)
        elif case == "products":
            copy_tree(src_input / "schemamatching", dst_input / "schemamatching")
            rewrite_schema_taxonomy_paths(dst_input / "schemamatching" / "target_schema.json")
            convert_product_data(src_input, dst_input)
            copy_product_labels(src_input, dst_input)
            convert_product_fusion(src_input, dst_input)
        elif case == "papers":
            copy_tree(src_input / "schemamatching", dst_input / "schemamatching")
            rewrite_schema_taxonomy_paths(dst_input / "schemamatching" / "target_schema.json")
            doi_to_ids = convert_paper_data(src_input, dst_input)
            copy_paper_labels(src_input, dst_input)
            convert_paper_fusion(src_input, dst_input, doi_to_ids)

        gold = src_input / "schemamatching" / "sm_mapping_gold.json"
        normalize_gold(case, gold, dst_input / "schemamatching" / "sm_mapping_gold.json")

    print(f"Prepared the base tasks from {tasks} in {out / 'usecases'}")


if __name__ == "__main__":
    main()
