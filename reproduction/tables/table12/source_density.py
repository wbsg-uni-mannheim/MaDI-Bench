#!/usr/bin/env python
"""Table 12, columns Rows (k) and Density: row count and pooled cell density of each task's sources.

Rule (the same as for Table 1): identifier columns are excluded; a cell counts as missing if it is blank, a
placeholder string (null, None, nan, n/a, na, <NA>; case-insensitive), an empty list ('[]', '{}', or an empty
JSON list or object), or a zero duration. Zero counts as missing only in the columns that map to the target
attribute 'duration' (zero written in any format, e.g. '0', '0:00', '0s'). Density = filled cells / all
non-identifier cells, pooled over the sources of a task, in percent; Table 12 prints it as an integer and the
rows in thousands with one decimal.

The identifier and duration columns of each source are read from the task's schema-matching gold
(base: input/schemamatching/sm_mapping_gold.json; variants: input/schemamatching/sm_mapping.csv). Products
also excludes cluster_id if present.

Standard library only. Run from the repository root (the Papers tasks need about 16 GB of memory):

    python reproduction/tables/table12/source_density.py --out source_rows_density.csv
    python reproduction/tables/table12/source_density.py --tasks companies_base,companies_easy --out x.csv

The output has the columns of results/paper_tables/table12/source_rows_density.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

csv.field_size_limit(10 ** 9)
REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "use cases"
DOMAINS = ["games", "companies", "music", "products", "papers"]
TIERS = ["base", "easy", "medium", "hard"]
PLACEHOLDERS = {"null", "none", "nan", "n/a", "na", "<na>"}
NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def load(path: Path) -> tuple[list[str], list[dict]]:
    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            header = next(rd)
            return header, [dict(zip(header, r)) for r in rd]
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols, rows


def classify(v) -> str:
    """blank / placeholder / emptylist / zero / filled"""
    if v is None:
        return "blank"
    if isinstance(v, float) and math.isnan(v):
        return "blank"
    if isinstance(v, (list, tuple, dict)):
        return "emptylist" if len(v) == 0 else "filled"
    if isinstance(v, bool):
        return "filled"
    if isinstance(v, (int, float)):
        return "zero" if v == 0 else "filled"
    s = str(v).strip()
    if s == "":
        return "blank"
    if s.lower() in PLACEHOLDERS:
        return "placeholder"
    if s in ("[]", "{}"):
        return "emptylist"
    if NUM_RE.match(s) and float(s) == 0:
        return "zero"
    return "filled"


def zero_any_format(v) -> bool:
    """A zero quantity in any format ('0', '0:00', 'PT0S', ...): at least one digit and no digit 1-9."""
    if isinstance(v, bool) or v is None:
        return False
    if isinstance(v, (int, float)):
        return not (isinstance(v, float) and math.isnan(v)) and v == 0
    if isinstance(v, str):
        s = v.strip()
        return bool(re.search(r"\d", s)) and not re.search(r"[1-9]", s)
    return False


def source_files(task_dir: Path) -> list[Path]:
    data = task_dir / "input" / "data"
    return [p for p in sorted(data.iterdir())
            if p.is_file() and p.suffix in {".csv", ".json", ".jsonl"} and not p.name.endswith("_metadata.json")]


def column_roles(domain: str, tier: str, task_dir: Path, files: list[Path]) -> dict[str, dict[str, set[str]]]:
    """Identifier and duration columns per source file, from the schema-matching gold (read in place)."""
    sm = task_dir / "input" / "schemamatching"
    rows = []
    if tier == "base":
        g = json.loads((sm / "sm_mapping_gold.json").read_text(encoding="utf-8"))
        fname = {src: Path(rel).name for src, rel in g["source_files"].items()}
        for m in g["mappings"]:
            if m.get("label", True):
                rows.append((fname[m["source_dataset"]], m["source_column"], m["target_column"]))
    else:
        with open(sm / "sm_mapping.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((r["source_dataset"] + ".csv", r["source_column"], r["target_column"]))
    roles = {p.name: {"id": set(), "duration": set()} for p in files}
    for fn, col, tgt in rows:
        if fn not in roles:
            raise ValueError(f"{domain}/{tier}: schema-matching gold names an unknown source file {fn}")
        if tgt in ("id", "duration"):
            roles[fn][tgt].add(col)
    for p in files:
        if len(roles[p.name]["id"]) != 1:
            raise ValueError(f"{domain}/{tier}/{p.name}: expected one identifier column")
        if domain == "products":
            roles[p.name]["id"].add("cluster_id")
    return roles


def source_counts(path: Path, ids: set[str], durations: set[str]) -> tuple[int, int, int]:
    """(rows, non-identifier cells, filled cells) of one source file."""
    cols, rows = load(path)
    attrs = [c for c in cols if c not in ids]
    filled = 0
    for a in attrs:
        is_duration = a in durations
        for r in rows:
            v = r.get(a)
            c = classify(v)
            if c == "filled" and not (is_duration and zero_any_format(v)):
                filled += 1
            elif c == "zero" and not is_duration:
                filled += 1
    return len(rows), len(rows) * len(attrs), filled


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", default=None, help="comma-separated <domain>_<tier> (default: all 20 tasks)")
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    a = ap.parse_args(argv)
    tasks = [t for t in a.tasks.split(",") if t] if a.tasks else [f"{d}_{t}" for d in DOMAINS for t in TIERS]
    out_rows = []
    for task in tasks:
        domain, tier = task.split("_")
        task_dir = TASKS / domain / tier
        files = source_files(task_dir)
        roles = column_roles(domain, tier, task_dir, files)
        n_rows = n_cells = n_filled = 0
        for p in files:
            r, c, f = source_counts(p, roles[p.name]["id"], roles[p.name]["duration"])
            n_rows, n_cells, n_filled = n_rows + r, n_cells + c, n_filled + f
        pct = round(100 * n_filled / n_cells, 3)
        out_rows.append([domain, tier, n_rows, n_cells, n_filled, pct])
        print(f"{task}: rows {n_rows} ({n_rows / 1000:.1f}k), density {pct} % ({100 * n_filled / n_cells:.0f})",
              flush=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["domain", "tier", "rows", "non_id_cells", "filled_cells", "density_pct"])
        w.writerows(out_rows)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
