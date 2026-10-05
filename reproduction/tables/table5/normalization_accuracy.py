#!/usr/bin/env python
"""Table 5: normalization accuracy of the pipelines on the normalization test sets of the base tasks.

Each pipeline's normalized source tables are scored with madi_bench.evaluation.score_normalization on
'use cases/<domain>/base/input/normalization/test.csv' (read in place): accuracy over all test cells, the
cells that need a transformation and the cells whose source value is already correct (identity). Table 5
prints the accuracy over all cells in percent.

Where the normalized tables are:
  P1  'use cases/<domain>/base/output/normalization/<source>.csv' (the shipped P1 outputs);
  P2  exported by reproduction/scoring/p2_norm_replay.py; pass its output root with --p2-root (its
      <domain>_baseline/tables/ folders);
  P3  'results/llm pipeline/<domain>/baseline/schema_matching/<source>.csv';
  P4  --p4 DOMAIN=DIR scores a folder of normalized CSV tables, and
      --p4-workspace DOMAIN=DIR first exports the normalized frame of a P4 (Claude Code) run workspace the way
      the paper did (see export_p4_workspace).

Run from the repository root:

    python reproduction/tables/table5/normalization_accuracy.py --out table5.csv
    python reproduction/tables/table5/normalization_accuracy.py --p2-root /tmp/p2_norm_replay --out table5.csv

The output has the columns of results/paper_tables/table5/normalization_accuracy.csv. No API calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOMAINS = ("games", "companies", "music", "products", "papers")

# P4 workspaces: the file holding the normalized records and the column that holds each target attribute.
# Only renames; no cleaning, taxonomy mapping or fusion is applied (games keeps its retained source values).
P4_FILES = {"games": "normalized.csv", "companies": "norm.csv", "music": "records.pkl",
            "products": "s2_normalized.pkl", "papers": "s2_normalized.pkl"}
P4_ALIASES = {
    "games": {"platform": "platform_raw", "genres": "genres_raw"},
    "companies": {"name": "display"},
    "music": {"release-date": "date", "release-country": "country", "label": "labels"},
    "products": {"brand": "n_brand", "bus_type": "n_bus", "color": "n_color", "form_factor": "n_ff",
                 "interface_type": "n_iface", "memory_type": "n_mem", "storage_connection_type": "n_conn",
                 "storage_gb": "n_storage_gb", "vram_gb": "n_vram_gb", "weight_g": "n_weight_g"},
    "papers": {},
}


def export_p4_workspace(domain: str, workspace: Path, dest: Path) -> Path:
    """Write the normalized records of a P4 run workspace (work/state/<file>) as one CSV with the
    target attribute names of the normalization test set."""
    import pandas as pd

    task_dir = REPO / "use cases" / domain / "base"
    with (task_dir / "input" / "normalization" / "test.csv").open(newline="", encoding="utf-8") as f:
        attributes = sorted({r["attribute"] for r in csv.DictReader(f)})
    source = workspace / "work" / "state" / P4_FILES[domain]
    frame = (pd.read_csv(source, dtype=str, keep_default_na=False) if source.suffix == ".csv"
             else pd.read_pickle(source))
    id_column = "record_id" if "record_id" in frame.columns else "id"

    def serial(v):
        if isinstance(v, (list, tuple)):
            return json.dumps(v, ensure_ascii=False)
        return "" if pd.isna(v) else str(v)

    export = pd.DataFrame({"source": frame["source"], "record_id": frame[id_column].map(serial)})
    for attr in attributes:
        export[attr] = frame[P4_ALIASES[domain].get(attr, attr)].map(serial)
    dest.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(dest, index=False)
    return dest


def score(domain: str, tables: Path) -> dict:
    from madi_bench.evaluation.normalization_score import NormalizedTables, score_normalization

    task_dir = REPO / "use cases" / domain / "base"
    s = score_normalization(task_dir, NormalizedTables(tables))
    m = s["metrics"]
    t, i = m.get("category:transformations", {"n": 0, "correct": 0}), m.get("category:identity", {"n": 0, "correct": 0})
    assert m["all"]["n"] == t["n"] + i["n"], (domain, m["all"], t, i)
    return {"correct": m["all"]["correct"], "cells": m["all"]["n"], "accuracy": round(m["all"]["correct"] / m["all"]["n"], 6),
            "transformation_correct": t["correct"], "transformation_cells": t["n"], "identity_correct": i["correct"],
            "identity_cells": i["n"], "record_coverage": s["record_coverage"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--p2-root", type=Path, default=None, help="output root of reproduction/scoring/p2_norm_replay.py")
    ap.add_argument("--p4", action="append", default=[], metavar="DOMAIN=DIR", help="P4 normalized tables")
    ap.add_argument("--p4-workspace", action="append", default=[], metavar="DOMAIN=DIR", help="P4 run workspace")
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    a = ap.parse_args(argv)
    p4 = dict(s.split("=", 1) for s in a.p4)
    p4_ws = dict(s.split("=", 1) for s in a.p4_workspace)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for d in [x for x in a.domains.split(",") if x]:
            folders = {"P1": REPO / "use cases" / d / "base" / "output" / "normalization",
                       "P2": (a.p2_root / f"{d}_baseline" / "tables") if a.p2_root else None,
                       "P3": REPO / "results" / "llm pipeline" / d / "baseline" / "schema_matching",
                       "P4": Path(p4[d]) if d in p4 else None}
            if d in p4_ws:
                folders["P4"] = export_p4_workspace(d, Path(p4_ws[d]), Path(tmp) / d / f"p4_{d}_normalization.csv")
            for name, folder in folders.items():
                if folder is None or not folder.exists():
                    print(f"{d} {name}: no normalized tables, skipped")
                    continue
                r = score(d, folder)
                rows.append([d, name] + list(r.values()) + [""])
                print(f"{d} {name}: {r['correct']}/{r['cells']} = {100 * r['correct'] / r['cells']:.2f} "
                      f"(record coverage {r['record_coverage']})", flush=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["domain", "pipeline", "correct", "cells", "accuracy", "transformation_correct",
                    "transformation_cells", "identity_correct", "identity_cells", "record_coverage", "note"])
        w.writerows(rows)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
