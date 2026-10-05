#!/usr/bin/env python
"""Table 8, reference-free rows "Record gain" and "Fusion ratio" from counts, without the panel's rounding.

The end-to-end panel (reproduction/scoring/e2e_panels.py) stores both values rounded to four decimals,
which can move the printed second decimal (Products P4 record gain: 69/812 = 0.08498, stored 0.085).
This script counts:
  output rows         rows of the fused table;
  multi-source rows   rows assembled from at least two distinct source records (_fusion_sources, or the
                      membership file if a system provides one);
  largest source      rows of the largest input source of the task.
Record gain = output rows / largest source rows - 1; fusion ratio = multi-source rows / output rows.

Run from the repository root:

    python reproduction/tables/table8/reference_free_counts.py --out rf_counts.csv
    python reproduction/tables/table8/reference_free_counts.py --no-p2 --system P4=/path/to/p4_outputs --out x.csv

The output has the columns of results/paper_tables/table8/reference_free_counts.csv.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DOMAINS = ("games", "companies", "music", "products", "papers")


def source_sizes(domain: str) -> dict[str, int]:
    data = REPO / "use cases" / domain / "base" / "input" / "data"
    out = {}
    for p in sorted(data.iterdir()):
        if not p.is_file() or p.name.endswith("_metadata.json"):
            continue
        if p.suffix == ".csv":
            out[p.stem] = len(pd.read_csv(p, dtype=str, usecols=[0]))
        elif p.suffix == ".jsonl":
            out[p.stem] = sum(1 for line in p.open(encoding="utf-8") if line.strip())
        elif p.suffix == ".json":
            out[p.stem] = len(json.loads(p.read_text(encoding="utf-8")))
    return out


def ids_of(value) -> set[str]:
    if isinstance(value, str):
        try:
            v = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            v = [x.strip() for x in value.split(",")]
    else:
        v = value
    return {str(x) for x in (v if isinstance(v, (list, tuple, set)) else [v]) if str(x) not in ("", "nan", "None")}


def counts(fused_path: Path, membership_path: Path | None) -> tuple[int, int]:
    fused = pd.read_csv(fused_path, dtype=str, usecols=lambda c: c in ("_id", "_fusion_sources"))
    if membership_path is not None and membership_path.is_file():
        m = pd.read_csv(membership_path, dtype=str)
        sizes = m.groupby("cluster_id")["record_id"].nunique()
        ids = set(fused["_id"].astype(str))
        multi = int((sizes[sizes.index.astype(str).isin(ids)] >= 2).sum())
    else:
        multi = int(sum(len(ids_of(v)) >= 2 for v in fused["_fusion_sources"]))
    return len(fused), multi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--system", action="append", default=[], metavar="NAME=DIR",
                    help="a further system: DIR/<domain>/fused.csv (+ membership.csv); repeatable")
    ap.add_argument("--no-p2", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    systems = [] if a.no_p2 else [("P2", None)]
    systems += [tuple(s.split("=", 1)) for s in a.system]
    rows = []
    for d in [x for x in a.domains.split(",") if x]:
        sizes = source_sizes(d)
        largest = max(sizes, key=sizes.get)
        for name, root in systems:
            if root is None:
                fused, mem = REPO / "results" / "best of breeds" / d / "baseline" / "fused.csv", None
            else:
                fused, mem = Path(root) / d / "fused.csv", Path(root) / d / "membership.csv"
            n, multi = counts(fused, mem)
            rows.append([d, name, n, multi, sizes[largest], largest])
            print(f"{d} {name}: record gain {n / sizes[largest] - 1:+.4f} ({n}/{sizes[largest]} - 1), "
                  f"fusion ratio {multi / n:.4f} ({multi}/{n})", flush=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["domain", "pipeline", "output_rows", "multi_source_rows", "largest_source_rows", "largest_source"])
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
