#!/usr/bin/env python
"""Table 6, P2 columns PC and RR, from the test_rescore.json files that p2_blocking_only.py writes.

For each task, the blockers P2 selected on the validation split (the per-pair winners of the stored stage
record, results/best of breeds/<domain>/baseline/stage_3_em_blocking_selection.json) are taken from the
per-pair x blocker table of p2_blocking_only.py:
  PC = unweighted mean over the source pairs of |test positives covered| / |test positives|
       (the shipped *_test.csv of the base task);
  RR = mean over the source pairs of 1 - |candidates| / (|left source| x |right source|).

Games: the stored P2 run selected its blockers on the test split, and the paper prints the values of the
selected blocker (standard_blocker) from the stored stage record. For Games this script prints those values
and, for information, the values of its p2_blocking_only.py run.

Standard library only. Run from the repository root:

    python reproduction/tables/table6/p2_blocking_values.py --runs p2_blocking_runs --out p2_blocking_rescore.csv

--runs holds one run folder per task (any folder name containing the domain, each with test_rescore.json).
The output has the columns of results/paper_tables/table6/p2_blocking_rescore.csv; the means are printed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STORED = REPO / "results" / "best of breeds"
DOMAINS = ("games", "companies", "music", "products", "papers")


def find_rescore(runs: Path, domain: str) -> Path | None:
    hits = sorted(p for p in runs.glob("*/test_rescore.json") if domain in p.parent.name)
    return hits[-1] if hits else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, required=True, help="folder with the p2_blocking_only.py run folders")
    ap.add_argument("--out", type=Path, required=True, help="output CSV (per pair x blocker)")
    a = ap.parse_args(argv)
    rows = []
    for d in DOMAINS:
        s3 = json.loads((STORED / d / "baseline" / "stage_3_em_blocking_selection.json").read_text())
        stored_winners = s3["notes"]["per_pair_winner"]
        if d == "games":
            w = sorted(set(stored_winners.values()))
            if len(w) == 1:
                w = w[0]
                print(f"games: stored record, winner {w}: PC {100 * s3['per_member_val'][w]:.2f} "
                      f"RR {100 * s3['notes']['per_member_reduction_ratio_val'][w]:.2f} (printed in Table 6)")
        path = find_rescore(a.runs, d)
        if path is None:
            print(f"{d}: no test_rescore.json under {a.runs}")
            continue
        r = json.loads(path.read_text())
        pcs, rrs = [], []
        for pair, blockers in r["per_pair"].items():
            for name, b in blockers.items():
                t = b.get("test") or {}
                selected = stored_winners.get(pair) == name
                if d != "games":          # Games is printed from the stored record (see above)
                    rows.append([d, pair, name, int(selected), b["selection_split_covered"],
                                 b["selection_split_positives"], b["selection_split_pair_recall"], t.get("test_covered"),
                                 t.get("test_positives"), t.get("test_pair_recall"), b["candidate_count"],
                                 b["full_space"], b["reduction_ratio"]])
                if selected:
                    pcs.append(t["test_pair_recall"])
                    rrs.append(b["reduction_ratio"])
        label = "p2_blocking_only.py run, for information" if d == "games" else "stored winners"
        print(f"{d} ({label}): PC {100 * sum(pcs) / len(pcs):.2f} RR {100 * sum(rrs) / len(rrs):.2f} "
              f"over {len(pcs)} pairs")
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["domain", "source_pair", "blocker", "selected", "val_covered", "val_positives",
                    "val_pair_completeness", "test_covered", "test_positives", "test_pair_completeness", "candidates",
                    "full_space", "reduction_ratio"])
        w.writerows(rows)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
