#!/usr/bin/env python
"""Table 8, ground-truth row "Fully-correct rate": the share of the 100 fusion test entities whose every
graded attribute value is correct, under the paper's fusion protocol (strict per-domain comparison rules,
v2 test gold, all gold entities in the denominator).

The steps of madi_bench.evaluation.score_fusion are replayed up to the alignment of the submitted rows with
the gold entities; then every graded gold cell (gold value present) is judged with the strict comparator of
its attribute. An entity is fully correct when the submission reaches it and all its graded cells are
correct; a graded cell in an attribute the submission lacks counts as wrong. The script also recomputes the
all-gold fusion accuracy from the same cells (it must equal madi_bench.evaluation.score_fusion, which is
checked, and Table 7).

Inputs: P2 from results/best of breeds/<domain>/baseline/fused.csv; further systems (e.g. P4) from a
submission folder <dir>/<domain>/ with fused.csv and optionally membership.csv.

Run from the repository root (Papers needs about 24 GB of memory):

    python reproduction/tables/table8/strict_fully_correct.py --domains companies,products --out fc.csv
    python reproduction/tables/table8/strict_fully_correct.py --system P4=/path/to/p4_outputs --out fc_p4.csv

Writes counts only (the columns of results/paper_tables/table8/gt_fully_correct_strict.csv). No API calls.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

for _k in list(os.environ):
    if _k.endswith("API_KEY"):
        os.environ.pop(_k)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
logging.disable(logging.CRITICAL)

import pandas as pd  # noqa: E402

from madi_bench.evaluation import fusion_rules as rules  # noqa: E402
from madi_bench.evaluation import fusion_score as fs  # noqa: E402
from madi_bench.evaluation.fusion_score import score_fusion  # noqa: E402

DOMAINS = ("games", "companies", "music", "products", "papers")
P2_ROOT = REPO / "results" / "best of breeds"


def entity_matrix(task_dir: Path, fused: pd.DataFrame, membership: pd.DataFrame | None) -> dict:
    """Replay of score_fusion up to the alignment, then the strict per-entity judgement."""
    domain, tier = fs.domain_of(task_dir), fs.tier_of(task_dir)
    if domain == "products" and tier == "base":
        fused = fs._products_base_fused_sources(fused)
        if membership is not None and not membership.empty:
            membership = fs._products_base_membership(membership)
    gold = fs.load_gold(task_dir, "test")
    attrs = [a for a in fs.target_schema_attributes(task_dir) if a in gold.columns]
    gold = gold[attrs + [fs.ANCHOR_ID]]
    exclude = fs.variant_only_record_ids(task_dir)
    sub = fs.derive_submission_anchor(fs.symmetric_prep(fused, domain), domain, membership, exclude)
    sub = sub.drop_duplicates(fs.ANCHOR_ID, keep="first")
    af, ag = fs._align(sub, gold)
    reached = set(ag[fs.ANCHOR_ID].astype(str)) if len(ag) else set()
    row_of = {str(a): i for i, a in enumerate(ag[fs.ANCHOR_ID].astype(str))} if len(ag) else {}
    graded = correct = n_full = 0
    for _, g in gold.iterrows():
        i = row_of.get(str(g[fs.ANCHOR_ID]))
        ok, cells = True, 0
        for a in attrs:
            expected = g[a]
            if rules.is_missing(expected):
                continue
            cells += 1
            graded += 1
            if i is None or a not in af.columns:
                ok = False
                continue
            got = af[a].iloc[i]
            if rules.is_missing(got):
                ok = False
                continue
            try:
                good = bool(rules.comparator_for(domain, a)(got, expected))
            except Exception:  # noqa: BLE001  (as the scorer: an unreadable value is wrong)
                good = False
            correct += good
            ok = ok and good
        if ok and cells:
            n_full += 1
    return {"gold_entities": len(gold), "gold_entities_reached": len(reached), "fully_correct_entities": n_full,
            "fully_correct_rate": n_full / len(gold), "graded_cells": graded, "correct_cells": correct,
            "fusion_accuracy_all_gold": correct / graded if graded else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--system", action="append", default=[], metavar="NAME=DIR",
                    help="a further system: DIR/<domain>/fused.csv (+ membership.csv); repeatable")
    ap.add_argument("--no-p2", action="store_true", help="skip P2 (results/best of breeds)")
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    a = ap.parse_args(argv)
    systems = [] if a.no_p2 else [("P2", None)]
    systems += [tuple(s.split("=", 1)) for s in a.system]
    rows = []
    for d in [x for x in a.domains.split(",") if x]:
        task_dir = REPO / "use cases" / d / "base"
        for name, root in systems:
            if root is None:
                fused = pd.read_csv(P2_ROOT / d / "baseline" / "fused.csv", dtype={"_id": str}, low_memory=False)
                mem = None
            else:
                sub = Path(root) / d
                fused = pd.read_csv(sub / "fused.csv", dtype={"_id": str}, low_memory=False)
                mem = pd.read_csv(sub / "membership.csv", dtype=str) if (sub / "membership.csv").is_file() else None
            rec = entity_matrix(task_dir, fused, mem)
            pub = score_fusion(task_dir, fused, membership=mem)["overall_accuracy_all_gold"]
            if abs(pub - rec["fusion_accuracy_all_gold"]) > 1e-12:
                raise AssertionError(f"{d} {name}: cell replay {rec['fusion_accuracy_all_gold']} != scorer {pub}")
            rows.append([d, name] + [rec[k] for k in ("gold_entities", "gold_entities_reached",
                                                       "fully_correct_entities", "fully_correct_rate",
                                                       "graded_cells", "correct_cells",
                                                       "fusion_accuracy_all_gold")])
            print(f"{d} {name}: fully correct {rec['fully_correct_entities']}/{rec['gold_entities']} "
                  f"= {rec['fully_correct_rate']:.2f}; fusion accuracy {rec['fusion_accuracy_all_gold']:.4f}",
                  flush=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["domain", "pipeline", "gold_entities", "gold_entities_reached", "fully_correct_entities",
                    "fully_correct_rate", "graded_cells", "correct_cells", "fusion_accuracy_all_gold"])
        w.writerows(rows)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
