#!/usr/bin/env python
"""Rebuild the printed cells of the paper's Tables 4, 5, 6, 8, 9 and 12 from the measurement files.

Reads the measurement files under results/paper_tables/ (written from the stored measurements),
results/scores_v2.json and the committee metrics under
usecases_synthetic/, applies the rounding and averaging rule of each table, and compares every cell
with the value printed in the paper (results/paper_tables/printed_cells.csv).

Writes results/paper_tables/table<N>/cells.csv: one row per printed cell with the printed value,
the unrounded value, the value as this script formats it, whether the two agree, and the source.
P3 (LLM-based pipeline) cells except in Table 5, the P4 runtime and cost cells of Table 9, and
Table 4's P4 and reference-matcher columns are listed with their printed value (status "as printed");
the outputs and scores of these runs are in results/llm pipeline/, results/claude code/ and
results/schema_matching_references/.

Standard library only; no task data or gold file is read.

    python reproduction/tables/assemble_cells.py            # from the repository root
    python reproduction/tables/assemble_cells.py --check    # exit code 1 if a computed cell differs
"""
from __future__ import annotations

import argparse
import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PT = REPO / "results" / "paper_tables"
SCORES = REPO / "results" / "scores_v2.json"
SYN = REPO / "usecases_synthetic"

DOMAINS = ["games", "companies", "music", "products", "papers"]
LABEL = {"games": "Games", "companies": "Companies", "music": "Music", "products": "Products",
         "papers": "Papers"}
TIERS = ["base", "easy", "medium", "hard"]
LVL = {"base": "Base", "easy": "Easy", "medium": "Med.", "hard": "Hard"}
P3_NOTE = "P3 (LLM-based pipeline): value of the P3 runs; their outputs and scores are in results/llm pipeline/"
P4_RUN_NOTE = "P4 (Claude Code) runtime and cost: value of the P4 runs; their records are in results/claude code/"
P3_T5_NOTE = "P3 normalized tables: results/llm pipeline/<domain>/baseline/schema_matching/"


# ----------------------------------------------------------------------------- helpers

def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ffmt(x: float, nd: int, signed: bool = False) -> str:
    """Python float formatting (the rule the table scripts used for Tables 5, 6, 8 and 12)."""
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def half_up(x: float, nd: int) -> str:
    """Half-up rounding of the shortest decimal form of x (Table 9, Table 12 normalization)."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{q:.{nd}f}"


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


class Table:
    def __init__(self, name: str, printed: dict):
        self.name = name
        self.printed = printed
        self.rows: list[dict] = []

    def add(self, row: str, column: str, value, formatted: str | None, source: str, note: str = "") -> None:
        p = self.printed.get((self.name, row, column))
        if p is None:
            raise KeyError(f"{self.name}: no printed cell {row!r} / {column!r}")
        status = "as printed" if formatted is None else ("match" if formatted == p else "DIFFERS")
        self.rows.append({"row": row, "column": column, "printed": p,
                          "value": "" if value is None else repr(float(value)) if not isinstance(value, str) else value,
                          "formatted": formatted or "", "status": status, "source": source, "note": note})

    def write(self) -> dict:
        missing = [k for k in self.printed if k[0] == self.name
                   and (k[1], k[2]) not in {(r["row"], r["column"]) for r in self.rows}]
        if missing:
            raise RuntimeError(f"{self.name}: printed cells without a row: {missing[:5]}")
        out = PT / self.name / "cells.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(self.rows)
        counts: dict[str, int] = {}
        for r in self.rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return counts


# ----------------------------------------------------------------------------- Table 4

def table4(t: Table, scores: dict) -> None:
    p1 = {r["domain"]: r for r in read_csv(PT / "table4" / "p1_schema_matching.csv")}
    for d in DOMAINS:
        r = p1[d]
        v = float(r["f1"])
        t.add(LABEL[d], "P1", v, ffmt(100 * v, 2), "table4/p1_schema_matching.csv f1",
              f"{r['correct']} correct of {r['predicted']} predicted, {r['gold_correspondences']} gold correspondences; {r['run']}")
        v = scores["results"]["p2_best_of_breeds"][f"{d}_base"]["stages"]["sm_f1"]
        t.add(LABEL[d], "P2", v, ffmt(100 * v, 2), "results/scores_v2.json p2_best_of_breeds stages.sm_f1")
        t.add(LABEL[d], "P3", None, None, "", P3_NOTE)
        t.add(LABEL[d], "P4", None, None, "",
              "P4 (Claude Code): value of the P4 runs; their outputs and scores are in results/claude code/")
        for c in ("Label-based", "Instance-based"):
            t.add(LABEL[d], c, None, None, "", "reference matchers: results/schema_matching_references/")


# ----------------------------------------------------------------------------- Table 5

def table5(t: Table, scores: dict) -> None:
    rows = {(r["domain"], r["pipeline"]): r for r in read_csv(PT / "table5" / "normalization_accuracy.csv")}
    for d in DOMAINS:
        for p in ("P1", "P2", "P3", "P4"):
            r = rows[(d, p)]
            v = int(r["correct"]) / int(r["cells"])
            src = f"table5/normalization_accuracy.csv ({r['correct']}/{r['cells']} cells)"
            note = r["note"]
            if p in ("P1", "P2", "P3"):
                s = scores["results"][{"P1": "p1_human", "P2": "p2_best_of_breeds",
                                       "P3": "p3_llm_pipeline"}[p]][f"{d}_base"]["normalization"]["accuracy"]
                same = abs(s - v) < 5e-5
                note = (note + "; " if note else "") + (
                    "equals results/scores_v2.json" if same else f"results/scores_v2.json has {s}")
            if p == "P3":
                t.add(LABEL[d], p, v, ffmt(100 * v, 2), src, (note + "; " if note else "") + P3_T5_NOTE)
            else:
                t.add(LABEL[d], p, v, ffmt(100 * v, 2), src, note)


# ----------------------------------------------------------------------------- Table 6

def table6(t: Table, scores: dict) -> None:
    rows = read_csv(PT / "table6" / "blocking_em_per_pair.csv")
    for d in DOMAINS:
        for metric, col in (("pc", "PC"), ("rr", "RR"), ("f1", "F1")):
            for p in ("P1", "P2", "P3", "P4"):
                cell = f"{col} {p}"
                if p == "P3":
                    t.add(LABEL[d], cell, None, None, "", P3_NOTE)
                    continue
                if p == "P2" and metric == "f1":
                    v = scores["results"]["p2_best_of_breeds"][f"{d}_base"]["stages"]["em_f1"]
                    pp = [float(r["value"]) for r in rows if (r["pipeline"], r["domain"], r["metric"]) == (p, d, "f1")]
                    t.add(LABEL[d], cell, v, ffmt(100 * v, 2),
                          "results/scores_v2.json p2_best_of_breeds stages.em_f1",
                          f"mean of the per-pair F1 in table6/blocking_em_per_pair.csv = {mean(pp):.6f}")
                    continue
                sel = [r for r in rows if (r["pipeline"], r["domain"], r["metric"]) == (p, d, metric)]
                v = mean([float(r["value"]) for r in sel])
                t.add(LABEL[d], cell, v, ffmt(100 * v, 2),
                      f"table6/blocking_em_per_pair.csv, unweighted mean over {len(sel)} row(s)",
                      sel[0]["note"] if sel and sel[0]["note"] else "")


# ----------------------------------------------------------------------------- Table 8

T8_RF = [("Record gain", "entity_gain", True), ("Density gain", "density_gain", True),
         ("Output density", "output_density", False), ("Fusion ratio", "fusion_ratio", False),
         ("Schema validity", "schema_validity", False)]
T8_REF = [("Entity recovery", "entity_recovery", False), ("Value drift", "value_drift", False),
          ("Value-density delta", "value_density_delta", True),
          ("Schema validity delta", "schema_validity_delta", True),
          ("BCubed precision", "bcubed_precision", False), ("BCubed recall", "bcubed_recall", False),
          ("BCubed F1", "bcubed_f1", False), ("Fusion accuracy", None, False),
          ("Fully-correct rate", "fully_correct_rate", False)]


def table8(t: Table, scores: dict) -> None:
    meas = jload(PT / "table8" / "e2e_measurements_20260926.json")["tasks"]
    prod = jload(PT / "table8" / "products_silver_20260930.json")
    games = jload(PT / "table8" / "games_gt_clusters_extended.json")
    fc = {(r["domain"], r["pipeline"]): r for r in read_csv(PT / "table8" / "gt_fully_correct_strict.csv")}
    rfc = {(r["domain"], r["pipeline"]): r for r in read_csv(PT / "table8" / "reference_free_counts.csv")}
    blocks = [("Reference-free", T8_RF), ("Silver", T8_REF), ("Ground truth", T8_REF)]
    for block, metrics in blocks:
        for name, key, signed in metrics:
            row = f"{block} / {name}"
            for p in ("P2", "P3", "P4"):
                vals = []
                for d in DOMAINS:
                    col = f"{LABEL[d]} {p}"
                    if p == "P3":
                        t.add(row, col, None, None, "", P3_NOTE)
                        continue
                    m = meas[d]
                    note = ""
                    if block == "Reference-free" and key in ("entity_gain", "fusion_ratio"):
                        # unrounded from counts; the panel value is the same number rounded to 4 decimals
                        c = rfc[(d, p)]
                        if key == "entity_gain":
                            v = int(c["output_rows"]) / int(c["largest_source_rows"]) - 1
                            src = (f"table8/reference_free_counts.csv: {c['output_rows']} output rows / "
                                   f"{c['largest_source_rows']} rows of the largest source ({c['largest_source']}) - 1")
                        else:
                            v = int(c["multi_source_rows"]) / int(c["output_rows"])
                            src = (f"table8/reference_free_counts.csv: {c['multi_source_rows']} multi-source rows / "
                                   f"{c['output_rows']} output rows")
                        panel = m[f"{p}_rf"][key]
                        if round(v, 4) != round(panel, 4):
                            raise AssertionError((d, p, key, v, panel))
                        note = f"panel value (4 decimals): {panel}"
                    elif block == "Reference-free":
                        v, src = m[f"{p}_rf"][key], f"table8/e2e_measurements_20260926.json {d}.{p}_rf.{key}"
                    elif block == "Silver":
                        k = "fusion_accuracy_micro" if key is None else key
                        if d == "products":
                            v, src = prod[f"{p}_silver"][k], f"table8/products_silver_20260930.json {p}_silver.{k}"
                        else:
                            v, src = m[f"{p}_silver"][k], f"table8/e2e_measurements_20260926.json {d}.{p}_silver.{k}"
                        if key is None:
                            note = "cell-weighted (micro) accuracy"
                    else:
                        if key is None:
                            if p == "P2":
                                v = scores["results"]["p2_best_of_breeds"][f"{d}_base"]["fusion"]["overall_accuracy_all_gold"]
                                src = "results/scores_v2.json p2_best_of_breeds fusion.overall_accuracy_all_gold (= Table 7)"
                            else:
                                v = m["P4_gt_fusion_accuracy_strict_v2"]["overall_accuracy_all_gold"]
                                src = f"table8/e2e_measurements_20260926.json {d}.P4_gt_fusion_accuracy_strict_v2"
                        elif key == "fully_correct_rate":
                            r = fc[(d, p)]
                            v = float(r["fully_correct_rate"])
                            src = (f"table8/gt_fully_correct_strict.csv ({r['fully_correct_entities']}/"
                                   f"{r['gold_entities']} entities, strict rules)")
                        elif d == "games" and key in ("entity_recovery", "bcubed_precision", "bcubed_recall", "bcubed_f1"):
                            v, src = games[p][key], f"table8/games_gt_clusters_extended.json {p}.{key}"
                            note = "games gold clusters extended with verified links"
                        else:
                            v, src = m[f"{p}_gt"][key], f"table8/e2e_measurements_20260926.json {d}.{p}_gt.{key}"
                    vals.append(float(v))
                    t.add(row, col, v, ffmt(float(v), 2, signed), src, note)
                col = f"Mean {p}"
                if p == "P3":
                    t.add(row, col, None, None, "", P3_NOTE)
                else:
                    v = mean(vals)
                    t.add(row, col, v, ffmt(v, 2, signed), "mean of the five unrounded per-task values")


# ----------------------------------------------------------------------------- Table 9

def table9(t: Table) -> None:
    p1 = {r["domain"]: r for r in read_csv(PT / "table9" / "p1_runtime.csv")}
    p2 = {r["domain"]: r for r in read_csv(PT / "table9" / "p2_runtime.csv")}
    v1, v2 = [], []
    for d in DOMAINS:
        a = float(p1[d]["corrected_s_file_stamps"])
        b = float(p2[d]["runtime_s"])
        v1.append(a)
        v2.append(b)
        t.add(LABEL[d], "P1", a, half_up(a, 0), "table9/p1_runtime.csv corrected_s_file_stamps",
              f"notebook timer {p1[d]['notebook_timer_s']} s minus the added export and scoring cells "
              f"(about {p1[d]['added_cells_s']} s; timed with the run folder's file timestamps)")
        t.add(LABEL[d], "P2", b, half_up(b, 0), "table9/p2_runtime.csv runtime_s",
              f"training {p2[d]['training_s']} s + pipeline without e2e panel {p2[d]['pipeline_wall_without_e2e_panel_s']} s")
        for c in ("P3", "P3 cost"):
            t.add(LABEL[d], c, None, None, "", P3_NOTE)
        for c in ("P4", "P4 cost"):
            t.add(LABEL[d], c, None, None, "", P4_RUN_NOTE)
    t.add("Mean", "P1", mean(v1), half_up(mean(v1), 0), "mean of the five unrounded P1 runtimes")
    t.add("Mean", "P2", mean(v2), half_up(mean(v2), 0), "mean of the five unrounded P2 runtimes")
    for c in ("P3", "P3 cost"):
        t.add("Mean", c, None, None, "", P3_NOTE)
    for c in ("P4", "P4 cost"):
        t.add("Mean", c, None, None, "", P4_RUN_NOTE)


# ----------------------------------------------------------------------------- Table 12

def committee_metrics(d: str, tier: str) -> tuple[dict, str]:
    if tier == "base":
        rel = f"usecases_synthetic/baselines/{d}/baseline_metrics.json"
    else:
        rel = f"usecases_synthetic/validation/{d}/{tier}/metrics.json"
    return jload(REPO / rel)["per_stage"], rel


def table12(t: Table) -> None:
    dens = {(r["domain"], r["tier"]): r for r in read_csv(PT / "table12" / "source_rows_density.csv")}
    fus = {(r["domain"], r["tier"]): r for r in read_csv(PT / "table12" / "fusion_committee_summary.csv")}
    for d in ("companies", "games", "music", "products", "papers"):          # the table's row order
        for tier in TIERS:
            row = f"{LABEL[d]} / {LVL[tier]}"
            x = dens[(d, tier)]
            rows = int(x["rows"])
            t.add(row, "Rows (k)", rows / 1000, ffmt(rows / 1000, 1), "table12/source_rows_density.csv rows")
            dp = float(x["density_pct"])
            t.add(row, "Density", dp, half_up(dp, 0), "table12/source_rows_density.csv density_pct",
                  f"{x['filled_cells']}/{x['non_id_cells']} non-identifier cells filled")
            ps, rel = committee_metrics(d, tier)
            sm = ps["sm"]["aggregated"]
            t.add(row, "SM mean", sm["macro_f1"], ffmt(100 * sm["macro_f1"], 1), f"{rel} sm.aggregated.macro_f1")
            t.add(row, "SM ceiling", sm["max_f1"], ffmt(100 * sm["max_f1"], 1), f"{rel} sm.aggregated.max_f1")
            nm = ps["norm"]["aggregated"]
            t.add(row, "Norm. mean", nm["mean_accuracy"], half_up_pct(nm["mean_accuracy"]),
                  f"{rel} norm.aggregated.mean_accuracy")
            t.add(row, "Norm. ceiling", nm["max_accuracy"], half_up_pct(nm["max_accuracy"]),
                  f"{rel} norm.aggregated.max_accuracy")
            em = ps["em_matching"]
            if tier == "base":
                mv, cv = em["aggregated"]["macro_f1"], em["aggregated"]["max_f1"]
                ms, cs = "em_matching.aggregated.macro_f1", "em_matching.aggregated.max_f1"
            else:
                mv = em["aggregated"]["macro_f1_baseline_model_on_baseline_test"]
                per = {k: v["metrics"]["f1_baseline_model_on_baseline_test"] for k, v in em["per_member"].items()}
                best = max(per, key=per.get)
                cv = per[best]
                ms = "em_matching.aggregated.macro_f1_baseline_model_on_baseline_test"
                cs = f"em_matching.per_member.{best}.metrics.f1_baseline_model_on_baseline_test (best member)"
            t.add(row, "EM mean", mv, ffmt(100 * mv, 1), f"{rel} {ms}")
            t.add(row, "EM ceiling", cv, ffmt(100 * cv, 1), f"{rel} {cs}")
            f = fus[(d, tier)]
            mf, cf = float(f["mean_8_members"]), float(f["ceiling_8_members"])
            t.add(row, "Fus. mean", mf, ffmt(100 * mf, 1), "table12/fusion_committee_summary.csv mean_8_members",
                  "8 non-LLM members, strict rules, v2 gold, all-gold cell-weighted accuracy")
            t.add(row, "Fus. ceiling", cf, ffmt(100 * cf, 1), "table12/fusion_committee_summary.csv ceiling_8_members",
                  f"best member: {f['best_member']}")


def half_up_pct(x: float) -> str:
    q = (Decimal(repr(float(x))) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{q:.1f}"


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit with 1 if a computed cell differs from the paper")
    a = ap.parse_args(argv)
    printed = {(r["table"], r["row"], r["column"]): r["printed"] for r in read_csv(PT / "printed_cells.csv")}
    scores = jload(SCORES)
    bad = 0
    for name, fn in (("table4", lambda t: table4(t, scores)), ("table5", lambda t: table5(t, scores)),
                     ("table6", lambda t: table6(t, scores)),
                     ("table8", lambda t: table8(t, scores)), ("table9", table9), ("table12", table12)):
        t = Table(name, printed)
        fn(t)
        counts = t.write()
        bad += counts.get("DIFFERS", 0)
        print(f"{name}: {counts}  -> results/paper_tables/{name}/cells.csv")
        for r in t.rows:
            if r["status"] == "DIFFERS":
                print(f"   differs: {r['row']} / {r['column']}: printed {r['printed']}, computed {r['formatted']}"
                      f" (value {r['value']}; {r['source']})")
    return 1 if (a.check and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
