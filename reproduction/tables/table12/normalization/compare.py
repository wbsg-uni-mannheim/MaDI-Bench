#!/usr/bin/env python3
"""Table 12, columns Norm.: build the normalization block of the committee metrics files from a run of
``measure.py`` (and ``llm_member.py``) and compare it value by value with the shipped files.

The committee metrics files (``usecases_synthetic/baselines/<domain>/baseline_metrics.json`` for the base
task, ``usecases_synthetic/validation/<domain>/<level>/metrics.json`` for the variants) hold the
normalization committee in ``per_stage.norm``, built from runs of the two scripts of this folder for all 20
tasks (2026-09-28). This script builds the block from such a run in the same way:

* per member (``rule_per_attribute_optimal``, ``llm_only``, ``passthrough``): ``accuracy`` = correct test
  cells / all test cells (categories identity and transformations), ``accuracy_transformations`` (the
  categories knowledge and normalization), the counts, ``record_coverage``, the scores per category and per
  attribute;
* ``aggregated``: ``mean_accuracy`` = sum of the three members' correct cells / (3 x test cells),
  ``max_accuracy``, ``min_accuracy``, the same for the transformation cells, and the best member (the first
  maximum in the order llm_only, rule_per_attribute_optimal, passthrough);
* variants: every numeric value also as ``<key>_baseline`` (the base task's value) and ``<key>_delta``
  (variant minus base), as ``validate_variant`` writes them;
* aliases for the generator's own report readers, which read the stage's ``macro_f1``: ``macro_f1`` / ``f1``
  = accuracy, ``max_f1`` = ``max_accuracy`` and so on (accuracies, not F1).

Table 12 prints ``mean_accuracy`` (Norm. mean) and ``max_accuracy`` (Norm. ceiling) in percent with one
decimal, rounded half up.

Without an ``llm_member.py`` result for a task (a paid run needs an OpenAI key), ``llm_only`` is taken from the
shipped file (status ``from_shipped``): the two re-run members are compared, and the mean and the ceiling are
computed from them and the shipped ``llm_only`` counts.

Outputs in ``<out>/compare/``: ``norm_values.csv`` (every compared value: task, field, re-run, shipped,
status), ``table12_norm_cells.csv`` (Norm. mean and ceiling, re-run and shipped, as printed) and
``norm_blocks/<task>.json`` (the built blocks). Statuses: ``equal``, ``differs``, ``from_shipped``,
``only_in_rerun``, ``only_in_shipped``, ``no_base_rerun`` (a variant's ``_baseline``/``_delta`` value when the
base task was not re-run). Standard library only; runs in seconds::

    python reproduction/tables/table12/normalization/compare.py --out /tmp/norm12
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
SYN = REPO / "usecases_synthetic"
DOMAINS = ("companies", "games", "music", "papers", "products")
LEVELS = ("baseline", "easy", "medium", "hard")
ROSTER = ["rule_per_attribute_optimal", "llm_only", "passthrough"]
BEST_ORDER = ["llm_only", "rule_per_attribute_optimal", "passthrough"]
SURFACE = "normalization_test_set"
TOL = 1e-12


def tier(level: str) -> str:
    return "base" if level == "baseline" else level


def metrics_path(domain: str, level: str) -> Path:
    if level == "baseline":
        return SYN / "baselines" / domain / "baseline_metrics.json"
    return SYN / "validation" / domain / level / "metrics.json"


def half_up_pct(x: float) -> str:
    q = (Decimal(repr(float(x))) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{q:.1f}"


def is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# block construction (the layout of per_stage.norm in the committee metrics files)
# ---------------------------------------------------------------------------


def cat_counts(per_category: dict) -> tuple[int, int, int, int]:
    """(cells, correct, transformation cells, correct transformation cells); every test cell is either an
    identity cell or a transformation cell (knowledge or normalization)."""
    ide = per_category.get("identity") or {"n": 0, "correct": 0}
    tra = per_category.get("transformations") or {"n": 0, "correct": 0}
    return ide["n"] + tra["n"], ide["correct"] + tra["correct"], tra["n"], tra["correct"]


def llm_from_shipped(block: dict) -> dict:
    """The shipped llm_only member in the shape of a re-run record."""
    pm = block["per_member"]["llm_only"]
    per_attribute = {a: {"n": e["n_cells"], "correct": round(e["llm_only"] * e["n_cells"])}
                     for a, e in block["per_attribute"].items() if "llm_only" in e}
    return {"per_category": pm["notes"]["per_category"], "per_attribute": per_attribute,
            "record_coverage": pm["metrics"]["record_coverage"], "n_cells": pm["metrics"]["n_cells"],
            "llm_scope": pm["notes"].get("llm_scope"), "api_errors_open": pm["notes"].get("api_errors_open"),
            "model_name": pm["notes"].get("model_name")}


def build_block(domain: str, level: str, level_info: dict, recs: dict[str, dict], source: dict) -> dict:
    task = f"{domain}_{tier(level)}"
    n_cells = recs["passthrough"]["n_cells"]
    members: dict[str, dict] = {}
    for m in ROSTER:
        rec = recs[m]
        n, c, nt, ct = cat_counts(rec["per_category"])
        if n != n_cells or rec["n_cells"] != n_cells:
            raise SystemExit(f"{task}/{m}: {n} / {rec['n_cells']} cells, expected {n_cells}")
        members[m] = {"rec": rec, "n": n, "c": c, "nt": nt, "ct": ct, "acc": c / n,
                      "acct": (ct / nt if nt else 0.0)}
    accs = {m: members[m]["acc"] for m in ROSTER}
    mean = float(Fraction(sum(members[m]["c"] for m in ROSTER), 3 * n_cells))
    mx, mn = max(accs.values()), min(accs.values())
    best = next(m for m in BEST_ORDER if accs[m] == mx)
    tied = sorted(m for m in ROSTER if accs[m] == mx and m != best)
    transf = {m: members[m]["acct"] for m in ROSTER}

    per_member: dict = {}
    for m in ROSTER:
        v = members[m]
        rec = v["rec"]
        metrics = {
            "accuracy": v["acc"],
            "accuracy_transformations": v["acct"],
            "n_cells": v["n"],
            "n_correct": v["c"],
            "n_transformation_cells": v["nt"],
            "n_transformations_correct": v["ct"],
            "record_coverage": rec["record_coverage"],
            "macro_f1": v["acc"],          # reader aliases: accuracies, not F1
            "f1": v["acc"],
            "scoring_surface": SURFACE,
        }
        notes: dict = {"per_category": rec["per_category"]}
        if m == "rule_per_attribute_optimal":
            notes["selection_map"] = level_info.get("selection_map") or {}
            notes["eligible_attributes"] = level_info.get("eligible_attributes") or []
            notes["rule_scope_of_test_attributes"] = level_info.get("rule_scope_of_test_attributes") or {}
            notes["lock_written_this_level"] = level_info.get("lock_written_this_level")
        elif m == "llm_only":
            notes["model_name"] = rec.get("model_name") or "gpt-5.4-mini"
            notes["llm_scope"] = rec.get("llm_scope")
            notes["api_errors_open"] = rec.get("api_errors_open")
        notes["source"] = source["members"][m]
        if m == best:
            notes["best_member"] = True
        per_member[m] = {"metrics": metrics, "runtime_s": None, "notes": notes}

    per_attribute: dict = {}
    for a in sorted({a for m in ROSTER for a in (members[m]["rec"].get("per_attribute") or {})}):
        entry: dict = {}
        ns = set()
        for m in ROSTER:
            pa = (members[m]["rec"].get("per_attribute") or {}).get(a)
            if pa is None:
                continue
            entry[m] = pa["correct"] / pa["n"] if pa["n"] else 0.0
            ns.add(pa["n"])
        if len(ns) != 1:
            raise SystemExit(f"{task}: the members disagree on the number of test cells of {a}")
        entry["n_cells"] = ns.pop()
        entry["best_member_accuracy"] = max(v for k, v in entry.items() if k in ROSTER)
        per_attribute[a] = entry

    aggregated = {
        "mean_accuracy": mean,
        "max_accuracy": mx,
        "min_accuracy": mn,
        "mean_accuracy_transformations": sum(transf.values()) / 3,
        "max_accuracy_transformations": max(transf.values()),
        "best_member_accuracy": mx,
        "best_member_name": best,
        "n_members": 3,
        "n_cells": n_cells,
        "macro_f1": mean,                  # reader aliases: accuracies, not F1
        "max_f1": mx,
        "min_f1": mn,
        "best_member_f1": mx,
    }
    if tied:
        aggregated["best_member_tied_with"] = ", ".join(tied)
    return {
        "stage": "norm",
        "domain": domain,
        "level": level,
        "scoring_surface": SURFACE,
        "metric": "accuracy",
        "runtime_s": None,
        "roster": list(ROSTER),
        "aggregated": aggregated,
        "per_attribute": per_attribute,
        "per_partition": {},
        "per_member": per_member,
        "metric_aliases": {
            "per_member.<m>.metrics.macro_f1 / f1": "= accuracy",
            "aggregated.macro_f1": "= mean_accuracy",
            "aggregated.max_f1 / best_member_f1": "= max_accuracy",
            "aggregated.min_f1": "= min_accuracy",
            "why": "the generator's readers (build_statistics, analyze_monotonicity / lib.monotonicity, the "
                   "report writers) key the norm stage on macro_f1; on this surface the values are "
                   "cell accuracies on the normalization test set, not F1",
        },
        "source": source,
    }


def add_twins(measured: dict, baseline: dict) -> dict:
    """validate_variant-style <key>_baseline / <key>_delta twins of the numeric values (counts excluded)."""
    out = {}
    for k, v in measured.items():
        out[k] = v
        b = baseline.get(k)
        if k.startswith("n_"):
            continue
        if is_num(v) and is_num(b):
            out[f"{k}_baseline"] = b
            out[f"{k}_delta"] = v - b
    return out


def twin_block(block: dict, base_block: dict) -> dict:
    blk = json.loads(json.dumps(block))
    blk["aggregated"] = add_twins(block["aggregated"], base_block["aggregated"])
    blk["per_attribute"] = {a: add_twins(e, base_block["per_attribute"].get(a, {}))
                            for a, e in block["per_attribute"].items()}
    for m, body in blk["per_member"].items():
        body["metrics"] = add_twins(block["per_member"][m]["metrics"], base_block["per_member"][m]["metrics"])
    return blk


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def same(a: Any, b: Any) -> bool:
    if is_num(a) and is_num(b):
        return abs(float(a) - float(b)) <= TOL
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def walk(built: Any, shipped: Any, path: tuple, rows: list, *, llm_shipped: bool, has_base: bool) -> None:
    if isinstance(built, dict) and isinstance(shipped, dict):
        for k in sorted(set(built) | set(shipped), key=str):
            p = path + (k,)
            if p == ("source",) or (len(p) == 4 and p[0] == "per_member" and p[2:] == ("notes", "source")):
                continue                                    # provenance, not a measured value
            twin = isinstance(k, str) and (k.endswith("_baseline") or k.endswith("_delta"))
            if k not in built:
                rows.append((p, None, shipped[k], "no_base_rerun" if (twin and not has_base) else "only_in_shipped"))
            elif k not in shipped:
                rows.append((p, built[k], None, "only_in_rerun"))
            else:
                walk(built[k], shipped[k], p, rows, llm_shipped=llm_shipped, has_base=has_base)
        return
    status = "equal" if same(built, shipped) else "differs"
    from_llm = (len(path) >= 2 and path[:2] == ("per_member", "llm_only")) or \
               (len(path) >= 3 and path[0] == "per_attribute" and path[2] == "llm_only")
    if llm_shipped and from_llm and status == "equal":
        status = "from_shipped"
    rows.append((path, built, shipped, status))


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return repr(v) if isinstance(v, float) else str(v)


def llm_record(llm_parts: dict[str, dict], domain: str, level: str) -> tuple[dict | None, str | None]:
    for name in (f"{domain}.json", f"{domain}.dry_run.json"):
        part = llm_parts.get(name)
        lv = ((part or {}).get("levels") or {}).get(level) or {}
        if "score" in lv:
            s = lv["score"]
            rec = {"per_category": s["per_category"], "per_attribute": s["per_attribute"],
                   "record_coverage": s["record_coverage"], "n_cells": s["n_cells"], "llm_scope": s["llm_scope"],
                   "api_errors_open": (lv.get("export") or {}).get("api_errors_this_level"),
                   "model_name": (lv.get("llm_params") or {}).get("model_name")}
            return rec, name
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the --out folder of measure.py (and llm_member.py)")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any value differs")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if out == REPO or REPO in out.parents:
        raise SystemExit(f"--out {out} is inside the repository; use the --out folder of measure.py")
    dest = out / "compare"
    (dest / "norm_blocks").mkdir(parents=True, exist_ok=True)

    value_rows: list[list[str]] = []
    cell_rows: list[list[str]] = []
    totals: Counter = Counter()
    differing: list[str] = []
    n_tasks = 0
    for domain in DOMAINS:
        part_path = out / "members" / f"{domain}.json"
        if not part_path.is_file():
            continue
        part = load(part_path)
        llm_dir = out / "llm_member"
        llm_parts = {p.name: load(p) for p in (llm_dir / f"{domain}.json", llm_dir / f"{domain}.dry_run.json")
                     if p.is_file()}
        built: dict[str, dict] = {}
        for level in LEVELS:
            lv = part["levels"].get(level)
            if lv is None:
                continue
            task = f"{domain}_{tier(level)}"
            shipped_file = metrics_path(domain, level)
            shipped = load(shipped_file)["per_stage"]["norm"]
            llm_rec, llm_name = llm_record(llm_parts, domain, level)
            llm_shipped = llm_rec is None
            if llm_shipped:
                llm_rec = llm_from_shipped(shipped)
            recs = {"rule_per_attribute_optimal": lv["members"]["rule_per_attribute_optimal"],
                    "passthrough": lv["members"]["passthrough"], "llm_only": llm_rec}
            source = {"task_key": task, "setting": part.get("setting"),
                      "members": {"rule_per_attribute_optimal": f"{part_path.name} levels.{level}",
                                  "passthrough": f"{part_path.name} levels.{level}",
                                  "llm_only": (f"llm_member/{llm_name} levels.{level}" if llm_name else
                                               f"shipped file {shipped_file.relative_to(REPO)}")}}
            block = build_block(domain, level, lv, recs, source)
            built[level] = block
            has_base = level == "baseline" or "baseline" in built
            if level != "baseline" and "baseline" in built:
                block = twin_block(block, built["baseline"])
            (dest / "norm_blocks" / f"{task}.json").write_text(json.dumps(block, indent=2) + "\n", encoding="utf-8")

            rows: list = []
            walk(block, shipped, (), rows, llm_shipped=llm_shipped, has_base=has_base)
            counts = Counter(r[3] for r in rows)
            totals.update(counts)
            n_tasks += 1
            for p, b, s, st in rows:
                field = ".".join(str(x) for x in p)
                value_rows.append([task, field, fmt(b), fmt(s), st])
                if st in ("differs", "only_in_rerun", "only_in_shipped"):
                    differing.append(f"{task} {field}: re-run {fmt(b)} / shipped {fmt(s)} ({st})")
            for col, key in (("Norm. mean", "mean_accuracy"), ("Norm. ceiling", "max_accuracy")):
                rv, sv = block["aggregated"][key], shipped["aggregated"][key]
                cell_rows.append([task, col, repr(rv), half_up_pct(rv), repr(sv), half_up_pct(sv),
                                  "yes" if half_up_pct(rv) == half_up_pct(sv) else "no",
                                  "re-run" if not llm_shipped else "shipped"])
            print(f"{task:18s} llm_only from {'re-run' if not llm_shipped else 'shipped file':12s} "
                  f"mean {half_up_pct(block['aggregated']['mean_accuracy']):>5s} "
                  f"(shipped {half_up_pct(shipped['aggregated']['mean_accuracy']):>5s})  "
                  f"ceiling {half_up_pct(block['aggregated']['max_accuracy']):>5s} "
                  f"(shipped {half_up_pct(shipped['aggregated']['max_accuracy']):>5s})  "
                  + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if not n_tasks:
        raise SystemExit(f"no measure.py output under {out / 'members'}")
    with open(dest / "norm_values.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "field", "rerun", "shipped", "status"])
        w.writerows(value_rows)
    with open(dest / "table12_norm_cells.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "column", "rerun_value", "rerun_printed", "shipped_value", "shipped_printed",
                    "printed_equal", "llm_only_from"])
        w.writerows(cell_rows)
    print(f"{n_tasks} tasks, {sum(totals.values())} values: " + ", ".join(f"{k} {v}" for k, v in sorted(totals.items())))
    for line in differing[:40]:
        print("  " + line)
    if len(differing) > 40:
        print(f"  ... {len(differing) - 40} more in {dest / 'norm_values.csv'}")
    print(f"wrote {dest / 'norm_values.csv'}, {dest / 'table12_norm_cells.csv'}, {dest / 'norm_blocks'}/")
    return 1 if (args.strict and differing) else 0


if __name__ == "__main__":
    sys.exit(main())
