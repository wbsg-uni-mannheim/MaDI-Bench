#!/usr/bin/env python
"""End-to-end metric panels of a base task (paper Table 8): reference-free,
silver-reference and ground-truth rows, plus the ground-truth fusion accuracy.

For every system:

* reference-free + ground-truth rows: ``score_e2e.score_e2e`` (PyDI
  ``compute_e2e_panel`` against the task's fusion test gold);
* ground-truth fusion accuracy: ``madi_bench.evaluation.score_fusion``
  (all-gold accuracy, strict rules, v2 gold), the value the paper prints in
  the ground-truth block;
* silver-reference rows: ``score_e2e.score_e2e_silver`` against P1's fused
  output (``use cases/<domain>/base/output/data_fusion/fused.csv``, or its
  compressed copy ``fused.csv.gz``), with the exact sparse cluster alignment.

Systems:

* P2 (default): ``results/best of breeds/<domain>/baseline/`` (fused.csv,
  correspondences.csv; membership built from ``_fusion_sources``);
* ``--submission NAME=DIR`` (repeatable; e.g. P4): a folder with fused.csv,
  membership.csv and correspondences.csv (the submission layout). As in the
  paper's measurement, the submitted membership is used for the ground-truth
  panel and the membership with source names repaired for the silver panel.

Output: ``<out>/<domain>.json`` (metrics only). ``--panel-dir`` additionally
writes PyDI's full panel folders; they contain per-cluster tables derived
from the gold (``*_gold.csv``) and must not be committed.

CPU only, no LLM / API calls. Memory: 8 GB for games, companies, music and
products; papers needs about 24 GB.

    python -m reproduction.scoring.e2e_panels --domain games --out /tmp/e2e
    python -m reproduction.scoring.e2e_panels --domain games --out /tmp/e2e --no-p2 \\
        --submission P4=/path/to/p4/games_base/submission
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd  # noqa: E402

from . import REPO_ROOT  # noqa: E402  (puts the repository root on sys.path first)
from .tasks import DOMAINS, TIERS, default_tasks_root, get_task  # noqa: E402

P2_ROOT = REPO_ROOT / "results" / "best of breeds"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _clean(x):
    """float('nan') -> None so json dumps; numpy scalars -> python."""
    try:
        import numpy as np
        if isinstance(x, np.generic):
            x = x.item()
    except ImportError:
        pass
    if isinstance(x, float) and x != x:
        return None
    return x


def default_silver(task) -> Path | None:
    for name in ("fused.csv", "fused.csv.gz"):
        path = task.root / "output" / "data_fusion" / name
        if path.is_file():
            return path
    return None


def compare_measurements(mine: dict, reference: dict, tol: float) -> dict:
    """Numeric comparison of the metric blocks two measurement files share."""
    rows = []
    for block, values in mine.items():
        if not isinstance(values, dict) or block not in reference or not isinstance(reference[block], dict):
            continue
        if not (block.endswith(("_rf", "_gt", "_silver", "_fusion_public"))):
            continue
        for key, a in values.items():
            b = reference[block].get(key)
            if not isinstance(a, (int, float)) and not isinstance(b, (int, float)):
                if a == b or a is None and b is None:
                    continue
            same = (a is None and b is None) or (
                isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)) or a == b
            rows.append({"block": block, "key": key, "here": a, "reference": b,
                         "status": "equal" if same else "different"})
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"tolerance": tol, "counts": counts, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--tier", default="base", choices=TIERS,
                    help="task tier (default base; the paper's table covers the base tasks)")
    ap.add_argument("--out", type=Path, required=True, help="output folder (created if needed)")
    ap.add_argument("--tasks-root", type=Path, default=None,
                    help="task tree <domain>/<tier>/ (default: $MADI_BENCH_USECASES or 'use cases/')")
    ap.add_argument("--no-p2", action="store_true", help="skip P2 (results/best of breeds)")
    ap.add_argument("--submission", action="append", default=[], metavar="NAME=DIR",
                    help="a further system: DIR with fused.csv, membership.csv, correspondences.csv")
    ap.add_argument("--silver", type=Path, default=None,
                    help="silver reference fused table (default: P1's output in the base task folder)")
    ap.add_argument("--no-silver", action="store_true", help="skip the silver-reference rows")
    ap.add_argument("--alignment", choices=("sparse", "original"), default="sparse",
                    help="cluster alignment for the silver rows (default: the exact sparse variant)")
    ap.add_argument("--panel-dir", type=Path, default=None,
                    help="also write PyDI's panel folders here (gold-derived files; do not commit)")
    ap.add_argument("--compare", type=Path, action="append", default=[],
                    help="a previous measurement file to compare with (same block names, at the top "
                         "or per domain under 'tasks'); repeatable")
    ap.add_argument("--tolerance", type=float, default=1e-9)
    args = ap.parse_args(argv)

    logging.disable(logging.CRITICAL)
    from madi_bench.evaluation import score_fusion as public_score_fusion

    from .score_e2e import (
        install_sparse_alignment,
        load_p1_silver,
        load_panel_sources,
        read_pipeline_fused,
        repaired_membership,
        score_e2e,
        score_e2e_silver,
        source_lookup,
    )
    from .score_em import load_correspondences

    t0 = time.time()
    domain = args.domain
    task = get_task(f"{domain}_{args.tier}", args.tasks_root or default_tasks_root())
    out: dict = {"domain": domain, "task": task.task_id, "task_root": rel(task.root), "inputs": {},
                 "errors": {}, "log": [], "script": "reproduction/scoring/e2e_panels.py"}

    def rec(path: Path) -> Path:
        out["inputs"][rel(path)] = sha256(path)
        return path

    def log(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s] {domain} {msg}"
        out["log"].append(line)
        print(line, flush=True)

    sources = load_panel_sources(task)
    lookup = source_lookup(task, sources)
    rec(task.root / "input" / "schemamatching" / "target_schema.json")
    for p in sorted((task.root / "input" / "fusion").glob("*")):
        if p.is_file():
            rec(p)
    marker = task.root / "input" / "fusion" / "GOLD_VERSION"
    out["gold_version_marker"] = marker.read_text().strip() if marker.is_file() else None

    systems: dict[str, dict] = {}
    if not args.no_p2:
        p2_dir = P2_ROOT / domain / ("baseline" if args.tier == "base" else args.tier)
        fused_path = rec(p2_dir / "fused.csv")
        fused = read_pipeline_fused(fused_path)
        from . import aliases
        mem, mem_info = repaired_membership(task, aliases.normalize_fused_sources(task, fused), lookup)
        systems["P2"] = {"fused": fused, "fused_path": fused_path,
                         "corr": load_correspondences(rec(p2_dir / "correspondences.csv")),
                         "membership_gt": mem, "membership_silver": mem, "membership_info": mem_info,
                         # the paper scored P2's fusion accuracy on the stored table as read by rescore.py
                         "fusion_frame": pd.read_csv(fused_path, dtype={"_id": str}, low_memory=False),
                         "fusion_membership": None}
    for item in args.submission:
        name, sep, folder = item.partition("=")
        if not sep or not name or not folder:
            ap.error(f"--submission expects NAME=DIR, got {item!r}")
        folder = Path(folder)
        fused = pd.read_csv(rec(folder / "fused.csv"), dtype={"_id": str})
        mem_path = folder / "membership.csv"
        mem_raw = pd.read_csv(rec(mem_path), dtype=str) if mem_path.is_file() else None
        corr_path = folder / "correspondences.csv"
        corr = (load_correspondences(rec(corr_path)) if corr_path.is_file()
                else pd.DataFrame(columns=["id1", "id2", "score"]))
        mem, mem_info = repaired_membership(task, fused, lookup, supplied=mem_raw)
        systems[name] = {"fused": fused, "fused_path": folder / "fused.csv", "corr": corr,
                         "membership_gt": mem_raw, "membership_silver": mem, "membership_info": mem_info,
                         "fusion_frame": fused, "fusion_membership": mem_raw}
    for name, s in systems.items():
        out[f"{name}_membership"] = s["membership_info"]
        out[f"{name}_fused_path"] = rel(s["fused_path"])
    log("inputs loaded")

    # reference-free + ground-truth rows, ground-truth fusion accuracy
    for name, s in systems.items():
        try:
            res = score_e2e(task, s["fused"], s["corr"], membership=s["membership_gt"],
                            out_dir=(args.panel_dir / domain / f"{name}_rf_gt") if args.panel_dir else None)
            out[f"{name}_rf"] = {k: _clean(v) for k, v in res["reference_free"].items()}
            out[f"{name}_gt"] = {k: _clean(v) for k, v in res["ground_truth"].items()}
            out[f"{name}_rf_gt_warnings"] = res.get("warnings")
            log(f"{name} RF/GT done")
        except Exception as exc:  # noqa: BLE001
            out["errors"][f"{name}_rf_gt"] = traceback.format_exc()
            log(f"{name} RF/GT ERROR {exc}")
        try:
            pub = public_score_fusion(task.root, s["fusion_frame"], membership=s["fusion_membership"])
            out[f"{name}_fusion_public"] = {k: _clean(pub.get(k)) for k in (
                "overall_accuracy_all_gold", "gold_coverage", "total_correct", "gold_cells_total", "rules",
                "gold_version")}
            log(f"{name} fusion done {out[f'{name}_fusion_public']['overall_accuracy_all_gold']}")
        except Exception as exc:  # noqa: BLE001
            out["errors"][f"{name}_fusion"] = traceback.format_exc()
            log(f"{name} fusion ERROR {exc}")

    # silver-reference rows
    if not args.no_silver:
        silver_path = args.silver or default_silver(get_task(f"{domain}_base", args.tasks_root or default_tasks_root()))
        use_alignment, align_info = install_sparse_alignment()
        out["alignment"] = {"used": args.alignment, **align_info}
        silver = None
        if silver_path is None:
            out["errors"]["silver"] = "no silver reference: P1's data_fusion/fused.csv[.gz] not found"
        else:
            try:
                silver, info = load_p1_silver(task, rec(silver_path), lookup)
                out["silver_reference"] = {"path": rel(silver_path), **info}
                log("silver reference built")
            except Exception:  # noqa: BLE001
                out["errors"]["silver_reference"] = traceback.format_exc()
        if silver is not None:
            use_alignment(args.alignment)
            for name, s in systems.items():
                try:
                    m, w = score_e2e_silver(
                        task, s["fused"], s["membership_silver"], silver, sources=sources,
                        out_dir=(args.panel_dir / domain / f"{name}_silver_{args.alignment}")
                        if args.panel_dir else None,
                        silver_label=f"P1 fused output ({rel(silver_path)})")
                    out[f"{name}_silver"] = m
                    out[f"{name}_silver_warnings"] = w
                    log(f"{name} silver done")
                except Exception as exc:  # noqa: BLE001
                    out["errors"][f"{name}_silver"] = traceback.format_exc()
                    log(f"{name} silver ERROR {exc}")
            use_alignment("original")

    out["inputs_changed_during_run"] = [p for p, h in out["inputs"].items()
                                        if sha256(REPO_ROOT / p if not Path(p).is_absolute() else Path(p)) != h]
    out["seconds"] = round(time.time() - t0, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{task.task_id if args.tier != 'base' else domain}.json"
    target.write_text(json.dumps(out, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    log(f"wrote {target} errors={list(out['errors'])}")
    for i, ref_path in enumerate(args.compare, start=1):
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
        if isinstance(reference.get("tasks"), dict):
            # a file with one block per base task under "tasks" (the paper's
            # measurement file, results/paper_tables/table8/e2e_measurements_*.json)
            reference = reference["tasks"].get(domain) or {}
        result = compare_measurements(out, reference, args.tolerance)
        result["reference"] = str(ref_path)
        cmp_path = args.out / f"{target.stem}.compare{i}.json"
        cmp_path.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
        print(f"compared with {ref_path}: {result['counts']} -> {cmp_path}")
        for row in result["rows"]:
            if row["status"] != "equal":
                print(f"  different {row['block']}.{row['key']}: here={row['here']} reference={row['reference']}")
    return 1 if out["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
