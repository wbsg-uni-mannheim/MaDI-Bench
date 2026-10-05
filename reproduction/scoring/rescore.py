#!/usr/bin/env python
"""Re-score the stored reference-pipeline outputs into one machine-readable
file, the way ``results/scores_v2.json`` was produced.

Ported from the authors' evaluation code; the scoring
itself is the public ``madi_bench.evaluation`` package of this repository.

Inputs (read only; paths relative to the repository root):

  results/best of breeds/<domain>/<tier>/fused.csv            P2 fused output
  results/best of breeds/<domain>/<tier>/per_stage_summary.csv P2 stored SM / EM test scores
  results/llm pipeline/<domain>/<tier>/fusion/optimization_provided_val/<case>/fused.csv
                                                               P3 fused output, the case named in
                                                               metrics/paper_results.csv, else the case
                                                               P3 selected on its validation set
                                                               (optimization_provided_val/best_case.json),
                                                               else fusion/fused_clean.csv; the "source"
                                                               field names it
  results/llm pipeline/<domain>/<tier>/metrics/paper_results.csv
                                                               P3 stored SM / EM test scores (else
                                                               scoring/scores.json)
  results/llm pipeline/<domain>/baseline/schema_matching/*.csv P3 normalized source tables (base)
  use cases/<domain>/base/output/data_fusion/fused.csv[.gz]    P1 fused output (base tasks)
  use cases/<domain>/base/output/normalization/*.csv           P1 normalized source tables
  use cases/<domain>/<tier>/                                   the tasks (gold)
  (<tier> is "baseline" for the base task under results/)

Optional inputs:

  --p2-norm-root DIR    P2's normalized source tables (stage 2 of the
                        pipeline, replayed with p2_norm_replay.py):
                        DIR/<domain>_<tier>/tables/*.csv and
                        DIR/<domain>_<tier>/stage_2_norm_selection_replay.json
  --record-id-maps DIR  a task tree holding <domain>/<tier>/output/provenance/
                        record_id_map.csv (the stored P2/P3 variant outputs
                        carry the generator's record ids; see
                        record_id_map.py). Default: the task folders.
  --submissions NAME=DIR  further systems (e.g. P4) in the submission layout
                        DIR/<domain>/<tier>/{fused.csv, membership.csv,
                        normalization/*.csv}; repeatable.

Fusion: strict rules on the v2 gold, all-gold accuracy (accuracy on the
graded cells x gold coverage x attribute cell coverage), cell-weighted.
Normalization: the list-aware comparator of ``madi_bench.evaluation``.
SM / EM ("stages"): the stored test scores as the pipelines reported them
(read from the run files; score_submission.py scores a system's
correspondences and schema mapping).

Output: <out>/scores_v2.json and <out>/scores_v2.md; with --compare (default:
results/scores_v2.json) also <out>/compare.json, a value-by-value comparison.

    python -m reproduction.scoring.rescore --out /tmp/rescore [--domains games] [--tiers base,easy]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import REPO_ROOT  # (puts the repository root on sys.path first)

import madi_bench  # noqa: E402
from madi_bench.evaluation import score_fusion, score_normalization  # noqa: E402
from madi_bench.evaluation.fusion_gold import gold_path, parse_id_list  # noqa: E402

from .record_id_map import MAP_RELPATH, load_record_id_map  # noqa: E402
from .tasks import DOMAINS, TIERS, default_tasks_root, get_task  # noqa: E402

PIPELINES = {"p2_best_of_breeds": REPO_ROOT / "results" / "best of breeds",
             "p3_llm_pipeline": REPO_ROOT / "results" / "llm pipeline"}
DEFAULT_REFERENCE = REPO_ROOT / "results" / "scores_v2.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Input roots outside the repository -> the label their paths are recorded under (set in main from the options).
PATH_LABELS: dict[Path, str] = {}


def rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        pass
    for root, label in PATH_LABELS.items():
        try:
            return str(Path(label) / path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def clone_tier(tier: str) -> str:
    """The tier folder name under results/ (the base task is 'baseline')."""
    return "baseline" if tier == "base" else tier


def p3_case_dir(run_dir: Path) -> tuple[Path | None, str]:
    """The fused table of the case the paper used (metrics/paper_results.csv),
    else of the case P3 selected on its validation set
    (fusion/optimization_provided_val/best_case.json), else fusion/fused_clean.csv."""
    cases = run_dir / "fusion" / "optimization_provided_val"
    metrics = run_dir / "metrics" / "paper_results.csv"
    if metrics.is_file():
        with open(metrics, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("category") == "fusion" and r.get("metric") == "accuracy":
                    src = r.get("source_file", "")
                    if "/optimization_provided_val/" in src:
                        case = src.split("/optimization_provided_val/")[1].split("/")[0]
                        candidate = cases / case / "fused.csv"
                        if candidate.is_file():
                            return candidate, case
    best = cases / "best_case.json"
    if best.is_file():
        key = json.loads(best.read_text(encoding="utf-8")).get("best_case_key")
        candidate = cases / f"case_{key}" / "fused.csv"
        if key and candidate.is_file():
            return candidate, f"case_{key}"
    fallback = run_dir / "fusion" / "fused_clean.csv"
    return (fallback, "fused_clean") if fallback.is_file() else (None, "missing")


_PRODUCTS_RAW: dict[str, str] | None = None


def products_raw_to_prefixed(fused: pd.DataFrame, tasks_root: Path) -> pd.DataFrame:
    """P3's products base tables carry raw ids (12198483) with no source
    name; the four base sources' id sets are disjoint, so every raw id maps
    to exactly one products_<N>_ prefix."""
    global _PRODUCTS_RAW
    if _PRODUCTS_RAW is None:
        _PRODUCTS_RAW = {}
        for n in range(1, 5):
            path = get_task("products_base", tasks_root).root / "input/data" / f"dataset_{n}.json"
            for rec in json.loads(path.read_text()):
                _PRODUCTS_RAW[str(rec["id"])] = f"products_{n}_{rec['id']}"
    out = fused.copy()

    def fix(v):
        return _PRODUCTS_RAW.get(str(v), str(v)) if not str(v).startswith("products_") else str(v)

    out["_id"] = out["_id"].map(fix)
    if "_fusion_sources" in out.columns:
        out["_fusion_sources"] = out["_fusion_sources"].map(lambda v: [fix(x) for x in parse_id_list(v)])
    return out


# Record ids of the stored variant outputs: the generator's original ids,
# translated original -> opaque through the task's record-id map before
# scoring (see record_id_map.py); a task without a map is scored unchanged.
FUSED_ID_COLUMNS = ("_id", "id")                 # one record id per cell
FUSED_ID_LIST_COLUMNS = ("_fusion_sources",)     # a list of record ids per cell


def translate_record_ids(frame: pd.DataFrame, to_opaque: dict[str, str],
                         scalar_columns=FUSED_ID_COLUMNS,
                         list_columns=FUSED_ID_LIST_COLUMNS) -> tuple[pd.DataFrame, dict[str, int]]:
    """Every exact record-id cell of ``scalar_columns`` and every id inside a
    list cell of ``list_columns`` mapped through ``to_opaque`` (original ->
    opaque); other cells keep their value and their representation (a
    changed list cell is written back as the Python list literal the stored
    tables use). Returns the frame and the number of ids replaced per column;
    an empty map returns the frame itself."""
    if not to_opaque:
        return frame, {}
    out = frame.copy()
    counts: dict[str, int] = {}
    for col in scalar_columns:
        if col not in out.columns:
            continue
        values = out[col]
        mask = values.notna() & values.astype(str).isin(to_opaque)
        counts[col] = int(mask.sum())
        if counts[col]:
            out[col] = values.astype(object).where(~mask, values[mask].astype(str).map(to_opaque))
    for col in list_columns:
        if col not in out.columns:
            continue
        n = 0

        def fix(value):
            nonlocal n
            ids = parse_id_list(value)
            hits = sum(1 for i in ids if i in to_opaque)
            if not hits:
                return value
            n += hits
            mapped = [to_opaque.get(i, i) for i in ids]
            return repr(mapped) if isinstance(value, str) else mapped

        out[col] = out[col].map(fix)
        counts[col] = n
    return out, counts


# column aliases: the same attribute under the name a pipeline's notebook or
# code used (P1 companies: founders; P3 companies: keypeople_name)
ALIASES = {"companies": (("keypeople_name", "keypeople"), ("founders", "keypeople"))}


def stage_scores(pipeline: str, run_dir: Path) -> tuple[dict, list[Path]]:
    """The stored schema-matching and entity-matching test scores of a run,
    as the paper's per-stage tables used them: P2 from per_stage_summary.csv
    (sm / em_matching test_score), P3 from metrics/paper_results.csv
    (schema_matching f1, entity_matching macro_f1 on the official test), else
    from scoring/scores.json (schema_matching f1, entity_matching per-pair mean F1)."""
    out: dict = {}
    used: list[Path] = []
    if pipeline == "p2_best_of_breeds":
        path = run_dir / "per_stage_summary.csv"
        if path.is_file():
            used.append(path)
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["stage"] == "sm":
                        out["sm_f1"] = float(r["test_score"])
                    elif r["stage"] == "em_matching":
                        out["em_f1"] = float(r["test_score"])
                        out["em_metric"] = r.get("metric_key")
    else:
        path = run_dir / "metrics" / "paper_results.csv"
        if path.is_file():
            used.append(path)
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["category"] == "schema_matching" and r["metric"] == "f1":
                        out["sm_f1"] = float(r["value"])
                    elif r["category"] == "entity_matching" and r["metric"] == "macro_f1":
                        out["em_f1"] = float(r["value"])
                        out["em_metric"] = "macro_f1 (paper_results.csv)"
                    elif r["category"] == "fusion" and r["metric"] == "accuracy":
                        out["fusion_accuracy_as_published"] = float(r["value"])
        elif (run_dir / "scoring" / "scores.json").is_file():
            path = run_dir / "scoring" / "scores.json"
            used.append(path)
            scores = json.loads(path.read_text(encoding="utf-8"))
            sm = (scores.get("schema_matching") or {}).get("f1")
            em = ((scores.get("entity_matching") or {}).get("per_pair_mean") or {}).get("f1")
            fusion = (scores.get("fusion") or {}).get("overall_accuracy_all_gold")
            if sm is not None:
                out["sm_f1"] = float(sm)
            if em is not None:
                out["em_f1"] = float(em)
                out["em_metric"] = "per-pair mean F1 (scoring/scores.json)"
            if fusion is not None:
                out["fusion_accuracy_as_published"] = float(fusion)
    return out, used


def read_fused(path: Path, domain: str | None = None) -> pd.DataFrame:
    fused = pd.read_csv(path, dtype={"_id": str}, low_memory=False)
    for src, dst in ALIASES.get(domain or "", ()):
        if src in fused.columns and dst not in fused.columns:
            fused = fused.rename(columns={src: dst})
    return fused


def fusion_entry(task_dir: Path, fused: pd.DataFrame, membership: pd.DataFrame | None = None) -> dict:
    r = score_fusion(task_dir, fused, membership)
    keep = ("overall_accuracy_all_gold", "overall_accuracy_evaluated", "gold_coverage", "attribute_cell_coverage",
            "n_gold", "n_gold_evaluated", "gold_cells_total", "total_evaluations", "total_correct",
            "n_submitted_aligned", "n_duplicate_anchor_rows", "missing_schema_attributes",
            "per_attribute_accuracy_all_gold", "per_attribute_count", "gold_version", "rules")
    return {k: r[k] for k in keep}


def normalization_entry(task_dir: Path, tables: Path) -> dict:
    r = score_normalization(task_dir, tables)
    return {k: r[k] for k in ("accuracy", "accuracy_transformations", "record_coverage", "rows", "comparator")} | {
        "per_category": {k.split(":", 1)[1]: v for k, v in r["metrics"].items() if k.startswith("category:")},
        "per_attribute": {k.split(":", 1)[1]: v["accuracy"] for k, v in r["metrics"].items() if k.startswith("attribute:")}}


def p1_fused_path(output_dir: Path) -> Path | None:
    for name in ("fused.csv", "fused.csv.gz"):
        path = output_dir / "data_fusion" / name
        if path.is_file():
            return path
    return None


def submission_dir(root: Path, domain: str, tier: str) -> Path | None:
    for name in (tier, clone_tier(tier)):
        path = root / domain / name
        if path.is_dir():
            return path
    return None


def git_commit(path: Path) -> str | None:
    """HEAD of the checkout, or None when there is none or tracked files have uncommitted
    changes (then no commit identifies the code that ran)."""
    try:
        out = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=path,
                             capture_output=True, text=True, timeout=30)
        changes = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"],
                                 cwd=path, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or changes.returncode != 0 or changes.stdout.strip():
        return None
    return out.stdout.strip() or None


# ------------------------------------------------------------------ compare

COMPARE_KEYS = {
    "fusion": ("overall_accuracy_all_gold", "overall_accuracy_evaluated", "gold_coverage",
               "attribute_cell_coverage", "n_gold_evaluated", "gold_cells_total", "total_evaluations",
               "total_correct", "n_submitted_aligned", "n_duplicate_anchor_rows", "missing_schema_attributes",
               "per_attribute_accuracy_all_gold", "per_attribute_count"),
    "normalization": ("accuracy", "accuracy_transformations", "record_coverage", "rows",
                      "per_category", "per_attribute"),
    "stages": ("sm_f1", "em_f1"),
}


def _same(a, b, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    return a == b


def compare(doc: dict, reference: dict, tol: float = 1e-9) -> dict:
    """Value-by-value comparison of a re-score with a reference file (the
    published results/scores_v2.json): every compared value, and the inputs
    whose sha256 differs from the reference's record."""
    rows = []
    ref_results = reference.get("results", {})
    run = set(doc["manifest"].get("pipelines_run", doc["results"]))
    for pipeline in sorted(set(ref_results) - run):
        rows.append({"pipeline": pipeline, "task": None, "block": None, "key": None,
                     "status": "pipeline not run here"})
    for pipeline, tasks in doc["results"].items():
        for task_id, entry in tasks.items():
            ref_entry = ref_results.get(pipeline, {}).get(task_id)
            if ref_entry is None:
                rows.append({"pipeline": pipeline, "task": task_id, "block": None, "key": None,
                             "status": "not in reference"})
                continue
            for block, keys in COMPARE_KEYS.items():
                mine, theirs = entry.get(block), ref_entry.get(block)
                if mine is None and theirs is None:
                    continue
                if mine is None or "error" in (mine or {}):
                    rows.append({"pipeline": pipeline, "task": task_id, "block": block, "key": None,
                                 "status": "missing here", "here": (mine or {}).get("error")})
                    continue
                if theirs is None or "error" in theirs:
                    rows.append({"pipeline": pipeline, "task": task_id, "block": block, "key": None,
                                 "status": "missing in reference"})
                    continue
                for key in keys:
                    a, b = mine.get(key), theirs.get(key)
                    if a is None and b is None:
                        continue
                    rows.append({"pipeline": pipeline, "task": task_id, "block": block, "key": key,
                                 "here": a, "reference": b,
                                 "status": "equal" if _same(a, b, tol) else "different"})
    # reference entries this run did not produce (not requested, or inputs absent)
    for pipeline, tasks in ref_results.items():
        if pipeline not in run:
            continue
        for task_id, ref_entry in tasks.items():
            if task_id in doc["results"].get(pipeline, {}):
                continue
            if task_id.rpartition("_")[0] not in doc["manifest"]["domains"] or \
                    task_id.rpartition("_")[2] not in doc["manifest"]["tiers"]:
                continue
            rows.append({"pipeline": pipeline, "task": task_id, "block": None, "key": None,
                         "status": "only in reference"})
    # inputs: same (pipeline, task, kind, file name) -> same sha256?
    def key(i):
        return (i["pipeline"], i["task"], i["kind"], Path(i["path"]).name.removesuffix(".gz"))

    ref_inputs = {key(i): i for i in reference.get("manifest", {}).get("inputs", [])}
    input_rows = []
    for i in doc["manifest"]["inputs"]:
        r = ref_inputs.get(key(i))
        input_rows.append({"pipeline": i["pipeline"], "task": i["task"], "kind": i["kind"], "path": i["path"],
                           "status": ("not in reference" if r is None else
                                      "identical" if r["sha256"] == i["sha256"] else
                                      "different" if not i["path"].endswith(".gz") else
                                      "compressed copy (sha256 not comparable)")})
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    input_counts: dict[str, int] = {}
    for r in input_rows:
        input_counts[r["status"]] = input_counts.get(r["status"], 0) + 1
    return {"tolerance": tol, "value_counts": counts, "input_counts": input_counts,
            "values": rows, "inputs": input_rows}


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--out", type=Path, required=True,
                    help="output folder (created; must not exist yet)")
    # the published file's task order (papers before products)
    ap.add_argument("--domains", default="companies,games,music,papers,products")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--tasks-root", type=Path, default=None,
                    help="task tree <domain>/<tier>/ (default: $MADI_BENCH_USECASES or 'use cases/')")
    ap.add_argument("--p1-root", type=Path, default=None,
                    help="tree holding P1's outputs as <domain>/base/output/ (default: the task tree)")
    ap.add_argument("--p2-norm-root", type=Path, default=None,
                    help="P2's replayed normalized tables, <domain>_<tier>/tables/ (written by p2_norm_replay.py)")
    ap.add_argument("--record-id-maps", type=Path, default=None,
                    help="tree holding <domain>/<tier>/output/provenance/record_id_map.csv "
                         "(default: the task tree, where the release has none)")
    ap.add_argument("--submissions", action="append", default=[], metavar="NAME=DIR",
                    help="score a further system in the submission layout DIR/<domain>/<tier>/")
    ap.add_argument("--no-p2", action="store_true", help="skip P2 (results/best of breeds)")
    ap.add_argument("--no-p3", action="store_true", help="skip P3 (results/llm pipeline)")
    ap.add_argument("--no-p1", action="store_true", help="skip P1 (the base tasks' output/)")
    ap.add_argument("--compare", type=Path, default=DEFAULT_REFERENCE,
                    help="reference file to compare with (default: results/scores_v2.json)")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args(argv)

    domains = [d for d in args.domains.split(",") if d]
    tiers = [t for t in args.tiers.split(",") if t]
    unknown = [d for d in domains if d not in DOMAINS] + [t for t in tiers if t not in TIERS]
    if unknown:
        ap.error(f"unknown domain/tier: {unknown}")
    tasks_root = args.tasks_root or default_tasks_root()
    p1_root = args.p1_root or tasks_root
    maps_root = args.record_id_maps or tasks_root
    extra: dict[str, Path] = {}
    for item in args.submissions:
        name, sep, root = item.partition("=")
        if not sep or not name or not root:
            ap.error(f"--submissions expects NAME=DIR, got {item!r}")
        extra[name] = Path(root)
    pipelines = {k: v for k, v in PIPELINES.items()
                 if not (k == "p2_best_of_breeds" and args.no_p2) and not (k == "p3_llm_pipeline" and args.no_p3)}
    for root, label in [(args.p2_norm_root, "<p2-norm-root>"), (args.record_id_maps, "<record-id-maps>")] + \
            [(root, f"<{name}>") for name, root in extra.items()]:
        if root is not None:
            PATH_LABELS[Path(root).resolve()] = label

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=False)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    inputs: list[dict] = []
    task_files: dict[str, dict] = {}
    results: dict[str, dict] = {}
    translations: dict[str, dict] = {}
    missing: list[str] = []
    if "p2_best_of_breeds" in pipelines and args.p2_norm_root is None:
        missing.append("p2_best_of_breeds: no --p2-norm-root, P2 normalization not scored "
                       "(create the tables with reproduction/scoring/p2_norm_replay.py)")

    def record_input(pipeline: str, task_id: str, kind: str, path: Path, extra_fields: dict | None = None) -> None:
        inputs.append({"pipeline": pipeline, "task": task_id, "kind": kind, "path": rel(path),
                       "sha256": sha256(path), **(extra_fields or {})})

    for domain in domains:
        for tier in tiers:
            task = get_task(f"{domain}_{tier}", tasks_root)
            task_id = task.task_id
            files = {"fusion_gold_test": gold_path(task.root, "test")}
            norm_set = task.root / "input" / "normalization" / "test.csv"
            if norm_set.is_file():
                files["normalization_test"] = norm_set
            task_files[task_id] = {k: {"path": rel(p), "sha256": sha256(p)} for k, p in files.items() if p.is_file()}
            to_opaque = load_record_id_map(maps_root / domain / tier)      # {} without a map
            if to_opaque:
                translations[task_id] = {"map": str(MAP_RELPATH),
                                         "map_sha256": sha256(maps_root / domain / tier / MAP_RELPATH),
                                         "synthetic_ids": len(to_opaque), "ids_replaced": {}}
            elif tier != "base":
                missing.append(f"{task_id}: no record-id map; stored P2/P3 outputs scored with their stored ids")

            # P2 / P3 stored outputs
            for pipeline, root in pipelines.items():
                run_dir = root / domain / clone_tier(tier)
                entry: dict = {}
                if pipeline == "p2_best_of_breeds":
                    fused_path, case = run_dir / "fused.csv", "fused.csv"
                else:
                    fused_path, case = p3_case_dir(run_dir)
                if fused_path is not None and fused_path.is_file():
                    fused = read_fused(fused_path, domain)
                    if to_opaque:
                        fused, replaced = translate_record_ids(fused, to_opaque)
                        entry["id_translation"] = {"map": str(MAP_RELPATH), "ids_replaced": replaced}
                        translations[task_id]["ids_replaced"][pipeline] = replaced
                    if pipeline == "p3_llm_pipeline" and domain == "products" and tier == "base":
                        fused = products_raw_to_prefixed(fused, tasks_root)
                        entry["id_alias"] = "raw products ids prefixed products_<N>_ by dataset membership"
                    try:
                        entry["fusion"] = fusion_entry(task.root, fused)
                        entry["fusion"]["source"] = case
                    except Exception as exc:  # noqa: BLE001 -- keep the run going, record the failure
                        entry["fusion"] = {"error": f"{type(exc).__name__}: {exc}"}
                    record_input(pipeline, task_id, "fused", fused_path, {"case": case})
                else:
                    entry["fusion"] = {"error": "no fused table"}
                    missing.append(f"{pipeline} {task_id}: no fused table under {rel(run_dir)}")
                if pipeline == "p2_best_of_breeds" and args.p2_norm_root is not None:
                    # stage 2 (norm) replayed: the winner member's output, never applied downstream
                    replay = args.p2_norm_root / f"{domain}_{clone_tier(tier)}"
                    tables = replay / "tables"
                    if tables.is_dir() and norm_set.is_file():
                        try:
                            entry["normalization"] = normalization_entry(task.root, tables)
                            selection = replay / "stage_2_norm_selection_replay.json"
                            if selection.is_file():
                                sel = json.loads(selection.read_text())
                                entry["normalization"]["stage2"] = {
                                    k: sel.get(k) for k in ("winner", "per_member_val", "spread", "vacuous")}
                                record_input(pipeline, task_id, "stage2_selection_replay", selection)
                            entry["normalization"]["source"] = ("stage-2 winner replayed; not applied "
                                                                "downstream in the stored run")
                        except Exception as exc:  # noqa: BLE001
                            entry["normalization"] = {"error": f"{type(exc).__name__}: {exc}"}
                        for t in sorted(tables.glob("*.csv")):
                            record_input(pipeline, task_id, "normalized_table", t)
                    elif norm_set.is_file():
                        missing.append(f"{pipeline} {task_id}: no replayed normalized tables under "
                                       f"{rel(replay)}")
                if pipeline == "p3_llm_pipeline" and tier == "base":
                    tables = run_dir / "schema_matching"
                    if tables.is_dir() and norm_set.is_file():
                        try:
                            entry["normalization"] = normalization_entry(task.root, tables)
                        except Exception as exc:  # noqa: BLE001
                            entry["normalization"] = {"error": f"{type(exc).__name__}: {exc}"}
                        for t in sorted(tables.glob("*.csv")):
                            record_input(pipeline, task_id, "normalized_table", t)
                stages, used = stage_scores(pipeline, run_dir)
                if stages:
                    entry["stages"] = stages
                    for path in used:
                        record_input(pipeline, task_id, "stage_metrics", path)
                results.setdefault(pipeline, {})[task_id] = entry

            # P1: the human-engineered pipeline's outputs (base tasks only)
            if tier == "base" and not args.no_p1:
                output = p1_root / domain / "base" / "output"
                fused_path = p1_fused_path(output)
                if fused_path is not None:
                    fused = read_fused(fused_path, domain)
                    entry = {"fusion": fusion_entry(task.root, fused)}
                    entry["fusion"]["source"] = rel(fused_path)
                    record_input("p1_human", task_id, "fused", fused_path)
                    if (output / "normalization").is_dir() and norm_set.is_file():
                        try:
                            entry["normalization"] = normalization_entry(task.root, output / "normalization")
                        except Exception as exc:  # noqa: BLE001
                            entry["normalization"] = {"error": f"{type(exc).__name__}: {exc}"}
                        for t in sorted((output / "normalization").glob("*.csv")):
                            record_input("p1_human", task_id, "normalized_table", t)
                    results.setdefault("p1_human", {})[task_id] = entry
                else:
                    missing.append(f"p1_human {task_id}: no data_fusion/fused.csv[.gz] under {rel(output)}")

            # further systems in the submission layout
            for name, root in extra.items():
                sub = submission_dir(root, domain, tier)
                if sub is None or not (sub / "fused.csv").is_file():
                    missing.append(f"{name} {task_id}: no fused.csv under {root}/{domain}/{tier}")
                    continue
                fused = pd.read_csv(sub / "fused.csv", dtype={"_id": str}, low_memory=False)
                membership = (pd.read_csv(sub / "membership.csv", dtype=str)
                              if (sub / "membership.csv").is_file() else None)
                entry = {}
                try:
                    entry["fusion"] = fusion_entry(task.root, fused, membership)
                    entry["fusion"]["source"] = rel(sub / "fused.csv")
                except Exception as exc:  # noqa: BLE001
                    entry["fusion"] = {"error": f"{type(exc).__name__}: {exc}"}
                record_input(name, task_id, "fused", sub / "fused.csv")
                if membership is not None:
                    record_input(name, task_id, "membership", sub / "membership.csv")
                if (sub / "normalization").is_dir() and norm_set.is_file():
                    try:
                        entry["normalization"] = normalization_entry(task.root, sub / "normalization")
                    except Exception as exc:  # noqa: BLE001
                        entry["normalization"] = {"error": f"{type(exc).__name__}: {exc}"}
                    for t in sorted((sub / "normalization").glob("*.csv")):
                        record_input(name, task_id, "normalized_table", t)
                results.setdefault(name, {})[task_id] = entry

            shown = list(pipelines) + ([] if args.no_p1 else ["p1_human"]) + list(extra)
            print(f"{task_id:18s} " + "  ".join(
                f"{p}={results[p][task_id]['fusion'].get('overall_accuracy_all_gold', float('nan')):.4f}"
                if task_id in results.get(p, {}) and "overall_accuracy_all_gold" in results[p][task_id]["fusion"]
                else f"{p}=--" for p in shown), flush=True)

    notes = [
        "Fusion: madi_bench.evaluation.score_fusion (strict rules, v2 gold, all-gold accuracy, cell-weighted).",
        "Normalization: madi_bench.evaluation.score_normalization, where a pipeline's normalized tables exist: "
        "P1 and P3 on the base tasks; P2 with --p2-norm-root (the stage-2 tables replayed by "
        "reproduction/scoring/p2_norm_replay.py).",
        "Stages (sm_f1, em_f1): the test scores stored with the runs (P2 per_stage_summary.csv; P3 "
        "metrics/paper_results.csv, else scoring/scores.json).",
        "Record ids: " + (
            f"the stored P2/P3 outputs of {len(translations)} variant task(s) ({', '.join(sorted(translations))}) "
            f"carry the generator's record ids; they were mapped to the task's record ids through the record-id "
            f"maps (<task>/{MAP_RELPATH}; columns {', '.join(FUSED_ID_COLUMNS + FUSED_ID_LIST_COLUMNS)})."
            if translations else "no record-id map found; stored outputs scored with their stored ids."),
        "End-to-end panels: reproduction/scoring/e2e_panels.py.",
    ]
    doc = {
        "manifest": {
            "generated_utc": stamp,
            "script": "reproduction/scoring/rescore.py",
            "repository_commit": git_commit(REPO_ROOT),
            "package_version": madi_bench.__version__,
            "package_path": rel(Path(madi_bench.__file__).parent),
            "fusion_rules_sha256": sha256(Path(madi_bench.__file__).parent / "evaluation/fusion_rules.py"),
            "gold_loader_sha256": sha256(Path(madi_bench.__file__).parent / "evaluation/fusion_gold.py"),
            "fusion_score_sha256": sha256(Path(madi_bench.__file__).parent / "evaluation/fusion_score.py"),
            "tasks_root": rel(tasks_root), "p1_root": rel(p1_root),
            "p2_norm_root": rel(args.p2_norm_root) if args.p2_norm_root else None,
            "record_id_maps_root": rel(maps_root),
            "submissions": {k: rel(v) for k, v in extra.items()},
            "pipelines_run": list(pipelines) + ([] if args.no_p1 else ["p1_human"]) + list(extra),
            "domains": domains, "tiers": tiers,
            "fusion_rules": "strict", "gold": "v2",
            "headline": "overall_accuracy_all_gold = evaluated accuracy x gold coverage x attribute cell coverage",
            "record_id_translation": {"applied": bool(translations), "tasks": translations},
            "task_files": task_files,
            "inputs": inputs, "missing_inputs": missing, "notes": notes,
        },
        "results": results,
    }
    (out_dir / "scores_v2.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    columns = ["p1_human", "p2_best_of_breeds", "p3_llm_pipeline"] + list(extra)
    lines = ["| task | " + " | ".join(f"{c} fusion" for c in columns) + " | "
             + " | ".join(f"{c} normalization (all / transformations)" for c in columns) + " |",
             "|---|" + "---|" * (2 * len(columns))]
    for domain in domains:
        for tier in tiers:
            task_id = f"{domain}_{tier}"
            cells = []
            for p in columns:
                e = results.get(p, {}).get(task_id, {}).get("fusion", {})
                cells.append(f"{e['overall_accuracy_all_gold']:.3f} (cov {e['gold_coverage']:.2f})"
                             if "overall_accuracy_all_gold" in e else "-")
            for p in columns:
                n = results.get(p, {}).get(task_id, {}).get("normalization", {})
                cells.append(f"{n['accuracy']:.3f} / {n['accuracy_transformations']:.3f}" if "accuracy" in n else "-")
            lines.append(f"| {task_id} | " + " | ".join(cells) + " |")
    (out_dir / "scores_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'scores_v2.json'} and scores_v2.md")
    for line in missing:
        print(f"  missing: {line}")

    if not args.no_compare and args.compare is not None and args.compare.is_file():
        reference = json.loads(args.compare.read_text(encoding="utf-8"))
        result = compare(doc, reference)
        result["reference"] = rel(args.compare)
        (out_dir / "compare.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        print(f"\ncompared with {rel(args.compare)}: values {result['value_counts']}, "
              f"inputs {result['input_counts']} -> {out_dir / 'compare.json'}")
        for row in result["values"]:
            if row["status"] not in ("equal",):
                print(f"  {row['status']:20s} {row['pipeline']:22s} {str(row['task'] or ''):18s} "
                      f"{row.get('block') or ''}.{row.get('key') or ''} here={row.get('here')} "
                      f"reference={row.get('reference')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
