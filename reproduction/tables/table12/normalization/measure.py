#!/usr/bin/env python3
"""Table 12, columns Norm.: the two non-LLM members of the generator's normalization committee, scored on
the normalization test set of each task.

It measures the two members' values behind Table 12's Norm. columns (stored in ``per_stage.norm`` of the
committee metrics files, see ``compare.py``). Paths are repository-relative or command-line arguments, and a
run uses one rule-lock setting (the default or ``--locks-from``).

What is measured
----------------
The normalization committee (``usecases_synthetic/lib/committee_norm_c12.py``, members configured in
``usecases_synthetic/config/committees/normalization_committee_<domain>.yaml``) has three members:
``rule_per_attribute_optimal`` (one rule normalizer per attribute), ``llm_only`` (an LLM normalizer,
gpt-5.4-mini; see ``llm_member.py``) and ``passthrough`` (the raw value). This script runs the two members
without an LLM exactly as the committee's normalization stage builds them and scores their output on the
task's ``input/normalization/test.csv`` with ``madi_bench.evaluation.normalization_score`` (read in place).

* Task data: ``usecases_synthetic.lib.variant_loader.load_variant(domain, level)`` (level ``baseline`` is the
  base task), the loader of ``measure_baseline.py`` and ``validate_variant.py``. For Products base this loader
  reads the generator's copy of the task in ``usecases_synthetic/usecases/products/``.
* Committee: ``NormCommitteeRunner(<yaml>, with_llm=False, scoring_surface="schema_constraints")``, as
  ``measure_baseline.py`` builds it (it returns the ``C12NormCommitteeRunner``). ``runner.run(task)`` is
  called unchanged; the member objects it builds are captured.
* Rule selection (``rule_per_attribute_optimal``): made by the committee itself, not here. The committee
  reads a per-domain lock file (the rule chosen per attribute). For an eligible attribute that the lock does
  not name, it tries the candidate rules of the YAML on the task's fusion VALIDATION gold values and saves its
  choice in the lock, which every later level reuses. Eligible attributes: attributes of the gold schema
  mapping that have a constraint in the target schema; all other attributes keep their raw value. Here the
  lock is ``<out>/locks/<domain>.json``, reused from level to level in the order baseline, easy, medium,
  hard, as a ``measure_baseline`` run followed by ``validate_variant`` runs would. It starts as
    - default (the setting of Table 12): a copy of the lock the repository ships
      (``usecases_synthetic/baselines/<domain>/norm_committee_selection.json``, Products only); the other
      four domains have none, so the committee chooses the rules on the base task's validation gold;
    - ``--locks-from DIR``: a copy of ``DIR/<domain>.json`` (or ``DIR/<domain>/norm_committee_selection.json``),
      e.g. ``reproduction/scoring/p2_norm_rule_selection/`` (the rules of the P2 runs).
* Export: every record of every source and every attribute of the gold schema mapping,
  ``member.normalize(raw, attribute, kind, domain)`` (the call the committee scores with), via
  ``_export_member`` of ``reproduction/scoring/p2_norm_replay.py``: one ``<source>.csv`` per source.
* Score: ``score_normalization(task_dir, tables, split="test")``: accuracy over all test cells (the value of
  Table 12), accuracy over the transformation cells (categories knowledge and normalization), record
  coverage and the counts behind them.

Output: ``<out>/members/<domain>.json`` with, per level, the rule selection, each member's scores (overall,
per category, per attribute), their mean and the better member, and the committee's own scores on its
``schema_constraints`` surface for reference (not the Table 12 value). The normalized tables are deleted after
scoring unless ``--keep-tables``. Nothing is written into the repository (checked at the end of the run).
No LLM or API call: ``llm_only`` is not built (``with_llm=False``) and every ``*_API_KEY`` variable is
removed from the environment.

Usage (from the repository root; 1 to 3 minutes per domain for the four levels on 2 CPUs, 8 GB of memory)::

    python reproduction/tables/table12/normalization/measure.py --domain music --out /tmp/norm12
    python reproduction/tables/table12/normalization/compare.py --out /tmp/norm12
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
TASKS = REPO / "use cases"
SHIPPED_LOCKS = REPO / "usecases_synthetic" / "baselines"
COMMITTEE_DIR = REPO / "usecases_synthetic" / "config" / "committees"
DOMAINS = ("companies", "games", "music", "papers", "products")
LEVELS = ("baseline", "easy", "medium", "hard")
SURFACE = "schema_constraints"     # the committee runs' surface (measure_baseline default)
MEMBERS = ("rule_per_attribute_optimal", "passthrough")

logger = logging.getLogger("table12_norm")


# ---------------------------------------------------------------------------
# helpers (also used by llm_member.py)
# ---------------------------------------------------------------------------


def tier(level: str) -> str:
    return "base" if level == "baseline" else level


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path | str) -> str:
    """Repository-relative path when inside the repository."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def outside_repo(path: Path, what: str) -> Path:
    p = path.resolve()
    if p == REPO or REPO in p.parents:
        raise SystemExit(f"{what} {p} is inside the repository; use a directory outside it")
    return p


def parse_levels(text: str) -> list[str]:
    asked = [lv.strip() for lv in text.split(",") if lv.strip()]
    unknown = sorted(set(asked) - set(LEVELS))
    if unknown:
        raise SystemExit(f"unknown level(s) {unknown}; choose from {', '.join(LEVELS)}")
    return [lv for lv in LEVELS if lv in asked]          # always in the order baseline, easy, medium, hard


def setup_repo() -> None:
    """Import the repository's own code and resolve relative paths (committee caches) against its root."""
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def check_modules(*modules: Any) -> None:
    for mod in modules:
        path = Path(mod.__file__).resolve()
        if REPO not in path.parents:
            raise RuntimeError(f"{mod.__name__} imported from {path}, not from {REPO}")


def init_lock(domain: str, lock_path: Path, locks_from: Path | None) -> dict[str, Any]:
    """Start the run's rule lock: a copy of the shipped lock or of --locks-from, else none."""
    from usecases_synthetic.lib.domain_config import _resolve_knob_config_alias

    canonical = _resolve_knob_config_alias(domain) or domain
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        lock_path.unlink()
    if locks_from is None:
        candidates = [SHIPPED_LOCKS / canonical / "norm_committee_selection.json"]
        setting = "shipped lock, else chosen by the committee at the first level (Table 12 setting)"
    else:
        candidates = [locks_from / f"{canonical}.json", locks_from / canonical / "norm_committee_selection.json"]
        setting = f"--locks-from {rel(locks_from)}"
    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        if locks_from is not None:
            raise SystemExit(f"--locks-from: no lock for {canonical} in {locks_from} "
                             f"(expected {canonical}.json or {canonical}/norm_committee_selection.json)")
        return {"setting": setting, "source": None,
                "note": "no lock shipped: the committee chooses the rules at the first level; later levels reuse them"}
    shutil.copyfile(src, lock_path)
    return {"setting": setting, "source": rel(src), "sha256": sha256_file(src),
            "content": json.loads(src.read_text(encoding="utf-8"))}


def redirect_committee(c12: Any, lock_path: Path, oplog_root: Path) -> None:
    """Point the committee's lock file and its LLM operation-log folder away from the repository."""
    c12._selection_cache_path = lambda _domain: lock_path  # noqa: E731

    def _op_log_dir(domain: str, level: str) -> Path:
        p = oplog_root / domain / level
        p.mkdir(parents=True, exist_ok=True)
        return p

    c12._op_log_dir = _op_log_dir


def score_tables(task_dir: Path, table_dir: Path) -> dict[str, Any]:
    from madi_bench.evaluation.normalization_score import score_normalization

    r = score_normalization(task_dir, table_dir, split="test")
    details = r.pop("details")
    m = r["metrics"]
    return {
        "accuracy": r["accuracy"],
        "accuracy_transformations": r["accuracy_transformations"],
        "record_coverage": r["record_coverage"],
        "n_cells": r["rows"],
        "n_not_found": sum(1 for d in details if not d["found"]),
        "n_correct": m["all"]["correct"],
        "n_transformation_cells": (m.get("category:transformations") or {}).get("n", 0),
        "skipped_files": r["skipped_files"],
        "per_category": {k.split(":", 1)[1]: v for k, v in m.items() if k.startswith("category:")},
        "per_attribute": {k.split(":", 1)[1]: v for k, v in m.items() if k.startswith("attribute:")},
    }


def files_modified_since(root: Path, t0: float) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "venv", "pydi-dev", "node_modules")]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                if p.stat().st_mtime >= t0:
                    out.append(str(p.relative_to(root)))
            except OSError:
                continue
    return sorted(out)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    for key in list(os.environ):                      # no API call can happen in this script
        if key.endswith("API_KEY"):
            os.environ.pop(key)
    out = outside_repo(Path(args.out), "--out")
    locks_from = Path(args.locks_from).resolve() if args.locks_from else None
    levels = parse_levels(args.levels)
    domain = args.domain
    setup_repo()
    t_start = time.time()

    import madi_bench
    import usecases_synthetic
    from madi_bench.evaluation.normalization_score import test_rows
    from usecases_synthetic.lib import committee_norm_c12 as c12
    from usecases_synthetic.lib import llm_client
    from usecases_synthetic.lib.committee_norm import NormCommitteeRunner, _build_source_attribute_index
    from usecases_synthetic.lib.committee_paths import resolve_committee_path
    from usecases_synthetic.lib.protection import kind_map_for_domain
    from usecases_synthetic.lib.variant_loader import load_variant
    from reproduction.scoring import p2_norm_replay

    check_modules(madi_bench, usecases_synthetic, p2_norm_replay)

    def _no_llm(*a: Any, **k: Any) -> Any:
        raise RuntimeError("LLM client construction is disabled in this measurement")

    llm_client.build_chat_openai = _no_llm

    lock_path = out / "locks" / f"{domain}.json"
    lock_init = init_lock(domain, lock_path, locks_from)
    if lock_init["source"] is None and levels[0] != "baseline":
        raise SystemExit("without a lock the committee chooses the rules at the first level it runs, and the "
                         "variants reuse the rules chosen on the base task: start --levels with baseline "
                         "(or pass --locks-from)")
    redirect_committee(c12, lock_path, out / "oplog")

    # capture the member objects the committee builds
    captured: dict[str, Any] = {}
    _Pass, _Comp = c12._PassthroughNormalizer, c12._CompositeRuleNormalizer

    class _CapPass(_Pass):
        def __init__(self) -> None:
            captured["passthrough"] = self

    class _CapComp(_Comp):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            captured["rule_per_attribute_optimal"] = self

    c12._PassthroughNormalizer = _CapPass
    c12._CompositeRuleNormalizer = _CapComp

    norm_yaml = resolve_committee_path("normalization_committee", domain, committee_dir=COMMITTEE_DIR)
    part: dict[str, Any] = {
        "domain": domain, "setting": lock_init["setting"], "lock_init": lock_init,
        "norm_yaml": rel(norm_yaml), "norm_yaml_sha256": sha256_file(norm_yaml),
        "scoring_surface_of_the_committee_run": SURFACE,
        "code": {"usecases_synthetic": rel(usecases_synthetic.__file__), "madi_bench": rel(madi_bench.__file__),
                 "export": rel(p2_norm_replay.__file__), "python": sys.version.split()[0]},
        "levels": {},
    }
    part_path = out / "members" / f"{domain}.json"
    part_path.parent.mkdir(parents=True, exist_ok=True)

    for level in levels:
        t_level = time.time()
        task_dir = TASKS / domain / tier(level)
        bundle = load_variant(domain, level=level)
        attr_index = _build_source_attribute_index(bundle.sm_mapping, bundle.knob_08_renames)
        kind_map = kind_map_for_domain(bundle.domain)
        test = test_rows(task_dir, "test")
        test_attrs = Counter(r["attribute"] for r in test)
        lv: dict[str, Any] = {
            "task": f"{domain}_{tier(level)}",
            "task_dir": rel(task_dir),
            "test_csv_sha256": sha256_file(task_dir / "input" / "normalization" / "test.csv"),
            "sources": {s: int(len(df)) for s, df in bundle.sources.items()},
            "variant_root": rel(bundle.variant_root),
            "test_rows_per_attribute": dict(test_attrs),
        }
        try:
            val_t, test_t = c12._load_eval_targets_for_bundle(bundle.domain, bundle)
            lv["committee_entity_sets"] = {"validation": len(val_t), "test": len(test_t)}
        except Exception as exc:  # noqa: BLE001 - the runner tolerates this on the schema surface
            lv["committee_entity_sets"] = {"error": f"{type(exc).__name__}: {exc}"}

        before = json.loads(lock_path.read_text()) if lock_path.exists() else None
        captured.clear()
        runner = NormCommitteeRunner(norm_yaml, with_llm=False, scoring_surface=SURFACE)
        result = runner.run(bundle)
        after = json.loads(lock_path.read_text()) if lock_path.exists() else None
        rule_notes = result.per_member["rule_per_attribute_optimal"].notes or {}
        selection_map = dict(rule_notes.get("selection_map", {}))
        eligible = sorted(selection_map)
        lv.update({
            "roster": list(runner.roster_names),
            "eligible_attributes": eligible,
            "selection_map": selection_map,
            "lock_before": before, "lock_written_this_level": before != after,
            "rule_scope_of_test_attributes": {
                a: (selection_map[a] if a in selection_map else "not eligible -> passthrough")
                for a in sorted(test_attrs)},
            "committee_own_surface": {
                n: {k: m.metrics.get(k) for k in ("macro_f1", "f1_val", "scoring_surface")}
                for n, m in result.per_member.items()},
            "members": {},
        })
        tables_root = out / "tables" / f"{domain}_{level}"
        for member in MEMBERS:
            table_dir = tables_root / member
            info = p2_norm_replay._export_member(
                member=captured[member], member_name=member, bundle=bundle, attr_index=attr_index,
                kind_map=kind_map, eligible=set(eligible), selection_map=selection_map, table_dir=table_dir)
            sc = score_tables(task_dir, table_dir)
            sc["export_multi_column_conflicts"] = sum(
                a["multi_column_conflicts"] for s in info["sources"].values() for a in s["attributes"].values())
            lv["members"][member] = sc
        accs = [lv["members"][m]["accuracy"] for m in MEMBERS]
        tras = [lv["members"][m]["accuracy_transformations"] or 0.0 for m in MEMBERS]
        lv["mean_of_these_two_members"] = {"accuracy": statistics.mean(accs),
                                           "accuracy_transformations": statistics.mean(tras)}
        best = max(MEMBERS, key=lambda m: lv["members"][m]["accuracy"])
        lv["better_of_these_two_members"] = {"member": best, "accuracy": lv["members"][best]["accuracy"],
                                             "accuracy_transformations":
                                                 lv["members"][best]["accuracy_transformations"]}
        if not args.keep_tables and tables_root.exists():
            shutil.rmtree(tables_root)
        lv["runtime_s"] = round(time.time() - t_level, 1)
        part["levels"][level] = lv
        part_path.write_text(json.dumps(part, indent=1, default=str) + "\n", encoding="utf-8")
        r, p = lv["members"]["rule_per_attribute_optimal"], lv["members"]["passthrough"]
        logger.info("%s/%s rule=%.4f/%.4f pass=%.4f/%.4f coverage=%s/%s n=%d (%.0f s)", domain, level,
                    r["accuracy"], r["accuracy_transformations"] or 0, p["accuracy"],
                    p["accuracy_transformations"] or 0, r["record_coverage"], p["record_coverage"],
                    r["n_cells"], lv["runtime_s"])

    part["final_lock"] = json.loads(lock_path.read_text()) if lock_path.exists() else None
    part["repository_files_modified_during_run"] = files_modified_since(REPO, t_start)
    part["runtime_total_s"] = round(time.time() - t_start, 1)
    part["created_utc"] = datetime.now(timezone.utc).isoformat()
    part_path.write_text(json.dumps(part, indent=1, default=str) + "\n", encoding="utf-8")
    if part["repository_files_modified_during_run"]:
        logger.warning("files modified in the repository during the run: %s",
                       part["repository_files_modified_during_run"])
    print(f"wrote {part_path}")
    print(f"{'task':18s} {'n':>4s} {'rule':>7s} {'pass':>7s}   (accuracy on the normalization test set)")
    for lv in part["levels"].values():
        print(f"{lv['task']:18s} {lv['members']['passthrough']['n_cells']:4d} "
              f"{lv['members']['rule_per_attribute_optimal']['accuracy']:7.4f} "
              f"{lv['members']['passthrough']['accuracy']:7.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--levels", default=",".join(LEVELS),
                    help="comma-separated subset of baseline,easy,medium,hard (always run in this order)")
    ap.add_argument("--out", required=True, help="output folder outside the repository")
    ap.add_argument("--locks-from", default=None,
                    help="folder with a rule lock per domain (<domain>.json) to start from instead of the "
                         "shipped lock, e.g. reproduction/scoring/p2_norm_rule_selection")
    ap.add_argument("--keep-tables", action="store_true", help="keep the normalized tables under <out>/tables/")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="[%(asctime)s %(levelname)s] %(name)s - %(message)s")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
