#!/usr/bin/env python
"""Table 12, columns Fus. (committee mean and ceiling): replay the generator's fusion committee and score
every member's fused output with the benchmark's fusion scorer.

Two steps:

  replay  Runs the fusion stage of the committee exactly as usecases_synthetic/scripts/measure_baseline.py
          (base) and validate_variant.py (variants) run it: load the task with
          usecases_synthetic.lib.variant_loader.load_variant, build the correspondences of the gold-derived
          "perfect clusters" (fusion_perfect_clusters), and run FusionCommitteeRunner with
          fused_dump_dir, which writes each member's fused table to <out>/dumps/<task>/<member>.csv.gz.
          The stored validation selection (usecases_synthetic/baselines/<domain>/fusion_committee_selection.json)
          is read and never rewritten. The llm_only member makes no API call: its LLM client is replaced by
          a stub, so every LLM decision that is not in the local LLM cache (usecases_synthetic/cache/) falls
          back to voting. llm_only is not part of Table 12.
          The lenient per-member metrics of the replay are compared with the stored committee metrics
          (usecases_synthetic/baselines/<domain>/baseline_metrics.json, validation/<domain>/<tier>/metrics.json);
          equal values mean the dumps are the outputs the committee scored. Writes <out>/replay/<task>.json.

  score   Reads the dumps and scores each member with madi_bench.evaluation.score_fusion (strict per-domain
          rules, v2 test gold, accuracy over all gold cells, cell-weighted). Table 12 prints the mean and the
          maximum ("ceiling") over the eight members other than llm_only, in percent with one decimal.
          Writes <out>/fusion_committee_members.csv and <out>/fusion_committee_summary.csv (the same columns
          as results/paper_tables/table12/).

Run from the repository root (papers needs about 24 GB of memory):

    python reproduction/tables/table12/committee_fusion_strict.py replay --domain products --out /tmp/t12
    python reproduction/tables/table12/committee_fusion_strict.py score --tasks products_base --out /tmp/t12

Nothing is written outside --out. No API key is used (every *_API_KEY variable is removed at start).
"""
from __future__ import annotations

import os
import sys

for _k in list(os.environ):
    if _k.endswith("API_KEY"):
        os.environ.pop(_k)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SYN = REPO / "usecases_synthetic"
COMMITTEE_DIR = SYN / "config" / "committees"
TASKS = REPO / "use cases"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOMAINS = ("companies", "games", "music", "papers", "products")
LEVELS = {"base": "baseline", "easy": "easy", "medium": "medium", "hard": "hard"}
LLM_MEMBER = "llm_only"
# Column names that some outputs use for the Companies target attribute keypeople; renamed before scoring,
# as the scoring of the stored pipeline outputs does (reproduction/scoring/rescore.py).
ALIASES = {"companies": (("keypeople_name", "keypeople"), ("founders", "keypeople"))}
KEEP = ("overall_accuracy_all_gold", "overall_accuracy_evaluated", "gold_coverage", "attribute_cell_coverage",
        "n_gold", "n_gold_evaluated", "gold_cells_total", "total_evaluations", "total_correct",
        "missing_schema_attributes", "gold_version", "rules")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- replay

def stored_metrics_path(domain: str, tier: str) -> Path:
    if tier == "base":
        return SYN / "baselines" / domain / "baseline_metrics.json"
    return SYN / "validation" / domain / tier / "metrics.json"


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def compare_metrics(replay: dict, stored: dict) -> dict:
    """Every numeric metric of the replay against the stored value of the same member."""
    diffs, compared = {}, 0
    for k, v in replay.items():
        a, b = _num(v), _num(stored.get(k))
        if a is None or b is None:
            continue
        compared += 1
        if math.isnan(a) and math.isnan(b):
            continue
        if math.isnan(a) or math.isnan(b) or abs(a - b) > 1e-12:
            diffs[k] = {"replay": a, "stored": b}
    return {"keys_compared": compared, "differing": diffs}


def replay(domain: str, tiers: list[str], out: Path) -> int:
    os.chdir(REPO)  # the committee resolves its caches relative to the repository root
    from usecases_synthetic.lib import committee_fusion as cf
    from usecases_synthetic.lib import committee_fusion_c12 as c12
    from usecases_synthetic.lib.committee_fusion import FusionCommitteeRunner
    from usecases_synthetic.lib.committee_paths import resolve_committee_path
    from usecases_synthetic.lib.fusion_perfect_clusters import build_perfect_clusters_correspondences
    from usecases_synthetic.lib.variant_loader import load_variant

    for k in list(os.environ):          # again, after the imports (a module could have loaded a .env file)
        if k.endswith("API_KEY"):
            os.environ.pop(k)

    stub_calls: Counter = Counter()

    def stub_llm_builder(model: str = "gpt-5.4-mini", temperature: float = 0.0, max_tokens: int = 2048):
        def _call(system_prompt: str, user_prompt: str, model_id: str) -> str:
            stub_calls[model_id] += 1
            raise RuntimeError("replay: no API call; llm_judge falls back to voting for this cell")
        return _call

    def refuse_selection_write(dom, cache):  # noqa: ANN001
        raise RuntimeError(f"replay must not rewrite the stored validation selection of {dom!r}")

    def oplog_dir(dom: str, level: str) -> Path:
        p = out / "oplog" / dom / level
        p.mkdir(parents=True, exist_ok=True)
        return p

    cf._build_openai_llm_callable = stub_llm_builder
    c12._save_selection_cache = refuse_selection_write
    c12._op_log_dir = oplog_dir

    bad = 0
    for tier in tiers:
        level = LEVELS[tier]
        task = f"{domain}_{tier}"
        dump_dir = out / "dumps" / task
        if dump_dir.exists():
            shutil.rmtree(dump_dir)
        stub_calls.clear()
        selection = SYN / "baselines" / domain / "fusion_committee_selection.json"
        before = sha256(selection)
        stored = json.loads(stored_metrics_path(domain, tier).read_text(encoding="utf-8"))
        stored = stored["per_stage"]["fusion"]["per_member"]
        t0 = time.monotonic()
        runner = FusionCommitteeRunner(resolve_committee_path("fusion_committee", domain, committee_dir=COMMITTEE_DIR))
        bundle = load_variant(domain, level=level)
        corr = build_perfect_clusters_correspondences(domain, bundle)
        result = runner.run(bundle, correspondences=corr, fused_dump_dir=dump_dir)
        members = {}
        for name, m in result.per_member.items():
            dump = Path(m.notes["fused_dump"]) if "fused_dump" in m.notes else None
            members[name] = {
                "dump": str(dump.relative_to(out)) if dump else None,
                "dump_sha256": sha256(dump) if dump else None,
                "runtime_s": round(m.runtime_s, 2),
                "lenient_overall_accuracy": m.metrics.get("overall_accuracy"),
                "vs_stored": compare_metrics(m.metrics, stored.get(name, {}).get("metrics", {})),
            }
        equal = {n: not v["vs_stored"]["differing"] for n, v in members.items() if n != LLM_MEMBER}
        rec = {"task": task, "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
               "roster": result.roster, "correspondences": int(len(corr)),
               "llm_stub_calls": int(sum(stub_calls.values())),
               "selection_cache_unchanged": sha256(selection) == before,
               "non_llm_members_equal_stored": equal, "wall_s": round(time.monotonic() - t0, 1),
               "members": members}
        (out / "replay").mkdir(parents=True, exist_ok=True)
        (out / "replay" / f"{task}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        ok = all(equal.values()) and rec["selection_cache_unchanged"]
        bad += not ok
        print(f"{task}: members={len(members)} non-LLM members equal to the stored metrics: {all(equal.values())} "
              f"llm stub calls={rec['llm_stub_calls']} wall={rec['wall_s']}s", flush=True)
    return 1 if bad else 0


# ----------------------------------------------------------------------------- score

def read_fused(path: Path, domain: str):
    import pandas as pd

    fused = pd.read_csv(path, dtype={"_id": str}, low_memory=False)
    for src, dst in ALIASES.get(domain, ()):
        if src in fused.columns and dst not in fused.columns:
            fused = fused.rename(columns={src: dst})
    return fused


def score(tasks: list[str], out: Path, dumps: Path) -> int:
    from madi_bench.evaluation import score_fusion

    rows, summary = [], []
    for task in tasks:
        domain, tier = task.split("_")
        task_dir = TASKS / domain / tier
        files = sorted((dumps / task).glob("*.csv.gz"))
        if not files:
            raise FileNotFoundError(f"no member dumps under {dumps / task}; run the replay step first")
        acc = {}
        for p in files:
            name = p.name[: -len(".csv.gz")]
            fused = read_fused(p, domain)
            r = score_fusion(task_dir, fused)
            acc[name] = r["overall_accuracy_all_gold"]
            rows.append([domain, tier, name, int(name != LLM_MEMBER), r["overall_accuracy_all_gold"],
                         r["gold_coverage"], r["overall_accuracy_evaluated"], r["attribute_cell_coverage"],
                         r["total_correct"], r["gold_cells_total"], int(len(fused))])
        eight = {k: v for k, v in acc.items() if k != LLM_MEMBER}
        best = max(eight, key=lambda k: eight[k])
        summary.append([domain, tier, sum(eight.values()) / len(eight), eight[best], best,
                        sum(acc.values()) / len(acc), max(acc.values())])
        print(f"{task}: {len(eight)} members without {LLM_MEMBER}: mean {100 * summary[-1][2]:.1f}, "
              f"ceiling {100 * eight[best]:.1f} ({best})", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fusion_committee_members.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["domain", "tier", "member", "in_table12", "accuracy_all_gold", "gold_coverage",
                    "accuracy_evaluated", "attribute_cell_coverage", "correct_cells", "gold_cells", "fused_rows"])
        w.writerows(rows)
    with (out / "fusion_committee_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["domain", "tier", "mean_8_members", "ceiling_8_members", "best_member",
                    "mean_9_members_incl_llm", "ceiling_9_members_incl_llm"])
        w.writerows(summary)
    print("wrote", out / "fusion_committee_members.csv", "and", out / "fusion_committee_summary.csv")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="write every member's fused output for one domain")
    r.add_argument("--domain", required=True, choices=DOMAINS)
    r.add_argument("--tiers", default="base,easy,medium,hard")
    r.add_argument("--out", required=True, type=Path, help="output folder (dumps/, replay/, oplog/)")
    s = sub.add_parser("score", help="score the member dumps with the benchmark's fusion scorer")
    s.add_argument("--tasks", required=True, help="comma-separated, e.g. products_base,products_easy")
    s.add_argument("--out", required=True, type=Path, help="output folder of the replay step")
    s.add_argument("--dumps", type=Path, default=None, help="dump folder (default: <out>/dumps)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    out = a.out.resolve()
    if a.cmd == "replay":
        out.mkdir(parents=True, exist_ok=True)
        return replay(a.domain, [t for t in a.tiers.split(",") if t], out)
    return score([t for t in a.tasks.split(",") if t], out, (a.dumps or out / "dumps").resolve())


if __name__ == "__main__":
    raise SystemExit(main())
