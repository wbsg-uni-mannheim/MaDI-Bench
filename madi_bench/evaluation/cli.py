"""Command-line entry points ``madi-score-fusion`` and
``madi-score-normalization``: the full result as JSON on stdout.

The task is named either by ``--domain`` (with ``--tier``, default ``base``)
under ``--tasks-root`` (alias ``--gold-root``; default: ``$MADI_BENCH_USECASES``
or the ``use cases/`` folder of the repository), or directly by ``--task``, the
task folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .fusion_gold import ANCHOR_PRIORITY
from .fusion_score import SubmissionError, load_table, score_fusion
from .normalization_score import score_normalization

TIERS = ("base", "easy", "medium", "hard")


def default_tasks_root() -> Path | None:
    env = os.environ.get("MADI_BENCH_USECASES")
    if env:
        return Path(env)
    for candidate in ("use cases", "usecases"):
        root = Path(__file__).resolve().parents[2] / candidate
        if root.is_dir():
            return root
    return None


def add_task_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--domain", choices=tuple(ANCHOR_PRIORITY),
                    help="the task's domain (with --tier under --tasks-root)")
    ap.add_argument("--tier", default="base", choices=TIERS, help="difficulty tier (default: base)")
    ap.add_argument("--tasks-root", "--gold-root", dest="tasks_root", type=Path, default=None,
                    help="folder holding <domain>/<tier>/ (default: $MADI_BENCH_USECASES or the "
                         "repository's 'use cases/')")
    ap.add_argument("--task", type=Path, default=None,
                    help="the task folder itself, e.g. 'use cases/games/base' (instead of --domain)")
    ap.add_argument("--split", default="test", choices=("test", "validation"))
    ap.add_argument("--json", dest="json_out", type=Path, default=None,
                    help="also write the JSON result to this file")


def resolve_task(args: argparse.Namespace) -> Path:
    if args.task is not None:
        if args.domain is not None:
            raise ValueError("give either --task or --domain, not both")
        return args.task
    if args.domain is None:
        raise ValueError("name the task: --domain <domain> [--tier <tier>] or --task <folder>")
    root = args.tasks_root or default_tasks_root()
    if root is None:
        raise ValueError("no task tree found: pass --tasks-root (or set MADI_BENCH_USECASES)")
    task = root / args.domain / args.tier
    if not task.is_dir():
        raise FileNotFoundError(f"task folder not found: {task}")
    return task


def emit(result: dict, json_out: Path | None) -> None:
    text = json.dumps(result, indent=1, default=str)
    if json_out is not None:
        json_out.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")


def score_fusion_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="madi-score-fusion",
        description="Score a fused table against a MaDI-Bench task's fusion gold set (strict rules; "
                    "all-gold accuracy with coverage). Prints the result as JSON.")
    add_task_arguments(ap)
    ap.add_argument("--fused", required=True, type=Path, help="fused table CSV with an _id column")
    ap.add_argument("--membership", type=Path, default=None,
                    help="membership CSV (record_id, source, cluster_id); optional when the fused "
                         "table carries _fusion_sources")
    args = ap.parse_args(argv)
    try:
        task = resolve_task(args)
        fused = load_table(args.fused)
        membership = load_table(args.membership) if args.membership else None
        result = score_fusion(task, fused, membership, split=args.split)
    except (SubmissionError, FileNotFoundError, ValueError) as exc:
        print(f"madi-score-fusion: {exc}", file=sys.stderr)
        return 2
    emit(result, args.json_out)
    return 0


def score_normalization_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="madi-score-normalization",
        description="Score normalized source tables against a MaDI-Bench task's normalization test "
                    "set. Prints the result as JSON.")
    add_task_arguments(ap)
    ap.add_argument("--tables", required=True, type=Path,
                    help="directory of per-source CSVs (<source>.csv with an id column and the target "
                         "attributes) or one CSV with source, record_id and the attributes")
    ap.add_argument("--no-details", action="store_true", help="leave the per-row verdicts out")
    args = ap.parse_args(argv)
    try:
        task = resolve_task(args)
        result = score_normalization(task, args.tables, split=args.split)
    except (FileNotFoundError, ValueError) as exc:
        print(f"madi-score-normalization: {exc}", file=sys.stderr)
        return 2
    if args.no_details:
        result.pop("details", None)
    emit(result, args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(score_fusion_main())
