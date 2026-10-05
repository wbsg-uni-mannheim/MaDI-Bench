#!/usr/bin/env python
"""Score one system's outputs for one task, stage by stage (e.g. a P4 run).

The submission folder holds any of (the layout the Claude Code baseline's
harness scored, file by file):

  sm_mapping.csv           schema mapping: source_dataset, source_column,
                           target_dataset, target_column, score
  blocking/candidates.csv  candidate pairs: id1, id2
  correspondences.csv      matched pairs: id1, id2[, score]
  fused.csv                fused table with _id (+ the target attributes)
  membership.csv           record_id, source, cluster_id (cluster_id = fused _id)
  normalization/           normalized source tables, one CSV per source

Each stage present is scored against the task's test gold:

  schema matching   score_sm.py: precision / recall / F1 of the correspondences
                    (PyDI SchemaMappingEvaluator)
  blocking          score_blocking.py: pair completeness and pair quality per
                    source pair (PyDI EntityMatchingEvaluator); the mean over the
                    pairs is reported as pair_completeness_mean; the reduction
                    ratio per pair is derived from the candidate count and the
                    source sizes (reduction_ratio_derived, reduction_ratio_mean)
  entity matching   score_em.py: precision / recall / F1 per source pair (PyDI
                    EntityMatchingEvaluator); the paper reports per_pair_mean.f1,
                    the unweighted mean over the source pairs
  fusion            madi_bench.evaluation.score_fusion (all-gold accuracy)
  normalization     madi_bench.evaluation.score_normalization
  end-to-end        score_e2e.py (reference-free and ground-truth rows; with --e2e)

Variant tiers: the entity-matching and blocking gold is the
``*_test_corner_filled.csv`` split (``--variant-gold`` to change).

    python -m reproduction.scoring.score_submission --task "use cases/games/base" \\
        --submission /path/to/submission [--e2e] [--json scores.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd  # noqa: E402

from . import REPO_ROOT, SubmissionContractError  # noqa: E402,F401  (repository root on sys.path)
from .gold import VARIANT_GOLD_POLICIES  # noqa: E402
from .tasks import task_from_dir  # noqa: E402

STAGES = ("sm", "blocking", "em", "fusion", "normalization", "e2e")


def source_id_sets(task) -> dict[str, set[str]]:
    """Record ids per source table of the task (CSV: first column; JSON /
    JSON lines: ``id``), keyed by the file stem."""
    import json

    out: dict[str, set[str]] = {}
    for p in sorted((task.root / "input" / "data").iterdir()):
        if not p.is_file() or p.name.endswith("_metadata.json"):
            continue
        if p.suffix == ".csv":
            out[p.stem] = set(pd.read_csv(p, dtype=str, usecols=[0]).iloc[:, 0])
        elif p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as fh:
                out[p.stem] = {str(json.loads(line)["id"]) for line in fh if line.strip()}
        elif p.suffix == ".json":
            out[p.stem] = set(pd.read_json(p)["id"].astype(str))
    return out


def derived_reduction_ratios(task, candidates: pd.DataFrame, test_pairs: dict[str, pd.DataFrame]) -> dict:
    """Reduction ratio per evaluated source pair, as the paper derived it for
    P4: 1 - (distinct candidate rows between the pair's two sources) /
    (|source A| x |source B|). The two sources of a pair file are the sources
    its test ids belong to."""
    ids = source_id_sets(task)
    owner: dict[str, str] = {}
    for src, members in ids.items():
        for i in members:
            owner.setdefault(i, src)
    c = candidates.astype(str).drop_duplicates()
    s1, s2 = c["id1"].map(owner), c["id2"].map(owner)
    out = {}
    for pair, test in test_pairs.items():
        a = test["id1"].astype(str).map(owner).mode()
        b = test["id2"].astype(str).map(owner).mode()
        if a.empty or b.empty:
            out[pair] = {"sources": None, "n_candidates": None, "reduction_ratio": None}
            continue
        a, b = a.iloc[0], b.iloc[0]
        n = int((((s1 == a) & (s2 == b)) | ((s1 == b) & (s2 == a))).sum())
        out[pair] = {"sources": [a, b], "n_candidates": n,
                     "reduction_ratio": 1 - n / (len(ids[a]) * len(ids[b]))}
    return out


def score_stages(task, folder: Path, stages, variant_gold: str | None = None) -> dict:
    from madi_bench.evaluation import score_fusion, score_normalization

    from .score_blocking import load_candidates, score_blocking
    from .score_em import load_correspondences, score_em
    from .score_sm import score_sm

    out: dict = {"task": task.task_id, "task_root": str(task.root), "submission": str(folder), "stages": {}}

    def guarded(stage, fn):
        try:
            out["stages"][stage] = fn()
        except SubmissionContractError as exc:
            out["stages"][stage] = {"error": f"contract: {exc}"}
        except Exception as exc:  # noqa: BLE001 -- one broken stage must not hide the others
            out["stages"][stage] = {"error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()}

    if "sm" in stages and (folder / "sm_mapping.csv").is_file():
        guarded("schema_matching", lambda: score_sm(task, folder / "sm_mapping.csv"))
    if "blocking" in stages and (folder / "blocking" / "candidates.csv").is_file():
        def blocking():
            from .gold import load_em_test_pairs

            candidates = load_candidates(folder / "blocking" / "candidates.csv")
            result = score_blocking(task, candidates, variant_gold=variant_gold)
            pcs = [p["pair_completeness"] for p in result["per_pair"].values()
                   if p.get("pair_completeness") is not None]
            result["pair_completeness_mean"] = sum(pcs) / len(pcs) if pcs else None
            rr = derived_reduction_ratios(task, candidates, load_em_test_pairs(task, variant_gold))
            result["reduction_ratio_derived"] = rr
            rrs = [v["reduction_ratio"] for v in rr.values() if v["reduction_ratio"] is not None]
            result["reduction_ratio_mean"] = sum(rrs) / len(rrs) if rrs else None
            return result
        guarded("blocking", blocking)
    if "em" in stages and (folder / "correspondences.csv").is_file():
        guarded("entity_matching", lambda: score_em(task, folder / "correspondences.csv", variant_gold=variant_gold))
    fused_path, mem_path = folder / "fused.csv", folder / "membership.csv"
    if ("fusion" in stages or "e2e" in stages) and fused_path.is_file():
        fused = pd.read_csv(fused_path, dtype={"_id": str})
        membership = pd.read_csv(mem_path, dtype=str) if mem_path.is_file() else None
        if "fusion" in stages:
            guarded("fusion", lambda: score_fusion(task.root, fused, membership))
        if "e2e" in stages:
            def e2e():
                from .score_e2e import score_e2e
                corr_path = folder / "correspondences.csv"
                corr = (load_correspondences(corr_path) if corr_path.is_file()
                        else pd.DataFrame(columns=["id1", "id2", "score"]))
                return score_e2e(task, fused, corr, membership=membership)
            guarded("e2e_panel", e2e)
    if "normalization" in stages and (folder / "normalization").is_dir():
        guarded("normalization", lambda: score_normalization(task.root, folder / "normalization"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--task", type=Path, required=True, help="the task folder, e.g. 'use cases/games/base'")
    ap.add_argument("--submission", type=Path, required=True, help="the system's output folder")
    ap.add_argument("--stages", default="sm,blocking,em,fusion,normalization",
                    help=f"comma-separated subset of {','.join(STAGES)} (default: all but e2e)")
    ap.add_argument("--e2e", action="store_true", help="also compute the end-to-end panel (slow)")
    ap.add_argument("--variant-gold", choices=VARIANT_GOLD_POLICIES, default=None,
                    help="EM/blocking gold of a variant tier (default: corner_filled)")
    ap.add_argument("--json", dest="json_out", type=Path, default=None, help="also write the result here")
    args = ap.parse_args(argv)
    stages = {s for s in args.stages.split(",") if s} | ({"e2e"} if args.e2e else set())
    unknown = stages - set(STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")
    logging.disable(logging.CRITICAL)
    result = score_stages(task_from_dir(args.task), args.submission, stages, args.variant_gold)
    text = json.dumps(result, indent=1, default=str)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    return 1 if any("error" in v for v in result["stages"].values() if isinstance(v, dict)) else 0


if __name__ == "__main__":
    sys.exit(main())
