#!/usr/bin/env python
"""Run the P2 entry point of the code tree given with --code (pipelines/scripts/run_best_of_breed.py main()) with
timing instrumentation. No pipeline logic is re-implemented.

    python reproduction/tables/table9/timed_bob.py --code . --timing-json <file> \
        [--per-pair-blockers <json>] -- <run_best_of_breed.py arguments>

Instrumentation (wrappers around the real functions; results unchanged):
  * wall-clock start/end of bundle loading, every stage runner, the e2e panel and the artifact
    writing (pipelines.lib.pipeline.* and run_best_of_breed.write_run_artifacts);
  * per source pair: each blocker's runtime / candidate count / pair recall and each matcher's
    runtime / prediction count, taken from the dict EMCommitteeRunner._run_pair returns.
With --per-pair-blockers {pair_key: blocker}, only the blocker the reported run selected for a
pair is built and run on that pair (companies: bm25_blocker on forbes_dbpedia, sc_block on
forbes_fullcontact); every other roster blocker is replaced by a stub that returns no candidates
without being built (no model load). The roster itself stays the reported one, because
EMCommitteeRunner._run_pair derives the blocking-key columns of every roster blocker on the pair
frames the matchers then read (see build_committees.py).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")          # as run_best_of_breed.py (set before numpy/torch)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

T_START = time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--timing-json", required=True)
    ap.add_argument("--per-pair-blockers", default=None)
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    sys.path.insert(0, a.code)
    import pipelines.scripts.run_best_of_breed as rbob  # noqa: E402  (it loads <code>/.env if present)
    from pipelines.lib import pipeline as pl  # noqa: E402
    from usecases_synthetic.lib import committee_em  # noqa: E402

    t_imported = time.time()
    events: list[dict] = []
    pairs: list[dict] = []

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                events.append({"name": name, "start": t0, "end": time.time(), "seconds": round(time.time() - t0, 3)})
        return wrapper

    for fname, label in [("load_pipeline_bundle", "load_bundle"), ("run_sm", "sm"), ("run_norm", "norm"),
                         ("run_em", "em"), ("run_refinement", "refinement"), ("run_fusion", "fusion")]:
        setattr(pl, fname, timed(label, getattr(pl, fname)))
    pl.BestOfBreedPipeline._compute_panel = timed("e2e_panel", pl.BestOfBreedPipeline._compute_panel)
    rbob.write_run_artifacts = timed("write_artifacts", rbob.write_run_artifacts)

    per_pair = json.loads(open(a.per_pair_blockers).read()) if a.per_pair_blockers else None
    orig_run_pair = committee_em.EMCommitteeRunner._run_pair
    orig_build_blocker = committee_em._build_blocker
    current: dict = {"run_spec": None, "skipped": []}

    class _SkippedBlocker:
        """Stands in for a roster blocker the reported run did not select on this pair."""

        def materialize(self):
            import pandas as pd
            return pd.DataFrame(columns=["id1", "id2"])

    def build_blocker(spec, *args, **kwargs):
        if current["run_spec"] is not None and spec is not current["run_spec"]:
            current["skipped"].append(f"{spec.get('class')}")
            return _SkippedBlocker()
        return orig_build_blocker(spec, *args, **kwargs)

    committee_em._build_blocker = build_blocker

    def run_pair(self, bundle, pair, *args, **kwargs):
        key = f"{pair[0]}_{pair[1]}"
        skipped_names: list[str] = []
        if per_pair is not None:
            want = per_pair.get(key)
            sel = [s for s in self._blocking_specs if s.name == want]
            if len(sel) != 1:
                raise RuntimeError(f"pair {key}: reported blocker {want!r} not in roster {[s.name for s in self._blocking_specs]}")
            current["run_spec"] = sel[0].blocker_spec
            skipped_names = [s.name for s in self._blocking_specs if s.name != want]
        t0 = time.time()
        try:
            res = orig_run_pair(self, bundle, pair, *args, **kwargs)
        finally:
            current["run_spec"] = None
        if per_pair is not None and res.get("winner_name") != per_pair.get(key):
            raise RuntimeError(f"pair {key}: selected {res.get('winner_name')!r}, reported {per_pair.get(key)!r}")
        pairs.append({
            "pair": key, "start": t0, "seconds": round(time.time() - t0, 3),
            "blocker_winner": res.get("winner_name"), "blockers_stubbed": skipped_names,
            "blockers": {n: {k: m.get(k) for k in ("runtime_s", "candidate_count", "pair_recall", "reduction_ratio", "full_space")}
                         for n, m in res.get("blocker_metrics", {}).items() if n not in skipped_names},
            "matchers": {n: {"runtime_s": round(rt, 3), "n_predictions": int(len(p)) if p is not None else None}
                         for n, (_m, p, rt) in res.get("matcher_outputs", {}).items()},
        })
        return res

    committee_em.EMCommitteeRunner._run_pair = run_pair

    device = {}
    try:
        import torch
        device = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as exc:  # pragma: no cover
        device = {"error": repr(exc)}

    sys.argv = [rbob.__file__] + rest
    rc, err = None, None
    try:
        rc = rbob.main()
    except BaseException as exc:  # record, then re-raise
        err = repr(exc)
        raise
    finally:
        t_end = time.time()
        payload = {
            "argv": rest, "code": a.code, "per_pair_blockers": per_pair,
            "host": platform.node(), "python": sys.version.split()[0], "device": device,
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK",
                                                    "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE")},
            "started_utc": datetime.fromtimestamp(T_START, timezone.utc).isoformat(),
            "t_start": T_START, "t_imports_done": t_imported, "t_end": t_end,
            "python_wall_s": round(t_end - T_START, 3), "imports_s": round(t_imported - T_START, 3),
            "events": events, "pairs": pairs, "return_code": rc, "error": err,
        }
        with open(a.timing_json, "w") as f:
            json.dump(payload, f, indent=1)
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
