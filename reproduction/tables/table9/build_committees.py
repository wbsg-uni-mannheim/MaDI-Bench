#!/usr/bin/env python
"""Best-only committees for timing the P2 pipeline that produced the reported numbers.

Differences from pipelines/scripts/build_best_only_committees.py:
  1. em_blocking keeps the reported run's FULL blocking roster; timed_bob.py builds and runs
     only the blocker the reported run selected for each source pair (stage_3
     notes.per_pair_winner) and replaces every other blocker by an empty stub. Two reasons:
     (a) per-pair winners differ in companies (bm25_blocker for forbes_dbpedia, sc_block for
     forbes_fullcontact), while build_best_only_committees.py keeps only the stage winner,
     bm25_blocker. (b) EMCommitteeRunner._run_pair derives the blocking-key columns of EVERY
     roster blocker (StandardBlocker 'on', SortedNeighbourhood 'key', e.g. name_first_3 /
     name_norm) on the pair's frames before any member runs, and the matchers see those columns:
     companies' Magellan built 44 / 52 comparator features over 4 / 6 attributes in the reported
     run, 22 / 30 over 2 / 4 with a single-blocker roster. Keeping the roster reproduces that
     input; the stubs cost ~0 s.
  2. ditto_plm gets params.cache_dir: "off". DittoMatcher's default per-batch inference cache
     (usecases_synthetic/cache/ditto_inference/) returns the scores it already holds for the same
     checkpoint and candidate pairs ("resumed N cached scores"); off, the timed run scores every
     candidate pair itself and writes nothing into the code tree. Scores are unaffected (the cache
     only stores them).
Everything else is as in build_best_only_committees.py: copy the code tree's committee dir, keep
only the winning member per stage (sm, norm, em_matching, fusion), restrict the config's refinement
methods to the winner. Winners are read from the reported run dir (--ref-run,
results/best of breeds/<d>/baseline; the SM winner is llm_openai in every domain).

    python reproduction/tables/table9/build_committees.py --domain games --code . \
        --ref-run "results/best of breeds/games/baseline" --out <run>/best_only

Writes <out>/committees/, <out>/<domain>.yaml, <out>/per_pair_blockers.json, <out>/build_log.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import yaml

STAGE_BASE = {
    "sm": "sm_committee",
    "norm": "normalization_committee",
    "em_blocking": "em_blocking_committee",
    "em_matching": "em_matching_committee",
    "fusion": "fusion_committee",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--code", required=True, type=Path, help="root of the code tree (this repository: .)")
    ap.add_argument("--ref-run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ditto-cache", default="off",
                    help="\"off\" (default): no Ditto inference cache; or a directory: a fresh cache there "
                         "(the reported code's default behaviour, one fsync per 16-pair batch)")
    a = ap.parse_args()

    sys.path.insert(0, str(a.code))
    from usecases_synthetic.lib.committee_paths import resolve_committee_path  # resolver of the code tree

    with (a.ref_run / "per_stage_summary.csv").open() as f:
        winners = {r["stage"]: r["winner"] for r in csv.DictReader(f)}
    st3 = json.loads((a.ref_run / "stage_3_em_blocking_selection.json").read_text())
    per_pair = dict(st3["notes"]["per_pair_winner"])
    keep = {s: [winners[s]] for s in STAGE_BASE if s in winners}
    # full reported blocking roster (per-pair winners run, the rest are stubbed by timed_bob.py)
    keep["em_blocking"] = sorted(st3.get("per_member_val", {}).keys()) or sorted(set(per_pair.values()))

    src = a.code / "usecases_synthetic" / "config" / "committees"
    out_c = a.out / "committees"
    if out_c.exists():
        raise SystemExit(f"refusing to overwrite {out_c}")
    shutil.copytree(src, out_c)

    log: dict = {"domain": a.domain, "code": str(a.code), "ref_run": str(a.ref_run),
                 "winners": winners, "per_pair_blocker_winner": per_pair, "rosters": {}}
    for stage, base in STAGE_BASE.items():
        path = resolve_committee_path(base, a.domain, committee_dir=out_c)
        doc = yaml.safe_load(path.read_text())
        names = [str(m.get("name")) for m in doc.get("members") or []]
        kept = [m for m in doc["members"] if str(m.get("name")) in keep[stage]]
        missing = set(keep[stage]) - {str(m.get("name")) for m in kept}
        if missing:
            raise SystemExit(f"{stage}: winner(s) {sorted(missing)} not in {path.name} roster {names}")
        if stage == "em_matching":
            for m in kept:
                if m.get("name") == "ditto_plm":
                    m.setdefault("matcher", {}).setdefault("params", {})["cache_dir"] = a.ditto_cache
        doc["members"] = kept
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        log["rosters"][stage] = {"file": path.name, "full": names, "kept": [m["name"] for m in kept]}

    # The reported run stored the committees it rewrote (effective_committees/); compare their
    # member specs (checkpoint paths aside) with the snapshot's rosters.
    def _members(doc: dict) -> dict:
        out = {}
        for m in doc.get("members") or []:
            m = json.loads(json.dumps(m))
            for sub in ("blocker", "matcher"):
                (m.get(sub) or {}).get("params", {}).pop("checkpoint_path", None)
            out[str(m.get("name"))] = m
        return out

    eff = {}
    for p in sorted((a.ref_run / "effective_committees").glob("*.yaml")):
        rep = _members(yaml.safe_load(p.read_text()))
        snap = _members(yaml.safe_load((src / p.name).read_text()))
        eff[p.name] = {"members_equal": rep == snap,
                       "differing": sorted(n for n in set(rep) | set(snap) if rep.get(n) != snap.get(n))}
    log["effective_committee_check"] = eff

    cfg = yaml.safe_load((a.code / "pipelines" / "configs" / f"{a.domain}.yaml").read_text())
    cfg.setdefault("stages", {}).setdefault("refinement", {})["methods"] = [winners.get("refinement", "baseline")]
    (a.out / f"{a.domain}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log["rosters"]["refinement"] = {"file": f"{a.domain}.yaml", "kept": [winners.get("refinement", "baseline")]}
    log["ditto_cache"] = a.ditto_cache if "ditto_plm" in keep["em_matching"] else "n/a"
    (a.out / "per_pair_blockers.json").write_text(json.dumps(per_pair, indent=1) + "\n")
    (a.out / "build_log.json").write_text(json.dumps(log, indent=1) + "\n")
    for s, r in log["rosters"].items():
        print(f"[best-only] {s}: {r['kept']}")
    print(f"[best-only] per-pair blockers: {per_pair}; ditto cache: {log['ditto_cache']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
