#!/usr/bin/env python
"""Table 6, P2 columns PC and RR: the P2 (best-of-breed) blocking stage only, replay mode, stored
configuration, plus a test-split re-scoring of every blocker's candidate set.

Run from the repository root, on a GPU node (SC-Block encodes the records), with an SC-Block checkpoint
under pipelines/<domain>/checkpoints/ (trained by cluster/slurm/run_bob_em_train.sbatch):

    python reproduction/tables/table6/p2_blocking_only.py --domain games \
        --sc-block-checkpoint-override pipelines/games/checkpoints/em_blocking/sc_block/baseline/best \
        --out p2_blocking_runs/games_baseline

What runs (all P2 code from this repository, nothing re-implemented):
  * config      reproduction/tables/table6/configs/<domain>_blocking_only.yaml (pipelines/configs/<domain>.yaml
                with sm/norm/refinement/fusion disabled; em enabled);
  * bundle      pipelines.lib.bundle.load_pipeline_bundle(domain, "baseline", config.bundle_source);
  * committee   the STORED effective committee YAMLs of the stored run
                (results/best of breeds/<domain>/baseline/effective_committees/), the
                sc_block checkpoint rewritten by BestOfBreedPipeline._maybe_filter_blocking_yaml
                exactly as --sc-block-checkpoint-override does in run_best_of_breed.py;
  * selection   stage_runners._swap_em_gold(split="val") -> EMCommitteeRunner.run (its
                blocking phase: every blocker on the full sources, pair recall on the swapped-in
                gold, _select_best_blocker per pair) -> stage_runners._build_em_blocking_selection.
                The matching phase is short-circuited (_run_matcher returns no predictions), so no
                matcher is built, trained or loaded;
  * test rescoring  every blocker's candidate set is additionally scored on the pair's TEST split
                (stage_runners._em_test_gold_for, the surface P2 reports EM test F1 on) with the
                pipeline's own blocking_pair_recall, and cross-checked against an independent parse
                of the shipped *_test.csv in the task folder ('use cases/<domain>/base', read in place).

Outputs (fresh run dir, created with exist_ok=False; numbers only, no gold rows, no candidate pairs):
  stage_3_em_blocking_selection.json   the stage record the pipeline would write
  per_stage_summary.csv, summary.md    one-row summary
  test_rescore.json                    per pair x blocker: val + test recall counts, RR, counts;
                                       the selected blockers' per-pair test PC / RR and their means;
                                       the comparison with the stored stage_3 record
  effective_committees/                the effective blocking committee YAML of this run
  run_meta.json                        arguments, timing, host, device

No LLM calls; API keys are removed from the environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _k in list(os.environ):
    if _k.endswith("API_KEY"):
        os.environ.pop(_k)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MB = Path(__file__).resolve().parents[3]                      # repository root
GOLD_ROOT = MB / "use cases"                                   # task folders (read in place)
HERE = Path(__file__).resolve().parent
STORED = MB / "results" / "best of breeds"
if str(MB) not in sys.path:
    sys.path.insert(0, str(MB))

import logging  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from pipelines.lib._resource_tracking import PeakRSSTracker  # noqa: E402
from pipelines.lib.bundle import PipelineState, load_pipeline_bundle  # noqa: E402
from pipelines.lib.pipeline import BestOfBreedPipeline, PipelineConfig  # noqa: E402
from pipelines.lib.report import _write_summary_csv  # noqa: E402
from pipelines.lib.stage_runners import (  # noqa: E402
    _build_em_blocking_selection,
    _em_test_gold_for,
    _restore_em_gold,
    _swap_em_gold,
)
from usecases_synthetic.lib import committee_em as ce  # noqa: E402
from usecases_synthetic.lib.committee_em_scoring import (  # noqa: E402
    _gold_positive_pairs,
    blocking_pair_recall,
)
from usecases_synthetic.lib.committee_paths import resolve_committee_path  # noqa: E402

logger = logging.getLogger("p2_blocking_only")

# Shipped test file per declared pair in the gold task folder (independent cross-check only).
TEST_FILES = {
    "games": {("dbpedia", "sales"): "dbpedia_2_sales_test.csv",
              ("metacritic", "dbpedia"): "dbpedia_2_metacritic_test.csv"},
    "companies": {("forbes", "dbpedia"): "forbes_2_dbpedia_test.csv",
                  ("forbes", "fullcontact"): "forbes_2_fullcontact_test.csv"},
    "music": {("musicbrainz", "discogs"): "musicbrainz_2_discogs_test.csv",
              ("musicbrainz", "lastfm"): "musicbrainz_2_lastfm_test.csv"},
    "products": {("products_1", "products_2"): "prod1_to_prod2_test.csv",
                 ("products_1", "products_3"): "prod1_to_prod3_test.csv",
                 ("products_1", "products_4"): "prod1_to_prod4_test.csv"},
    "papers": {("dblp", "crossref"): "dblp_crossref_test.csv",
               ("dblp", "open_alex"): "dblp_openalex_test.csv"},
}
# products: the canonical loader prefixes EM ids with their source (canonical_loader._prefix_ids).
ID_PREFIX = {"products": True}


def _canon(pairs) -> set[tuple[str, str]]:
    return {tuple(sorted((str(a), str(b)))) for a, b in pairs}


_TRUE = {"true", "1", "1.0"}
_FALSE = {"false", "0", "0.0"}


def independent_test_positives(domain: str, pair: tuple[str, str]) -> set[tuple[str, str]]:
    """Positive test pairs from the shipped file, parsed the way the paper's table protocol parses it
    (reproduction/scoring/), without pipeline code:
    header iff the last field of row 1 is no label; headered files select by column name,
    headerless rows are anchored on both ends (id1 = first field, label = last field)."""
    path = GOLD_ROOT / domain / "base" / "input" / "entitymatching" / TEST_FILES[domain][pair]
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and not (len(r) == 1 and not r[0].strip())]
    header = rows[0][-1].strip().lower() not in _TRUE | _FALSE
    out = []
    if header:
        names = [c.strip().lower() for c in rows[0]]
        lab = names.index("label")
        ids = [i for i in range(len(names)) if i != lab][:2]
        for r in rows[1:]:
            out.append((r[ids[0]], r[ids[1]], r[lab]))
    else:
        for r in rows:
            out.append((r[0], ",".join(r[1:-1]), r[-1]))
    pos = [(a, b) for a, b, lab_ in out if lab_.strip().lower() in _TRUE]
    assert all(lab_.strip().lower() in _TRUE | _FALSE for _, _, lab_ in out), path
    if ID_PREFIX.get(domain):
        pos = [(f"{pair[0]}_{a}", f"{pair[1]}_{b}") for a, b in pos]
    return _canon(pos)


class BlockingOnlyEMCommitteeRunner(ce.EMCommitteeRunner):
    """The P2 EM committee with its matching phase short-circuited.

    ``run`` / ``_run_pair`` / ``_select_best_blocker`` are the pipeline's own; only
    ``_run_matcher`` returns an empty prediction frame, so no matcher is ever built.
    """

    current_pair: tuple[str, str] | None = None

    def _run_pair(self, bundle, pair, gold_df, column_mapping):  # noqa: D401
        self.current_pair = tuple(pair)
        return super()._run_pair(bundle, pair, gold_df, column_mapping)

    def _run_matcher(self, spec, df_left, df_right, candidates, *, pair_train_path=None):
        return pd.DataFrame(columns=["id1", "id2", "score"])


class _CapturingBlocker:
    """Wraps a built blocker; scores its candidate set on the test split once materialized."""

    def __init__(self, inner, name: str, ctx: dict) -> None:
        self._inner = inner
        self._name = name
        self._ctx = ctx

    def materialize(self, *args, **kwargs):
        cands = self._inner.materialize(*args, **kwargs)
        ctx = self._ctx
        pair = ctx["runner"].current_pair
        key = f"{pair[0]}_{pair[1]}"
        test_gold = _em_test_gold_for(ctx["bundle"], pair)
        # Only candidate pairs touching a test-positive id can intersect the positives, so the
        # recall is computed on that subset (identical value, bounded memory for token_blocker).
        ids = ctx["pos_ids"][pair]
        if len(cands):
            a, b = cands["id1"].astype(str), cands["id2"].astype(str)
            sub = cands.loc[a.isin(ids) | b.isin(ids), ["id1", "id2"]]
        else:
            sub = pd.DataFrame(columns=["id1", "id2"])
        r = blocking_pair_recall(sub, test_gold)
        cset = _canon(zip(sub["id1"].astype(str), sub["id2"].astype(str)))
        ind_pos = ctx["independent_pos"][pair]
        ctx["capture"][(key, self._name)] = {
            "test_pair_recall": r["pair_recall"],
            "test_positives": int(r["gold_positives"]),
            "test_covered": int(r["covered"]),
            "independent_test_positives": len(ind_pos),
            "independent_test_covered": len(ind_pos & cset),
        }
        return cands

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _stored_run(domain: str) -> Path:
    return STORED / domain / "baseline"


def _effective_yaml(stored: Path, kind: str) -> Path:
    hits = sorted((stored / "effective_committees").glob(f"{kind}*.yaml"))
    assert len(hits) == 1, (stored, kind, hits)
    return hits[0]


def _members(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    out = {}
    for m in raw.get("members") or []:
        spec = dict(m.get("blocker") or {})
        params = dict(spec.get("params") or {})
        params.pop("checkpoint_path", None)
        out[m["name"]] = {"enabled": m.get("enabled_by_default", True), "class": spec.get("class"),
                          "module": spec.get("module"), "params": params}
    return {"members": out, "composition": raw.get("composition"),
            "column_mapping": raw.get("column_mapping"), "seed": raw.get("seed"),
            "preprocess_text": raw.get("preprocess_text"),
            "blocking_name_column": raw.get("blocking_name_column")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", required=True, choices=sorted(TEST_FILES))
    ap.add_argument("--config", type=Path, default=None,
                    help="blocking-only pipeline config (default: configs/<domain>_blocking_only.yaml)")
    ap.add_argument("--sc-block-checkpoint-override", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="fresh run folder (must not exist), e.g. p2_blocking_runs/<domain>_baseline")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s - %(message)s")

    domain = args.domain
    cfg_path = args.config or HERE / "configs" / f"{domain}_blocking_only.yaml"
    config = PipelineConfig.from_yaml(cfg_path)
    assert config.domain == domain, (config.domain, domain)
    enabled = {k: bool(v.get("enabled", True)) for k, v in config.stages.items()}
    assert enabled == {"sm": False, "norm": False, "em": True, "refinement": False, "fusion": False}, enabled

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)

    stored = _stored_run(domain)
    stored_blocking_yaml = _effective_yaml(stored, "em_blocking_committee")
    stored_matching_yaml = _effective_yaml(stored, "em_matching_committee")
    stored_s3 = json.loads((stored / "stage_3_em_blocking_selection.json").read_text())

    # The pipeline's own in-memory rewrite (writes out/effective_committees/<name>).
    bob = BestOfBreedPipeline(
        config,
        committee_dir=MB / "usecases_synthetic" / "config" / "committees",
        with_llm_sm=False, with_llm_em=False, with_llm_fusion=False,
        mode="replay", level="baseline",
        sc_block_checkpoint_override=args.sc_block_checkpoint_override,
        out_dir=out,
    )
    eff_blocking_yaml = bob._maybe_filter_blocking_yaml(stored_blocking_yaml)
    current_yaml = resolve_committee_path("em_blocking_committee", domain,
                                          committee_dir=MB / "usecases_synthetic" / "config" / "committees")
    same_as_current = _members(stored_blocking_yaml) == _members(current_yaml)

    t_all = time.monotonic()
    bundle = load_pipeline_bundle(domain, level="baseline", bundle_source=config.bundle_source)
    state = PipelineState(bundle=bundle)
    pairs = [tuple(p) for p in (list(bundle.em_gold) or list(bundle.em_splits))]
    independent_pos = {p: independent_test_positives(domain, p) for p in pairs}
    pipeline_test_pos = {p: _canon(_gold_positive_pairs(_em_test_gold_for(bundle, p))) for p in pairs}
    test_gold_equal = {f"{p[0]}_{p[1]}": pipeline_test_pos[p] == independent_pos[p] for p in pairs}
    pos_ids = {p: {x for pr in (pipeline_test_pos[p] | independent_pos[p]) for x in pr} for p in pairs}
    test_split_available = any(len(pipeline_test_pos[p]) > 0 for p in pairs)

    runner = BlockingOnlyEMCommitteeRunner(
        blocking_roster_path=eff_blocking_yaml,
        matching_roster_path=stored_matching_yaml,
        with_llm=False,
        clustering=str(config.stages.get("em", {}).get("clustering", "greedy")),
        retain_predictions_for=set(),
    )
    capture: dict[tuple[str, str], dict] = {}
    ctx = {"runner": runner, "bundle": bundle, "independent_pos": independent_pos, "pos_ids": pos_ids,
           "capture": capture}
    spec_name = {id(s.blocker_spec): s.name for s in runner._blocking_specs}
    orig_build = ce._build_blocker

    def _build_capturing(spec, df_left, df_right, id_column, **kw):
        return _CapturingBlocker(orig_build(spec, df_left, df_right, id_column, **kw),
                                 spec_name[id(spec)], ctx)

    ce._build_blocker = _build_capturing
    rss = PeakRSSTracker()
    rss.__enter__()
    t0 = time.monotonic()
    prev = _swap_em_gold(state, split="val")
    val_split_available = prev is not None
    selection_gold = {f"{p[0]}_{p[1]}": {"rows": int(len(g)), "positives": len(_gold_positive_pairs(g))}
                      for p, g in state.bundle.em_gold.items()}
    try:
        result = runner.run(state.bundle)
    finally:
        _restore_em_gold(state, prev)
        ce._build_blocker = orig_build
    runtime_s = time.monotonic() - t0
    rss.__exit__(None, None, None)

    sel, per_pair_winner = _build_em_blocking_selection(
        per_blocker=result.per_blocker,
        runtime_s=runtime_s,
        val_split_available=val_split_available,
        test_split_available=test_split_available,
    )
    sel.peak_memory_mb = rss.peak_mb
    (out / "stage_3_em_blocking_selection.json").write_text(json.dumps(sel.as_dict(), indent=2, default=str))
    _write_summary_csv(out / "per_stage_summary.csv", [sel])

    # ---------------------------------------------------------------- rescoring
    per_pair: dict[str, dict] = {}
    for p in pairs:
        key = f"{p[0]}_{p[1]}"
        rows = {}
        for name, member in result.per_blocker.items():
            pm = member.notes["per_pair"][key]
            cap = capture.get((key, name))
            rows[name] = {
                "selection_split_pair_recall": pm["pair_recall"],
                "selection_split_positives": int(pm["gold_positives"]),
                "selection_split_covered": int(pm["covered"]),
                "reduction_ratio": pm["reduction_ratio"],
                "candidate_count": int(pm["candidate_count"]),
                "full_space": int(pm["full_space"]),
                "selected": bool(pm["selected"]),
                "recall_floor_cleared_by_some_blocker": bool(pm["recall_floor_cleared"]),
                **({"test": cap} if cap is not None else {"test": None, "failed": True}),
            }
        per_pair[key] = rows

    def _composed(winners: dict[str, str]) -> dict:
        pc = {k: per_pair[k][w]["test"]["test_pair_recall"] for k, w in winners.items()}
        pc_ind = {k: per_pair[k][w]["test"]["independent_test_covered"] / per_pair[k][w]["test"]["independent_test_positives"]
                  for k, w in winners.items()}
        rr = {k: per_pair[k][w]["reduction_ratio"] for k, w in winners.items()}
        return {"winners": winners, "test_pc_per_pair": pc, "test_pc_per_pair_independent": pc_ind,
                "rr_per_pair": rr,
                "test_pc_mean": sum(pc.values()) / len(pc), "rr_mean": sum(rr.values()) / len(rr),
                "test_pc_mean_independent": sum(pc_ind.values()) / len(pc_ind)}

    rerun_sel = _composed(dict(per_pair_winner))
    stored_winners = dict(stored_s3["notes"]["per_pair_winner"])
    stored_sel = _composed(stored_winners)

    # Reproduction check: re-run per-member macro on the split the STORED run selected on.
    stored_split = "val" if stored_s3["notes"]["val_split_available"] else "test"
    repro = {}
    for name in stored_s3["per_member_val"]:
        if name not in result.per_blocker:
            repro[name] = {"stored_pc": stored_s3["per_member_val"][name], "rerun_pc": None}
            continue
        if stored_split == "val":
            vals = [per_pair[k][name]["selection_split_pair_recall"] for k in per_pair]
        else:
            vals = [per_pair[k][name]["test"]["test_pair_recall"] for k in per_pair]
        rrs = [per_pair[k][name]["reduction_ratio"] for k in per_pair]
        pc_now, rr_now = sum(vals) / len(vals), sum(rrs) / len(rrs)
        pc_st = stored_s3["per_member_val"][name]
        rr_st = stored_s3["notes"]["per_member_reduction_ratio_val"][name]
        repro[name] = {"stored_pc": pc_st, "rerun_pc": pc_now, "pc_equal": abs(pc_now - pc_st) < 1e-12,
                       "stored_rr": rr_st, "rerun_rr": rr_now, "rr_equal": abs(rr_now - rr_st) < 1e-12}
    report = {
        "domain": domain,
        "run_dir": str(out),
        "val_split_available": val_split_available,
        "test_split_available": test_split_available,
        "selection_gold_counts": selection_gold,
        "test_gold_pipeline_equals_independent_parse": test_gold_equal,
        "stored_run": str(stored),
        "stored_selection_split": stored_split,
        "stored_effective_blocking_yaml": str(stored_blocking_yaml),
        "stored_blocking_yaml_equals_current_committee_yaml": same_as_current,
        "current_committee_yaml": str(current_yaml),
        "per_pair": per_pair,
        "rerun_selection": rerun_sel,
        "stored_selection_rescored": stored_sel,
        "selection_differs_from_stored": dict(per_pair_winner) != stored_winners,
        "reproduction_vs_stored_per_member": repro,
        "runtime_s_blocking_only": runtime_s,
        "peak_memory_mb": rss.peak_mb,
    }
    (out / "test_rescore.json").write_text(json.dumps(report, indent=1, default=str))

    try:
        import torch  # noqa: PLC0415
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # pragma: no cover
        device = f"unknown ({exc})"
    meta = {"argv": sys.argv, "config": str(cfg_path), "utc_start": utc,
            "host": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
            "device": device, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "effective_blocking_yaml": str(eff_blocking_yaml),
            "matching_yaml_never_executed": str(stored_matching_yaml),
            "total_wall_s": time.monotonic() - t_all}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    (out / "summary.md").write_text(
        f"# P2 blocking stage only — {domain} base\n\n"
        f"Replay of the stored configuration ({stored_blocking_yaml.relative_to(MB)}), sc_block checkpoint "
        f"{args.sc_block_checkpoint_override}. Matching phase short-circuited (no matcher built).\n\n"
        f"| Stage | Winner | Metric | Val score | Test score (mirrors val) | Runtime (s) |\n|---|---|---|---|---|---|\n"
        f"| {sel.stage} | `{sel.winner}` | {sel.metric_key} | {sel.val_score:.4f} | {sel.test_score:.4f} | {sel.runtime_s:.1f} |\n\n"
        f"- val_split_available: {val_split_available}\n"
        f"- per-pair winners: {dict(per_pair_winner)} (stored: {stored_winners})\n"
        f"- selected blockers on the TEST split: PC mean {rerun_sel['test_pc_mean']:.6f}, RR mean {rerun_sel['rr_mean']:.6f}\n"
        f"- test_rescore.json holds the per-pair numbers.\n")

    print(json.dumps({"domain": domain, "val_split_available": val_split_available,
                      "rerun_winners": dict(per_pair_winner), "stored_winners": stored_winners,
                      "rerun_test_pc_mean": rerun_sel["test_pc_mean"], "rerun_rr_mean": rerun_sel["rr_mean"],
                      "stored_sel_test_pc_mean": stored_sel["test_pc_mean"], "stored_sel_rr_mean": stored_sel["rr_mean"],
                      "test_gold_equal": test_gold_equal,
                      "repro_all_equal": all(v.get("pc_equal") and v.get("rr_equal") for v in repro.values()),
                      "out": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
