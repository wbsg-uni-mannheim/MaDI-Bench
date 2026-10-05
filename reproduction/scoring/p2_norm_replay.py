#!/usr/bin/env python3
"""Replay stage 2 (normalization) of the MaDI-Bench best-of-breed pipeline (P2)
and export its members' normalized source tables.

The P2 normalization tables scored in ``results/scores_v2.json`` (paper
Tables 5 and 13, P2 column) come from this replay. Score its output with
``rescore.py --p2-norm-root <out-root>``.

P2's stage 2 (``pipelines/lib/stage_runners.py::run_norm``) runs the
normalization committee (``usecases_synthetic/lib/committee_norm_c12.py``) over
``state.bundle`` with two members, ``passthrough`` and
``rule_per_attribute_optimal``, picks the member with the higher ``macro_f1`` on
the ``schema_constraints`` surface and records ``stage_2_norm_selection.json``
with ``applied_to_downstream = false``: the normalized output is not used
downstream. This script re-runs the stage exactly as a stored run did and
exports what it produced.

What is replayed exactly
------------------------
* bundle: ``pipelines.lib.bundle.load_pipeline_bundle(domain, level=tier,
  bundle_source=<config>)`` - the stored run's loader. Stage 2 reads the RAW
  source frames of the bundle and locates each target attribute through the
  bundle's SM *gold* mapping (``bundle.sm_mapping``), not through the stage-1
  winner, and stage 1 does not modify the bundle, so stage 1 (LLM-backed) is
  not needed and is not run.
* committee YAML, epsilon, surface: ``pipelines/configs/<domain>.yaml`` ->
  ``stages.norm`` and ``resolve_committee_path("normalization_committee", ...)``
  under ``usecases_synthetic/config/committees`` (the CLI default).
* the stage itself: ``run_norm(state, norm_yaml=..., vacuous_epsilon=...,
  apply_winner=..., scoring_surface=...)`` is called unchanged. ``llm_only`` is
  dropped by the runner (``with_llm`` defaults to False in ``run_norm``), so no
  LLM or API call happens.

Redirected (so that nothing is written into the MaDI-Bench checkout)
------------------------------------------------------------------
* the rule-selection cache ``usecases_synthetic/baselines/<domain>/
  norm_committee_selection.json`` (the per-attribute rule choice of
  ``rule_per_attribute_optimal``, locked once by a sweep on the fusion
  VALIDATION gold values). ``--rule-selection cache`` (default) copies a lock
  file with the rules of the stored runs (``--selection-cache``, default
  ``reproduction/scoring/p2_norm_rule_selection/<domain>.json``: rule names per
  attribute only) into the output directory and points the runner at that
  copy; ``--rule-selection sweep`` points it at an empty location in the
  output directory, so the runner sweeps on the task's fusion validation gold
  and saves the lock there.
* the LLM operation-log directory (``_op_log_dir``), created unconditionally.
* byte-code: ``sys.dont_write_bytecode`` is set before any checkout import.

Export
------
The members' normalizer objects are captured from the runner (no
re-implementation) and applied to EVERY record of every source: for each
(source, target attribute) of the SM gold mapping the cell is
``member.normalize(raw, attribute=..., kind=kind_map[attr], domain=...)``,
exactly the call the stage scores with. Attributes outside the stage's scope
(not schema-constrained, or no rule selected) go through the member's
passthrough fallback, i.e. they keep their raw value, as they would downstream.
A member that abstains (returns None) yields an empty cell. When the SM gold
maps several source columns to one attribute, the first non-empty value in
mapping order is kept (counted in ``export_manifest.json``). Output format: one
``<source>.csv`` per source with the record's real id in ``id`` plus the target
attributes, the input format of ``madi_bench.evaluation.normalization_score``.

The output folder holds the normalized tables (pipeline output) and, with
``--score``, per-cell score details keyed on the normalization test set; keep
it outside the repository.

Usage (from the repository root; about 1-2 minutes and up to 16 GB per task)::

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONDONTWRITEBYTECODE=1 \\
      python -m reproduction.scoring.p2_norm_replay \\
      --domain companies --tier baseline --out-root /tmp/p2_norm_replay --score
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

DEFAULT_MADI_ROOT = Path(__file__).resolve().parents[2]       # this repository
RULE_SELECTION_DIR = Path(__file__).resolve().parent / "p2_norm_rule_selection"
DOMAINS = ("companies", "games", "music", "papers", "products")
TIERS = ("baseline", "easy", "medium", "hard")

logger = logging.getLogger("p2_norm_replay")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--domain", required=True, choices=DOMAINS)
    p.add_argument("--tier", required=True, choices=TIERS)
    p.add_argument("--madi-root", type=Path, default=DEFAULT_MADI_ROOT,
                   help="MaDI-Bench checkout whose code and task data are replayed.")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Output root (outside the repository); the task's folder "
                        "<domain>_<tier> is created in it.")
    p.add_argument("--out-name", default=None,
                   help="Output folder name (default <domain>_<tier>, "
                        "suffixed _sweep in sweep mode).")
    p.add_argument("--rule-selection", choices=("cache", "sweep"), default="cache",
                   help="cache: the per-attribute rule lock the stored runs read; "
                        "sweep: a fresh val-gold sweep (saved in the output dir only).")
    p.add_argument("--selection-cache", type=Path, default=None,
                   help="Lock file for --rule-selection cache (default: the rules of the stored "
                        "runs, reproduction/scoring/p2_norm_rule_selection/<domain>.json).")
    p.add_argument("--stored-json", type=Path, default=None,
                   help="Stored stage_2_norm_selection.json to compare against "
                        "(default: <madi-root>/results/best of breeds/<domain>/<tier>/).")
    p.add_argument("--no-export", action="store_true",
                   help="Replay and compare only; do not export tables.")
    p.add_argument("--score", action="store_true",
                   help="Score the exported winner and passthrough tables with the "
                        "public normalization scorer (test and validation splits).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="[%(levelname)s] %(name)s - %(message)s")
    madi_root = args.madi_root.resolve()
    os.chdir(madi_root)
    sys.path.insert(0, str(madi_root))

    out_name = args.out_name or (
        f"{args.domain}_{args.tier}" + ("_sweep" if args.rule_selection == "sweep" else "")
    )
    out_dir = (args.out_root / out_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- checkout imports (after sys.path is set) -------------------------
    import usecases_synthetic
    import pipelines
    import PyDI
    from usecases_synthetic.lib import committee_norm_c12 as c12
    from usecases_synthetic.lib.committee_norm import _build_source_attribute_index
    from usecases_synthetic.lib.committee_paths import resolve_committee_path
    from usecases_synthetic.lib.domain_config import _resolve_knob_config_alias
    from usecases_synthetic.lib.protection import kind_map_for_domain
    from pipelines.lib.bundle import PipelineState, load_pipeline_bundle
    from pipelines.lib.pipeline import PipelineConfig
    from pipelines.lib.stage_runners import run_norm

    for mod in (usecases_synthetic, pipelines):
        mod_path = Path(mod.__file__).resolve()
        if madi_root not in mod_path.parents:
            raise RuntimeError(f"{mod.__name__} imported from {mod_path}, not {madi_root}")

    # ---- redirect the rule-selection lock + op-log dir --------------------
    canonical_domain = _resolve_knob_config_alias(args.domain) or args.domain
    work_cache = out_dir / "norm_committee_selection.json"
    input_cache_copy = out_dir / "norm_committee_selection.input.json"
    for stale in (work_cache, input_cache_copy):
        if stale.exists():
            stale.unlink()
    selection_source: dict[str, Any] = {"mode": args.rule_selection}
    if args.rule_selection == "cache":
        src = args.selection_cache or (RULE_SELECTION_DIR / f"{canonical_domain}.json")
        if not src.is_file():
            raise FileNotFoundError(f"selection cache not found: {src}")
        shutil.copyfile(src, input_cache_copy)
        shutil.copyfile(src, work_cache)
        selection_source.update(path=str(src), sha256=_sha256(src),
                                mtime=datetime.fromtimestamp(
                                    src.stat().st_mtime, timezone.utc).isoformat())

    c12._selection_cache_path = lambda domain: work_cache  # noqa: E731

    def _op_log_dir(domain: str, level: str) -> Path:
        d = out_dir / "norm_diagnostics" / domain / level
        d.mkdir(parents=True, exist_ok=True)
        return d

    c12._op_log_dir = _op_log_dir

    # ---- capture the members' normalizer objects + the committee result ---
    captured: dict[str, Any] = {}

    class _CapturingPassthrough(c12._PassthroughNormalizer):
        def __init__(self) -> None:
            captured["passthrough"] = self

    class _CapturingComposite(c12._CompositeRuleNormalizer):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            captured["rule_per_attribute_optimal"] = self

    c12._PassthroughNormalizer = _CapturingPassthrough
    c12._CompositeRuleNormalizer = _CapturingComposite
    _orig_run = c12.C12NormCommitteeRunner.run

    def _capturing_run(self: Any, bundle: Any) -> Any:
        result = _orig_run(self, bundle)
        captured["result"] = result
        captured["runner"] = self
        return result

    c12.C12NormCommitteeRunner.run = _capturing_run

    # ---- config + bundle, exactly as BestOfBreedPipeline.run ---------------
    config_path = madi_root / "pipelines" / "configs" / f"{args.domain}.yaml"
    config = PipelineConfig.from_yaml(config_path)
    stage_cfg = config.stages.get("norm", {})
    if not stage_cfg.get("enabled", True):
        raise RuntimeError(f"norm stage disabled in {config_path}")
    loader_adaptations: list[str] = []
    try:
        bundle = load_pipeline_bundle(config.domain, level=args.tier,
                                      bundle_source=config.bundle_source)
    except Exception as exc:  # noqa: BLE001
        if not (config.domain == "products" and args.tier == "baseline"
                and config.bundle_source == "canonical"):
            raise
        loader_adaptations.append(
            f"load_canonical_products_bundle raised {type(exc).__name__}: {exc}; "
            "bundle rebuilt from the loader's own source/EM/SM/schema helpers with "
            "the fusion frames omitted (stage 2 never reads bundle.fusion_*)")
        bundle = _products_bundle_without_fusion()
    if config.domain == "products" and args.tier == "baseline" and not bundle.em_gold:
        em_gold, em_splits, em_dir = _products_em_public_layout()
        if em_gold:
            bundle.em_gold, bundle.em_splits = em_gold, em_splits
            loader_adaptations.append(
                f"canonical loader found no EM gold; read the prod1_to_prodN files from {em_dir}")
    state = PipelineState(bundle=bundle)
    committee_dir = madi_root / "usecases_synthetic" / "config" / "committees"
    norm_yaml = resolve_committee_path("normalization_committee", config.domain,
                                       committee_dir=committee_dir)
    vacuous_epsilon = float(stage_cfg.get("vacuous_epsilon", 0.005))
    apply_winner = bool(stage_cfg.get("apply_winner", False))
    scoring_surface = str(stage_cfg.get("scoring_surface", "schema_constraints"))

    sel = run_norm(state, norm_yaml=norm_yaml, vacuous_epsilon=vacuous_epsilon,
                   apply_winner=apply_winner, scoring_surface=scoring_surface)
    result = captured["result"]
    runner = captured["runner"]

    # ---- selection lock actually used -------------------------------------
    rule_member = result.per_member.get("rule_per_attribute_optimal")
    selection_map = dict((rule_member.notes or {}).get("selection_map", {})) if rule_member else {}
    eligible_attributes = sorted(selection_map)
    lock_after = json.loads(work_cache.read_text()) if work_cache.exists() else None
    lock_before = json.loads(input_cache_copy.read_text()) if input_cache_copy.exists() else None
    selection_source["lock_written_by_runner"] = lock_after != lock_before
    lock_entries = (lock_after or {}).get("rule_per_attribute_optimal", {})
    selection_source["map_differs_from_lock_file"] = any(
        lock_entries.get(a) != v for a, v in selection_map.items())

    # entity sets the surface was scoped to
    val_targets, test_targets = c12._load_eval_targets_for_bundle(bundle.domain, bundle)

    # ---- stored comparison -------------------------------------------------
    stored_path = args.stored_json or (madi_root / "results" / "best of breeds" /
                                       args.domain / args.tier / "stage_2_norm_selection.json")
    replay = sel.as_dict()
    comparison: dict[str, Any] = {"stored_path": str(stored_path)}
    if stored_path.is_file():
        stored = json.loads(stored_path.read_text())
        comparison.update(
            stored_winner=stored.get("winner"),
            stored_per_member_val=stored.get("per_member_val"),
            winner_match=stored.get("winner") == replay["winner"],
            per_member_val_exact_match=stored.get("per_member_val") == replay["per_member_val"],
            per_member_val_abs_diff={
                m: (replay["per_member_val"].get(m, float("nan")) - v)
                for m, v in (stored.get("per_member_val") or {}).items()},
            spread_match=(stored.get("notes") or {}).get("spread") == replay["notes"]["spread"],
            vacuous_match=(stored.get("notes") or {}).get("vacuous") == replay["notes"]["vacuous"],
            note="runtime_s and peak_memory_mb are not compared",
        )
    else:
        comparison["note"] = "no stored selection file"

    member_detail = {}
    for name, m in result.per_member.items():
        member_detail[name] = {
            "metrics": {k: v for k, v in m.metrics.items()},
            "per_attribute_test": m.predictions.to_dict(orient="records"),
        }

    replay["replay"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "domain": args.domain,
        "tier": args.tier,
        "madi_root": str(madi_root),
        "code": {"usecases_synthetic": usecases_synthetic.__file__,
                 "pipelines": pipelines.__file__, "PyDI": PyDI.__file__,
                 "python": sys.executable},
        "pipeline_config": str(config_path),
        "bundle_source": config.bundle_source,
        "loader_adaptations": loader_adaptations,
        "em_gold_pairs": {f"{a}|{b}": int(len(df)) for (a, b), df in (bundle.em_gold or {}).items()},
        "norm_yaml": str(norm_yaml),
        "norm_yaml_sha256": _sha256(norm_yaml),
        "scoring_surface": scoring_surface,
        "vacuous_epsilon": vacuous_epsilon,
        "apply_winner": apply_winner,
        "roster_active": list(runner.roster_names),
        "rule_selection": selection_source,
        "selection_map": selection_map,
        "eligible_attributes": eligible_attributes,
        "n_entities_val": len(val_targets),
        "n_entities_test": len(test_targets),
        "note_metric": ("per_member_val/val_score hold the committee's headline "
                        "macro_f1, which run_norm reads from the TEST entity set; "
                        "f1_val in member_detail is the validation entity set"),
        "sources": {s: int(len(df)) for s, df in bundle.sources.items()},
        "member_detail": member_detail,
        "comparison": comparison,
    }
    (out_dir / "stage_2_norm_selection_replay.json").write_text(
        json.dumps(replay, indent=2, default=str) + "\n")
    logger.info("winner=%s per_member=%s comparison=%s", replay["winner"],
                replay["per_member_val"],
                {k: comparison.get(k) for k in ("winner_match", "per_member_val_exact_match")})

    # ---- export -------------------------------------------------------------
    if not args.no_export:
        attr_index = _build_source_attribute_index(bundle.sm_mapping, bundle.knob_08_renames)
        kind_map = kind_map_for_domain(bundle.domain)
        manifest: dict[str, Any] = {"winner": replay["winner"], "members": {}, "tables": {}}
        targets = [(replay["winner"], out_dir / "tables")]
        targets.append(("passthrough", out_dir / "tables_passthrough"))
        if replay["winner"] != "rule_per_attribute_optimal":
            # the losing rule member, for context only
            targets.append(("rule_per_attribute_optimal",
                            out_dir / "tables_rule_per_attribute_optimal"))
        for member_name, table_dir in targets:
            member = captured[member_name]
            manifest["tables"][table_dir.name] = member_name
            manifest["members"][member_name] = _export_member(
                member=member, member_name=member_name, bundle=bundle,
                attr_index=attr_index, kind_map=kind_map,
                eligible=set(eligible_attributes), selection_map=selection_map,
                table_dir=table_dir)
        manifest["kind_map"] = kind_map
        (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        if args.score:
            from madi_bench.evaluation.normalization_score import score_normalization

            task_tier = "base" if args.tier == "baseline" else args.tier
            task_dir = madi_root / "use cases" / args.domain / task_tier
            scores: dict[str, Any] = {"task_dir": str(task_dir)}
            scored = [("winner", out_dir / "tables"),
                      ("passthrough", out_dir / "tables_passthrough")]
            if (out_dir / "tables_rule_per_attribute_optimal").is_dir():
                scored.append(("rule_per_attribute_optimal",
                               out_dir / "tables_rule_per_attribute_optimal"))
            for label, table_dir in scored:
                scores[label] = {}
                for split in ("test", "validation"):
                    r = score_normalization(task_dir, table_dir, split=split)
                    details = r.pop("details")
                    r["n_not_found"] = sum(1 for d in details if not d["found"])
                    scores[label][split] = r
                    if split == "test":
                        _write_details(out_dir / f"score_details_{label}_test.csv", details)
            scores["winner_member"] = replay["winner"]
            (out_dir / "normalization_scores.json").write_text(
                json.dumps(scores, indent=2) + "\n")
            for label, _ in scored:
                t = scores[label]["test"]
                logger.info("score %s (test): accuracy=%s transformations=%s coverage=%s rows=%s",
                            label, t["accuracy"], t["accuracy_transformations"],
                            t["record_coverage"], t["rows"])

    # ---- guard: nothing written into the checkout --------------------------
    written = _files_modified_since(madi_root, t_start)
    replay_path = out_dir / "stage_2_norm_selection_replay.json"
    data = json.loads(replay_path.read_text())
    data["replay"]["checkout_files_modified_during_run"] = written
    data["replay"]["runtime_total_s"] = time.time() - t_start
    replay_path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    if written:
        logger.warning("files modified in the checkout during the run: %s", written)
    print(json.dumps({"out_dir": str(out_dir), "winner": replay["winner"],
                      "per_member_val": replay["per_member_val"],
                      "spread": replay["notes"]["spread"],
                      "vacuous": replay["notes"]["vacuous"],
                      "comparison": {k: comparison.get(k) for k in (
                          "winner_match", "per_member_val_exact_match",
                          "per_member_val_abs_diff")},
                      "checkout_writes": written}, indent=2))
    return 0


def _products_em_public_layout() -> tuple[dict, dict, Path]:
    """``canonical_loader._load_canonical_em`` with the EM folder fixed to
    ``input/entitymatching/`` (the prod1_to_prodN_* files)."""
    from pipelines.lib import canonical_loader as cl
    from usecases_synthetic.lib.domain_config import task_dir

    em_dir = task_dir("products") / "input" / "entitymatching"
    em_gold: dict = {}
    em_splits: dict = {}
    for src1, src2, stem in cl._PRODUCTS_EM_PAIRS:
        all_path = em_dir / f"{stem}_all.csv"
        if not all_path.exists():
            continue
        em_gold[(src1, src2)] = cl._read_canonical_em_csv(all_path, src1, src2)
        pair_splits = {}
        for split in ("train", "val", "test", "all"):
            sp_path = em_dir / f"{stem}_{split}.csv"
            if sp_path.exists():
                pair_splits[split] = cl._read_canonical_em_csv(sp_path, src1, src2)
        if pair_splits:
            em_splits[(src1, src2)] = pair_splits
    return em_gold, em_splits, em_dir


def _products_bundle_without_fusion() -> Any:
    """``canonical_loader.load_canonical_products_bundle`` without the fusion
    frames (stage 2 does not read them), used when that loader raises."""
    from pipelines.lib import canonical_loader as cl
    from usecases_synthetic.lib.domain_config import task_dir
    from usecases_synthetic.lib.variant_loader import VariantBundle

    root = task_dir("products")
    em_gold, em_splits = cl._load_canonical_em(root)
    return VariantBundle(
        domain="products",
        level="baseline",
        sources=cl._load_canonical_sources(root),
        target_schema=cl._load_canonical_target_schema(root),
        sm_mapping=cl._load_canonical_sm_gold(root),
        em_gold=em_gold,
        em_splits=em_splits,
        fusion_gold=None,
        fusion_validation=None,
        pooled_positives=None,
        variant_root=root,
    )


def _export_member(*, member: Any, member_name: str, bundle: Any,
                   attr_index: dict[tuple[str, str], list[str]],
                   kind_map: dict[str, str], eligible: set[str],
                   selection_map: dict[str, str], table_dir: Path) -> dict[str, Any]:
    """Apply one member to every record of every source and write <source>.csv."""
    if table_dir.exists():
        shutil.rmtree(table_dir)
    table_dir.mkdir(parents=True)
    domain = bundle.domain
    info: dict[str, Any] = {"table_dir": str(table_dir), "sources": {}}
    for source_name, df in bundle.sources.items():
        attrs = sorted({a for (s, a) in attr_index if s == source_name and a != "id"})
        ids = ["" if v is None else str(v) for v in df["id"].tolist()]
        columns: dict[str, list[str]] = {}
        src_info: dict[str, Any] = {"rows": len(df), "attributes": {}}
        for attr in attrs:
            cols = [c for c in attr_index[(source_name, attr)] if c in df.columns]
            kind = kind_map.get(attr, "long_string")
            values = [""] * len(df)
            n_multi_conflict = 0
            for ci, col in enumerate(cols):
                raw_col = df[col].tolist()
                for i, raw in enumerate(raw_col):
                    try:
                        out = member.normalize(raw, attribute=attr, kind=kind, domain=domain)
                    except Exception:  # the runner records such cells as None
                        out = None
                    s = "" if out is None else str(out)
                    if ci == 0:
                        values[i] = s
                    elif s:
                        if not values[i]:
                            values[i] = s
                        elif values[i] != s:
                            n_multi_conflict += 1
            columns[attr] = values
            src_info["attributes"][attr] = {
                "source_columns": cols,
                "kind": kind,
                "in_stage_scope": attr in eligible,
                "rule": (selection_map.get(attr, "_passthrough")
                         if member_name == "rule_per_attribute_optimal" else "passthrough"),
                "non_empty": sum(1 for v in values if v),
                "multi_column_conflicts": n_multi_conflict,
            }
        with open(table_dir / f"{source_name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", *attrs])
            for i, rid in enumerate(ids):
                w.writerow([rid, *(columns[a][i] for a in attrs)])
        info["sources"][source_name] = src_info
    return info


def _write_details(path: Path, details: list[dict]) -> None:
    if not details:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(details[0]))
        w.writeheader()
        w.writerows(details)


def _files_modified_since(root: Path, t0: float) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "pydi-dev", ".venv")]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                if p.stat().st_mtime >= t0:
                    out.append(str(p.relative_to(root)))
            except OSError:
                continue
    return sorted(out)


if __name__ == "__main__":
    sys.exit(main())
