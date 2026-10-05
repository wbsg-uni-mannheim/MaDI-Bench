#!/usr/bin/env python
"""Patch the five human-pipeline (P1) notebooks:

  * the fusion evaluation cells (PyDI's DataFusionEvaluator with the notebooks'
    own comparator registrations; music loaded a file that does not exist,
    products used an absolute 0.15 tolerance, games' genres comparator was
    always true, papers registered none) are replaced by ONE cell that scores
    the fused table with the public package, madi_bench.evaluation.score_fusion,
    on the validation and the test split (strict rules, v2 gold, all-gold
    accuracy with coverage), writes output/data_fusion/fused.csv and
    fusion_scores_v2.json;
  * a cell is added after the normalization stage that writes the normalized
    per-source tables (id + target-schema columns) to output/normalization/
    <source>.csv, the input of madi_bench.evaluation.score_normalization. The
    tables are the frames the notebook passes on after its normalization steps
    (for music the copies cells 21-22 post-process, which feed matching and
    fusion, not the translator output).

Nothing else in the notebooks changes. The package is imported from the
installed distribution, from $MADI_BENCH_ROOT, or from the clone root three
levels above the task folder (use cases/<domain>/base).

The notebooks shipped in 'use cases/<domain>/base/' carry both cells already;
on them the script reports "already patched" (or "refreshed" if the cell
templates below changed). Run from the repository root:

    python reproduction/p1/p1_notebooks_patch.py --root <dir>                    # patch copies in <dir>/<domain>/
    python reproduction/p1/p1_notebooks_patch.py --notebook <copy.ipynb> --domain games   # one notebook in place
    python reproduction/p1/p1_notebooks_patch.py                                 # the task folders' notebooks

"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # reproduction/p1/ -> repository root
USECASES = REPO / "use cases"

ALIASES = {"companies": (("keypeople_name", "keypeople"), ("founders", "keypeople"))}

NOTEBOOKS = {
    "companies": {"file": "companies_workflow.ipynb", "task_dir": "NOTEBOOK_DIR", "fused": "fused",
                  "eval_cells": [53, 54], "export_after": 22,
                  "tables": '{"forbes": forbes_normalized, "dbpedia": dbpedia_normalized, "fullcontact": fullcontact_normalized}'},
    "games": {"file": "games_workflow.ipynb", "task_dir": "NOTEBOOK_DIR", "fused": "fused",
              "eval_cells": [62, 63], "export_after": 27,
              "tables": '{"dbpedia": dbpedia_normalized, "metacritic": metacritic_normalized, "sales": sales_normalized}'},
    # music: cells 21-22 (still under "Step 5: Translate and Normalize") copy the target columns
    # of the *_normalized frames into mbrainz / discogs / lastfm, parse the track lists and
    # complete the ISO release dates there; those copies feed matching and fusion (cells 35
    # and 43 of the patched notebook), so they are what the export writes (the
    # *_normalized frames miss both steps).
    "music": {"file": "music_workflow.ipynb", "task_dir": "NOTEBOOK_DIR", "fused": "fused",
              "eval_cells": [46], "export_after": 22,
              "tables": '{"musicbrainz": mbrainz, "discogs": discogs, "lastfm": lastfm}'},
    "papers": {"file": "papers_workflow_minimal.ipynb", "task_dir": "BASE_DIR", "fused": "fused_v5",
               "eval_cells": [23], "export_after": 22,
               "tables": '{_n: _df.rename(columns={f"{_n}_id": "id"}) for _n, _df in normalized_loaded_v5.items()}'},
    "products": {"file": "products_workflow_minimal.ipynb", "task_dir": "NOTEBOOK_DIR", "fused": "fused",
                 "eval_cells": [45], "export_after": 5,
                 "tables": '{"products_1": products_1_cleaned, "products_2": products_2_cleaned, '
                           '"products_3": products_3_cleaned, "products_4": products_4_cleaned}'},
}
# the first line of the cell each replacement expects (a guard against index drift)
EXPECTED_FIRST_LINE = {
    "companies": ("from PyDI.fusion import DataFusionEvaluator", "from PyDI.fusion import DataFusionEvaluator"),
    "games": ("from PyDI.io import load_xml", "from PyDI.io import load_xml"),
    "music": ("from PyDI.fusion import DataFusionEvaluator",),
    "papers": ("",),          # the papers cell starts with a blank line
    "products": ("# 1. Identity (Must be exact)",),
}
MARK_EVAL = "# --- MaDI-Bench fusion evaluation"
MARK_EXPORT = "# --- MaDI-Bench normalization output"

EVAL_CELL = '''{mark}: the public evaluation package scores the
# fused table on the validation and the test split -- strict comparators on the v2 gold set,
# all-gold accuracy (a gold record the pipeline did not produce counts as wrong) with its coverage.
# This is the scorer the benchmark's reference numbers were produced with; the notebook's own
# comparator registrations above are not used for scoring.
import json as _json, os as _os, sys as _sys
from pathlib import Path as _Path
try:
    from madi_bench.evaluation import score_fusion as _score_fusion
except ImportError:
    _sys.path.insert(0, _os.environ.get("MADI_BENCH_ROOT") or str(_Path({task_dir}).resolve().parents[2]))
    from madi_bench.evaluation import score_fusion as _score_fusion
_fused_out = {fused}.copy()
# the notebook's own column names for target-schema attributes (the same aliases the benchmark's
# rescoring applies, reproduction/scoring/rescore.py): scoring is by target-schema name
for _src, _dst in {aliases}:
    if _src in _fused_out.columns and _dst not in _fused_out.columns:
        _fused_out = _fused_out.rename(columns={{_src: _dst}})
_fusion_dir = OUTPUT_DIR / "data_fusion"
_fusion_dir.mkdir(parents=True, exist_ok=True)
_fused_out.to_csv(_fusion_dir / "fused.csv", index=False)
_scores = {{}}
for _split in ("validation", "test"):
    _r = _score_fusion({task_dir}, _fused_out, split=_split)
    _scores[_split] = _r
    print(f"{{_split}}: all-gold accuracy {{_r['overall_accuracy_all_gold']:.4f}} "
          f"(evaluated {{_r['overall_accuracy_evaluated']:.4f}} on {{_r['n_gold_evaluated']}}/{{_r['n_gold']}} gold records, "
          f"attribute cell coverage {{_r['attribute_cell_coverage']:.2f}})")
    for _a, _v in sorted(_r["per_attribute_accuracy_all_gold"].items()):
        print(f"    {{_a:28s}} {{_v:.4f}}")
(_fusion_dir / "fusion_scores_v2.json").write_text(_json.dumps(_scores, indent=1, default=str))
'''

EXPORT_CELL = '''{mark}: the normalized source tables, one CSV per source
# (record id + target-schema columns), the input of madi_bench.evaluation.score_normalization,
# which scores them against input/normalization/<split>.csv.
import json as _json
from pathlib import Path as _Path
_norm_dir = OUTPUT_DIR / "normalization"
_norm_dir.mkdir(parents=True, exist_ok=True)
_target = list(_json.loads((_Path({task_dir}) / "input" / "schemamatching" / "target_schema.json").read_text())["properties"])
for _name, _df in {tables}.items():
    _id_col = "id" if "id" in _df.columns else next((c for c in _df.columns if c.endswith("_id")), _df.columns[0])
    _cols = [_id_col] + [c for c in _df.columns if c in _target and c != "id" and c != _id_col]
    _df[_cols].rename(columns={{_id_col: "id"}}).to_csv(_norm_dir / f"{{_name}}.csv", index=False)
    print(f"normalized table {{_name}}: {{len(_df)}} rows, {{len(_cols)}} columns -> {{_norm_dir / (_name + '.csv')}}")
'''


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def patch_notebook(path: Path, domain: str, spec: dict) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    marked = [i for i, c in enumerate(cells) if MARK_EVAL in "".join(c["source"]) or MARK_EXPORT in "".join(c["source"])]
    if marked:
        # already patched: refresh the two generated cells in place, touch nothing else
        eval_src = EVAL_CELL.format(mark=MARK_EVAL, task_dir=spec["task_dir"], fused=spec["fused"],
                                    aliases=repr(ALIASES.get(domain, ())))
        export_src = EXPORT_CELL.format(mark=MARK_EXPORT, task_dir=spec["task_dir"], tables=spec["tables"])
        changed = 0
        for i in marked:
            new = eval_src if MARK_EVAL in "".join(cells[i]["source"]) else export_src
            if "".join(cells[i]["source"]) != new:
                cells[i] = code_cell(new); changed += 1
        if changed:
            path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"status": "refreshed" if changed else "already patched", "cells_refreshed": changed}
    for idx, expected in zip(spec["eval_cells"], EXPECTED_FIRST_LINE[domain]):
        first = "".join(cells[idx]["source"]).split("\n", 1)[0]
        if first != expected:
            raise SystemExit(f"{domain}: cell {idx} starts with {first!r}, expected {expected!r}; refusing")
    eval_src = EVAL_CELL.format(mark=MARK_EVAL, task_dir=spec["task_dir"], fused=spec["fused"],
                                aliases=repr(ALIASES.get(domain, ())))
    first, *rest = spec["eval_cells"]
    cells[first] = code_cell(eval_src)
    for idx in sorted(rest, reverse=True):
        del cells[idx]
    export_src = EXPORT_CELL.format(mark=MARK_EXPORT, task_dir=spec["task_dir"], tables=spec["tables"])
    cells.insert(spec["export_after"] + 1, code_cell(export_src))
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"status": "patched", "eval_cell": first, "export_cell": spec["export_after"] + 1,
            "removed": rest, "cells": len(cells)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None,
                    help="rehearsal: copy the notebooks to <ROOT>/<domain>/ and patch the copies")
    ap.add_argument("--domains", default=",".join(NOTEBOOKS))
    ap.add_argument("--src", type=Path, default=USECASES,
                    help="task-folder root the notebooks are taken from (default: the repository's 'use cases')")
    ap.add_argument("--notebook", type=Path, default=None,
                    help="patch this one notebook file in place (needs --domain)")
    ap.add_argument("--domain", choices=sorted(NOTEBOOKS), default=None,
                    help="the domain of --notebook")
    args = ap.parse_args(argv)
    if args.notebook is not None:
        if args.domain is None:
            ap.error("--notebook needs --domain")
        print(f"{args.domain:10s} {patch_notebook(args.notebook, args.domain, NOTEBOOKS[args.domain])}")
        return 0
    for domain in [d for d in args.domains.split(",") if d]:
        spec = NOTEBOOKS[domain]
        source = args.src / domain / "base" / spec["file"]
        if args.root:
            target = args.root / domain / spec["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            target = source
        print(f"{domain:10s} {patch_notebook(target, domain, spec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
