"""Turn a human-baseline (P1) notebook into a headless script that also dumps the fused frame.
usage: python reproduction/p1/p1_nb2script.py <domain> <notebook.ipynb> <out.py> <dump_dir>

IPython-only lines (display imports, cell magics) are dropped; display() is a no-op. The script
logs the start time of every cell to <dump_dir>/cells.log and, after the last cell, writes the
fused frame to <dump_dir>/fused.{csv,jsonl} and a summary to <dump_dir>/run_info.json.
run_p1.sh calls it."""
import json, sys
domain, nb_path, out_path, dump_dir = sys.argv[1:5]
FUSED_VAR = {'papers': 'fused_v5'}.get(domain, 'fused')
cells = json.load(open(nb_path))['cells']
parts = [f'''# auto-generated from {nb_path}
import os, sys, json, time as _time
from pathlib import Path
_T0 = _time.time()
def display(*a, **k):
    pass
DUMP_DIR = Path({dump_dir!r}); DUMP_DIR.mkdir(parents=True, exist_ok=True)
_LOG = open(DUMP_DIR / "cells.log", "a")
def _mark(i, name):
    _LOG.write(f"{{_time.time()-_T0:9.1f}}s  cell {{i}} {{name}}\\n"); _LOG.flush()
''']
for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    # headless: no IPython (the notebooks import display explicitly), no cell magics
    src = '\n'.join(l for l in src.splitlines()
                    if not l.lstrip().startswith(('from IPython', 'import IPython', '%', '!')))
    if not src.strip():
        continue
    first = next((l for l in src.splitlines() if l.strip()), '')[:60]
    parts.append(f'\n# ===== cell {i}\n_mark({i}, {first!r})\n{src}\n')
parts.append(f'''
# ===== epilogue: dump the fused frame (dump/fused.csv, fused.jsonl, run_info.json)
_mark("end", "epilogue")
import pandas as pd
_f = {FUSED_VAR}.copy()
if "_id" not in _f.columns:
    _f["_id"] = _f["_fusion_sources"].apply(lambda s: "+".join(sorted(str(x) for x in s)))
_f["_id"] = _f["_id"].astype(str)
_f.to_json(DUMP_DIR / "fused.jsonl", orient="records", lines=True, force_ascii=False)
_f.to_csv(DUMP_DIR / "fused.csv", index=False)
_info = {{"domain": {domain!r}, "rows": int(len(_f)), "columns": list(map(str, _f.columns)),
         "elapsed_s": round(_time.time() - _T0, 1)}}
for _name in ("evaluation_results", "baseline_metrics"):
    if _name in globals():
        try:
            _info[_name] = {{k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in globals()[_name].items()}}
        except Exception as e:
            _info[_name] = str(e)
json.dump(_info, open(DUMP_DIR / "run_info.json", "w"), indent=1)
print("DUMPED", _info["rows"], "rows in", _info["elapsed_s"], "s")
''')
open(out_path, 'w').write('\n'.join(parts))
print('wrote', out_path)
