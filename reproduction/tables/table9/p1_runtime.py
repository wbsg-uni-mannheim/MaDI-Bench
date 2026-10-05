#!/usr/bin/env python
"""Table 9, P1 runtime: the notebook's own timer minus the two cells that reproduction/p1/p1_notebooks_patch.py
writes into every P1 notebook: the normalization export ('# --- MaDI-Bench normalization output') and the
fusion scoring ('# --- MaDI-Bench fusion evaluation'). The Products notebook also carries a
normalization stage ('# --- MaDI-Bench normalization stage'); it is part of the pipeline and is not
subtracted.

Input: P1 run folders as written by reproduction/p1/run_p1.sh, <run_root>/<domain>/ with run.log (holds
'Total execution time: <s> seconds') and dump/cells.log (the start of every cell in seconds since the start
of the script, one decimal). The duration of a cell is the next cell's start minus its own, so each
corrected runtime is exact to about 0.1-0.2 s. The paper's values narrow the added cells' time further with the
file timestamps of the run folders (results/paper_tables/table9/p1_runtime.csv: column run names the runs,
column corrected_s_file_stamps holds the values).

    python reproduction/tables/table9/p1_runtime.py --p1-run <run_root> [--domains games,music] --out p1.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

DOMAINS = ("games", "companies", "music", "products", "papers")
EXPORT = "# --- MaDI-Bench normalization output"
EVAL = "# --- MaDI-Bench fusion evaluation"
END_CELL = ("end_time = time.time()", "final_time = time.time()")


def half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def cells(run: Path) -> list[tuple[float, str, str]]:
    marks = []
    for line in (run / "dump" / "cells.log").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([0-9.]+)s\s+cell (\S+) (.*)$", line)
        if m:
            marks.append((float(m.group(1)), m.group(2), m.group(3)))
    return marks


def runtime(run: Path) -> dict:
    log = (run / "run.log").read_text(encoding="utf-8", errors="replace")
    timer = float(re.search(r"Total execution time: ([0-9.]+) seconds", log).group(1))
    mk = cells(run)
    ie = [i for i, c in enumerate(mk) if c[2].startswith(EXPORT)]
    iv = [i for i, c in enumerate(mk) if c[2].startswith(EVAL)]
    if len(ie) != 1 or len(iv) != 1:
        raise ValueError(f"{run}: expected one export and one evaluation cell, found {len(ie)} and {len(iv)}")
    ie, iv = ie[0], iv[0]
    if not mk[iv + 1][2].startswith(END_CELL):
        raise ValueError(f"{run}: the evaluation cell is not followed by the timer cell")
    added = (mk[ie + 1][0] - mk[ie][0]) + (mk[iv + 1][0] - mk[iv][0])
    return {"notebook_timer_s": timer, "added_cells_s": round(added, 1), "corrected_s": round(timer - added, 2),
            "runtime_s": half_up(timer - added)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p1-run", type=Path, required=True, help="run root with <domain>/ folders (run_p1.sh)")
    ap.add_argument("--domain-run", action="append", default=[], metavar="DOMAIN=ROOT",
                    help="a different run root for one domain (repeatable)")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    roots = {d: a.p1_run for d in DOMAINS}
    roots.update({k: Path(v) for k, v in (s.split("=", 1) for s in a.domain_run)})
    rows = []
    for d in [x for x in a.domains.split(",") if x]:
        r = runtime(roots[d] / d)
        rows.append([d, r["notebook_timer_s"], r["added_cells_s"], r["corrected_s"], r["runtime_s"]])
        print(f"{d}: timer {r['notebook_timer_s']} s - added cells {r['added_cells_s']} s = {r['corrected_s']} s "
              f"({r['runtime_s']})", flush=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["domain", "notebook_timer_s", "added_cells_s", "corrected_s_cells_log", "runtime_s"])
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
