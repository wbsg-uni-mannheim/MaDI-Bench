#!/usr/bin/env python
"""Summarise one P2 timing run (retime.sh) into <run>/best_pipeline_time.json.

Inputs: <run>/phases.tsv (wall clock of every timed phase, written by retime.sh with `date +%s.%N` around
each command), <run>/job.env and <run>/timing_pipeline.json (written by timed_bob.py).

Table 9 prints, per task, training + pipeline_wall_no_panel, rounded to seconds:
  training                 the training time of the checkpoints the reported P2 run used (Ditto, SC-Block),
                           passed with --training (the values of the paper are in
                           results/paper_tables/table9/p2_runtime.csv, column training_s_per_model);
  pipeline_wall            wall clock of the whole run_best_of_breed.py process (bundle load, the six stages
                           with their own scoring, e2e panel, artifact writing);
  pipeline_wall_no_panel   pipeline_wall minus the end-to-end panel the pipeline computes after the run.

    python reproduction/tables/table9/timing_summary.py --domain games --run <run> --training ditto_plm=54.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", required=True)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--training", action="append", default=[], metavar="MODEL=SECONDS",
                    help="training seconds of a checkpoint the run loads (repeatable)")
    a = ap.parse_args(argv)
    ph: dict = {}
    for line in (a.run / "phases.tsv").read_text().splitlines():      # "<name> <start> <end> <exit>"
        name, s, e, rc = line.split()
        ph[name] = {"seconds": round(float(e) - float(s), 2), "exit": int(rc)}
    job = dict(line.split("=", 1) for line in (a.run / "job.env").read_text().splitlines() if "=" in line)
    tj = json.loads((a.run / "timing_pipeline.json").read_text())
    events: dict[str, float] = {}
    for e in tj["events"]:
        events[e["name"]] = round(events.get(e["name"], 0.0) + e["seconds"], 3)
    wall = ph["pipeline"]["seconds"]
    panel = events.get("e2e_panel", 0.0)
    training = {k: float(v) for k, v in (t.split("=", 1) for t in a.training)}
    out = {
        "domain": a.domain, "job": job.get("JOB"), "node": job.get("NODE"), "gpu": tj.get("device", {}).get("gpu"),
        "training_s": training, "training_total_s": round(sum(training.values()), 1),
        "pipeline_wall_s": round(wall, 2), "pipeline_python_wall_s": tj["python_wall_s"],
        "pipeline_imports_s": tj["imports_s"], "pipeline_events_s": events,
        "pipeline_wall_no_panel_s": round(wall - panel, 2),
        "em_per_pair": tj["pairs"],
        "runtime_s": round(sum(training.values()) + wall - panel, 2),
    }
    (a.run / "best_pipeline_time.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.domain}: training {out['training_total_s']} s + pipeline without e2e panel "
          f"{out['pipeline_wall_no_panel_s']} s = {out['runtime_s']} s (pipeline with panel {out['pipeline_wall_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
