"""Replay the P3 Products base run with the June 2026 timing protocol
(DI-Bench/scripts/time_pipeline_runs.py): cached labeled sets and active
learning, official validation fusion only, fusion case all, wall-clock time
of the run_pipeline.py subprocess."""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
S = os.path.dirname(os.path.abspath(__file__))
env = os.environ.copy()
env.update({
 "PYTHONPATH": f"{S}/code",
 "PYDI_SCHEMA_LLM_MODEL": "gpt-5.5", "PYDI_LABELING_LLM_MODEL": "gpt-5.2", "PYDI_FUSION_LLM_MODEL": "gpt-5.5",
 "PYDI_LLM_LABELING_WORKERS": "10", "PYDI_LLM_REQUEST_TIMEOUT": "120", "PYDI_LLM_MAX_RETRIES": "2",
 "PYDI_LLM_LABELING_HEARTBEAT_SECONDS": "60", "PYDI_OPENAI_EMBEDDING_TIMEOUT": "120",
 "PYDI_TAXONOMY_LLM_WORKERS": "10", "PYDI_FAISS_THREADS": "1", "PYDI_ACTIVE_ENSEMBLE_WORKERS": "2",
 "PYDI_ACTIVE_TARGETED_POOL_MULTIPLIER": "2",
 "PYDI_ACTIVE_LEARNING_CACHE_ONLY": "1", "PYDI_LABELED_SET_CACHE_ONLY": "1",
 "PYDI_PIPELINE_METRICS_PATH": f"{S}/timing/pipeline_metrics.json",
 "PYDI_PIPELINE_METRICS_CSV_PATH": f"{S}/timing/pipeline_metrics.csv",
 "PYDI_PIPELINE_LOG_PATH": f"{S}/timing/pipeline.log", "PYDI_PIPELINE_LOG_MODE": "w",
})
os.makedirs(f"{S}/timing", exist_ok=True)
cmd = [f"{S}/.venv/bin/python", "scripts/run_pipeline.py", "--manifest", f"{S}/replay_manifest.json",
       "--training-set-construction", "active", "--fusion-case", "all", "--official-validation-fusion-only"]
start = datetime.now(timezone.utc); t0 = time.perf_counter()
with open(f"{S}/timing/console.log", "w") as con:
    rc = subprocess.run(cmd, cwd=f"{S}/code", env=env, stdout=con, stderr=subprocess.STDOUT).returncode
wall = time.perf_counter() - t0
out = {"case": "products", "protocol": "June 2026 cached replay (time_pipeline_runs.py)", "command": cmd,
       "return_code": rc, "started_at_utc": start.isoformat(timespec="seconds"),
       "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "wall_seconds": wall}
json.dump(out, open(f"{S}/timing/timing.json", "w"), indent=2)
print(json.dumps(out, indent=2))
