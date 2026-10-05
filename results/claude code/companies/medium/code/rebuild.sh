#!/usr/bin/env bash
# Regenerates every data file in submission/ from source data + cached embeddings (work/state/emb_cache.json).
# No network calls, no training. s0_embed_cache.py is NOT run (its output is cached).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p submission/blocking work/state
for s in s1_schema s2_normalize s3_blocking s4_matching s5_cluster s6_fusion export; do
  python work/$s.py > work/state/$s.log
done
python work/diagnostics.py >> work/state/diagnostics_run.log
echo "rebuild done"
