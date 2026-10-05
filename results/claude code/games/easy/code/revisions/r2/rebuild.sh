#!/usr/bin/env bash
# Regenerates every data file in submission/ from task/input (deterministic; no network, no training).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p state ../submission/blocking
python s1_schema.py
python s2_normalize.py
python s3_blocking.py
python s4_matching.py > state/s4_log.txt
python s5_cluster.py
python s6_fusion.py
python s3b_export_candidates.py
python diagnostics.py
