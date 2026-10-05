#!/usr/bin/env bash
# Regenerates every file in submission/ from task/input via the six stage scripts (no network, no training).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/state submission/blocking
python work/s1_schema.py
python work/s2_normalize.py
python work/s3_blocking.py
python work/s4_features.py
python work/s5_match_cluster.py
python work/s6_fusion.py
cp work/report_src.md submission/report.md
