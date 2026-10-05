#!/usr/bin/env bash
# Regenerates all submission data files from task/input + work scripts (no network, no training).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p state ../submission/blocking
python s1_schema.py
python s2_normalize.py
python s3_blocking.py
python s4_features.py
python s4_score.py
python s5_cluster.py
python s6_fusion.py
python s7_diagnostics.py
