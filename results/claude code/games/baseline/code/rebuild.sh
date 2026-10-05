#!/usr/bin/env bash
# Rebuild all submission data files deterministically from task inputs (no network, no training).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p submission/blocking work/state
python work/s1_schema.py
python work/s2_normalize.py
python work/s3_blocking.py
python work/s4_matching.py
python work/s5_cluster.py
python work/s6_fusion.py
python work/s7_export_diag.py > /dev/null
echo "rebuild done"
