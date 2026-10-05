#!/usr/bin/env bash
# Rebuild every submission data file from task/input (deterministic; no network, no training).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p state ../submission/blocking
python s1_schema.py > /dev/null
python s2_normalize.py > /dev/null
python s3_blocking.py > /dev/null
python s4_match.py
python s5_cluster.py
python s6_fusion.py
python s7_diagnostics.py rebuild > /dev/null
