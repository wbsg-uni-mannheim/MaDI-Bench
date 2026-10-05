#!/usr/bin/env bash
# Regenerates every data file in submission/ from task/input using the stage scripts. No network, no training.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p submission/blocking work/state
python work/s1_schema.py
python work/s2_normalize.py
python work/s3_blocking.py
python work/s4a_features.py
python work/s4b_match.py
python work/s5_cluster.py
python work/s6_fusion.py
python work/s7_diagnostics.py rebuild
