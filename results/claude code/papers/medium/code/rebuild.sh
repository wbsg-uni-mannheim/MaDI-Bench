#!/usr/bin/env bash
# Rebuild every submission data file from task/input only (no network, no training). ~35 min (blocking dominates).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/state submission/blocking
python work/s1_schema.py        # schema mapping -> submission/sm_mapping.csv
python work/s2_normalize.py     # normalization
python work/s3_blocking.py      # blocking candidates
python work/s4_features.py      # pair evidence
python work/s4_match.py         # scoring (work/score.py), threshold 3.0, 1:1 per source pair
python work/s5_cluster.py       # constrained clustering
python work/s6_fusion.py        # attribute fusion
python work/s7_export.py        # submission csvs
python work/s8_diagnostics.py rebuild
