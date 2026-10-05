#!/usr/bin/env bash
# Regenerates every data file in submission/ from task/input using the saved scripts and configs
# (work/match_config.json, work/overrides.json). Deterministic; no network, no training.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p state ../submission/blocking
python s1_schema.py
python s2_normalize.py
python s2b_taxonomy.py
python s3_blocking.py
python s4_match.py
python s5_cluster.py
python s6_fusion.py
cp report.md ../submission/report.md
python diagnostics.py
