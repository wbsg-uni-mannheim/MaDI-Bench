#!/usr/bin/env bash
# Regenerates every file in submission/ from the raw inputs (deterministic; no network, no training).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/state submission/blocking
python work/s1_schema.py
python work/s2_normalize.py
python work/s3_blocking.py
python work/s4_features.py
python work/s5_match.py
python work/s6_cluster.py 3.5
python work/s7_fusion.py
python work/s8_diagnostics.py rebuild
cp work/report.md submission/report.md
