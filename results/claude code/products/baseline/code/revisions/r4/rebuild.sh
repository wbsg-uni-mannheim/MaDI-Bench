#!/usr/bin/env bash
# Regenerates every submission/ data file from the raw sources (deterministic, no network, no training).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p state ../submission/blocking
python s1_schema.py
python s2_normalize.py
python s3_block.py
python s4_match.py
python s6_fusion.py
python s7_diag.py rebuild
