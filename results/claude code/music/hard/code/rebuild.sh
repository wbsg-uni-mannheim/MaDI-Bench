#!/usr/bin/env bash
# Rebuild every submission data file from the raw inputs (deterministic; no network, no training).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p state ../submission/blocking
python s1_schema.py > /dev/null
python s2_normalize.py > /dev/null
python s3_blocking.py > /dev/null
python s4_features.py > /dev/null
python s5_match_cluster.py > /dev/null
python s6_fusion.py
