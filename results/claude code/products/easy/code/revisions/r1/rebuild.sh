#!/usr/bin/env bash
# Regenerates every data file in submission/ from task/input and the scripts in work/. Deterministic; no network, no training.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/state submission
for s in s1_schema s2_normalize s3_blocking s4_matching s5_cluster s6_fusion s7_export s8_diagnostics; do
  python work/$s.py
done
