#!/usr/bin/env bash
# Regenerates every data file in submission/ from the raw inputs. Deterministic; no network, no training.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/state submission/blocking
python work/s1_schema.py
python work/s2_normalize.py
python work/s3_blocking.py
python work/s4_features.py
python work/s4_match.py
python work/s5_cluster.py
python work/s6_fusion.py
python work/diagnose.py || true
