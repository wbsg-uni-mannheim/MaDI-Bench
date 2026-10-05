#!/usr/bin/env bash
# Rebuild all submission data files from task inputs + cached embeddings (work/state/emb_cache.pkl). No network, no training.
set -euo pipefail
cd "$(dirname "$0")"
export NO_NETWORK=1
mkdir -p state ../submission/blocking
python s1_schema.py
python s2_normalize.py
python s3_blocking.py
python s4_match.py
python s4_match.py decide
python s5_cluster.py
python s6_fusion.py
echo "rebuild done"
