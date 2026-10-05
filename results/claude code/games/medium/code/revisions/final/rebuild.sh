#!/usr/bin/env bash
# Regenerates every data file in submission/ from task/input and the scripts in work/ (no network, no training).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p state ../submission/blocking
python s1_schema.py        # schema mapping -> submission/sm_mapping.csv, state/s1_unified.pkl
python s2_normalize.py     # normalization  -> state/s2_norm.pkl, state/platform_surface.json
python s3_blocking.py      # blocking       -> state/s3_candidates.pkl, submission/blocking/candidates.csv (~8 min)
python s4_matching.py      # pair scoring   -> state/s4_pairs.pkl, state/s4_nodes.pkl
python s5_cluster.py       # clustering     -> state/s5_clusters.pkl
python s6_fusion.py        # fusion         -> submission/fused.csv, membership.csv, correspondences.csv
python diagnostics.py rebuild > /dev/null
echo "rebuild done"
