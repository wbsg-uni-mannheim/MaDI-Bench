#!/usr/bin/env bash
# Regenerates submission/ from source data + retained local artifacts (cached embeddings, manual_decisions.csv).
# No network, no training. Embeddings were computed once by work/embed_cache.py and are read from work/state/.
set -euo pipefail
cd "$(dirname "$0")"
python s1_schema.py
python s2_normalize.py
python s3_blocking.py
python s4_features.py
python s4_bands.py
python s4_match.py
python s5_cluster.py
python s6_fusion.py
python diagnostics.py
