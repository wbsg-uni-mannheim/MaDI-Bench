#!/usr/bin/env bash
# Rebuild every submission data file from the source tables + retained manual review artifacts.
# No network, no model training. Run from the workspace root (or anywhere; it cd's there).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/state submission/blocking
python work/s1_schema.py        # schema mapping -> submission/sm_mapping.csv, work/state/s1_translated.csv
python work/s2_normalize.py     # identity features -> work/state/s2_features.csv
python work/s3_blocking.py      # automatic candidates -> work/state/s3_candidates_all.csv
python work/s4_matching.py      # automatic pair scores (proposal/diagnostic) -> work/state/s4_scored_pairs.csv
python work/s5_cluster.py       # clusters incl. work/manual/decisions.txt -> work/state/s5_membership.csv
python work/s6_fusion.py        # fused table -> submission/fused.csv (+ work/state/s6_provenance.csv)
python work/s7_export.py        # correspondences, membership, blocking candidates
python work/diagnostics.py rebuild >/dev/null
echo "rebuild complete"
