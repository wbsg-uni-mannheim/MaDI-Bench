#!/bin/bash
# Headless run of one human-engineered pipeline (P1) notebook on a scratch copy of its task folder.
#
# usage: reproduction/p1/run_p1.sh <domain> <run_root>
#   domain    companies | games | music | papers | products
#   run_root  a folder OUTSIDE 'use cases/' (created if missing); the run goes to <run_root>/<domain>/
#
# What it does (the procedure behind the P1 outputs in 'use cases/<domain>/base/output/'):
#   1. copies 'use cases/<domain>/base/' (inputs and notebook; no output/, no __pycache__) to
#      <run_root>/<domain>/base/ -- products also gets its EM gold from
#      output/entity_matching_final_ground_truth/, which its notebook reads;
#   2. patches the notebook copy with p1_notebooks_patch.py (the shipped notebooks are already
#      patched: this reports "already patched") and converts it to a script with p1_nb2script.py;
#   3. runs the script with the working directory set to the copy; the notebook writes
#      output/data_fusion/{fused.csv,fusion_scores_v2.json} and output/normalization/<source>.csv
#      there, and the script dumps the fused frame to <run_root>/<domain>/dump/;
#   4. deletes the copy's input/ and the copied EM gold again (gold lives only in the task folders).
#
# Environment:
#   PYTHON          the interpreter of an environment with this repository installed as described in
#                   the top-level README (pinned PyDI; default: python3)
#   OPENAI_API_KEY  needed by the companies, games and music notebooks (LLM schema matching);
#                   this script never reads a .env file -- export the key yourself
#   P1_THREADS      threads for OpenMP/MKL (default 6); the reference runs used 8 CPUs and 32 GB
#
# The P1 outputs of the paper came from this procedure (2026-09-24; products 2026-09-30); the
# music normalized tables were exported from the tables of the 2026-09-24 run with the export
# cell that p1_notebooks_patch.py generates.
#
# Paths are relative to this
# repository, no .env file is sourced, and the interpreter is a parameter.
set -o pipefail
D=$1; ROOT=$2
[ -n "$D" ] && [ -n "$ROOT" ] || { echo "usage: reproduction/p1/run_p1.sh <domain> <run_root>" >&2; exit 2; }
case "$D" in companies|games|music|papers|products) ;; *) echo "unknown domain: $D" >&2; exit 2;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$REPO/use cases"
PY="${PYTHON:-python3}"
# a relative interpreter path (e.g. PYTHON=.venv/bin/python) must survive the cd into the run folder below;
# make it absolute without resolving symlinks (a venv's python is a symlink to the base interpreter)
case "$PY" in */*) PYDIR="$(cd "$(dirname "$PY")" && pwd)" || exit 1; PY="$PYDIR/$(basename "$PY")" ;; esac
# resolve both paths (symlinks included) BEFORE creating anything, so that a run root inside the
# task folders is refused without a folder being made there
abspath() { "$PY" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }
ROOT="$(abspath "$ROOT")" || exit 1
SRC_REAL="$(abspath "$SRC")" || exit 1
case "$ROOT/" in "$SRC_REAL/"*) echo "run_root must lie outside '$SRC'" >&2; exit 2;; esac
mkdir -p "$ROOT" || exit 1
if [ -e "$ROOT/$D" ]; then echo "$ROOT/$D exists; use a fresh run_root" >&2; exit 2; fi
case "$D" in companies|games|music)
  [ -n "$OPENAI_API_KEY" ] || echo "warning: OPENAI_API_KEY is not set; the $D notebook's LLM schema matching needs it" >&2;;
esac

RUN="$ROOT/$D"
cleanup() { rm -rf "$RUN/base/input" "$RUN/base/output/entity_matching_final_ground_truth"; }
trap cleanup EXIT

mkdir -p "$RUN/base"
# 1. scratch copy of the task folder (inputs + notebook)
if command -v rsync >/dev/null; then
  rsync -a --exclude 'output/' --exclude 'data_backups/' --exclude '__pycache__' "$SRC/$D/base/" "$RUN/base/" || exit 1
else   # GNU tar and bsdtar match an exclude pattern without '/' against every path component
  (cd "$SRC/$D/base" && tar --exclude=output --exclude=data_backups --exclude=__pycache__ -cf - .) \
    | (cd "$RUN/base" && tar -xf -) || exit 1
fi
if [ -d "$SRC/$D/base/output/entity_matching_final_ground_truth" ]; then
  mkdir -p "$RUN/base/output"
  cp -R "$SRC/$D/base/output/entity_matching_final_ground_truth" "$RUN/base/output/" || exit 1
fi
NB=$(ls "$RUN/base"/*workflow*.ipynb | head -1)
[ -n "$NB" ] || { echo "no *workflow*.ipynb in $SRC/$D/base" >&2; exit 1; }

# 2. patch (idempotent) and convert to a headless script
"$PY" "$HERE/p1_notebooks_patch.py" --notebook "$NB" --domain "$D" || exit 1
"$PY" "$HERE/p1_nb2script.py" "$D" "$NB" "$RUN/run_$D.py" "$RUN/dump" || exit 1

# 3. run with the copy as working directory
export MADI_BENCH_ROOT="${MADI_BENCH_ROOT:-$REPO}"
T="${P1_THREADS:-6}"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
cd "$RUN/base" || exit 1
echo "START $(date -u +%FT%TZ) $D on $(hostname)" > "$RUN/run.log"
"$PY" -u "$RUN/run_$D.py" >> "$RUN/run.log" 2>&1
rc=$?
echo "END $(date -u +%FT%TZ) rc=$rc" >> "$RUN/run.log"
ls "$RUN/base/output/data_fusion" "$RUN/base/output/normalization" >> "$RUN/run.log" 2>&1
# 4. the trap removes the copied gold
exit $rc
