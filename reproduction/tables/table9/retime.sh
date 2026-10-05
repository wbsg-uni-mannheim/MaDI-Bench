#!/bin/bash
# Table 9, P2 runtime: time the P2 (best-of-breed) pipeline whose outputs are in
# 'results/best of breeds/<domain>/baseline/', on one base task.
#
# usage: DOMAIN=games RUN=<fresh folder> DITTO_CKPT=<dir> SCB_CKPT=<dir> \
#            bash reproduction/tables/table9/retime.sh          (from the repository root)
#
#   DOMAIN      companies | games | music | papers | products
#   RUN         a fresh output folder (must not exist); PyDI's blockers write output/ into it
#   DITTO_CKPT  the Ditto checkpoint of the reported run (pipelines/<domain>/checkpoints/em_matching/ditto/...)
#   SCB_CKPT    the SC-Block checkpoint of the reported run (pipelines/<domain>/checkpoints/em_blocking/sc_block/...)
#   PYTHON      interpreter of an environment with this repository installed (default: python3)
#   OPENAI_API_KEY  needed: the P2 SM stage calls gpt-5.4-mini (3-4 calls per task, about 0.01 USD)
#
# The checkpoints are trained by cluster/slurm/run_bob_em_train.sbatch. The paper's runs used one
# RTX A6000 GPU, 12 CPUs and 128 GB per task.
#
# Steps:
#   1. build_committees.py: best-only committees from the reported run (winner per stage; the full
#      blocking roster with only each pair's selected blocker built; Ditto inference cache off).
#   2. timed_bob.py: the repository's run_best_of_breed.py in replay mode with the best-only committees
#      and the reported checkpoints, timed per stage; the wall clock of this phase goes to phases.tsv.
#   3. timing_summary.py: training + pipeline without the post-hoc e2e panel = the printed runtime.
#      The training seconds of the reported checkpoints are passed with TRAINING (e.g.
#      TRAINING="ditto_plm=54.3"); the paper's values are in results/paper_tables/table9/p2_runtime.csv.
#
# The paper's timing jobs also re-trained the models (timed only, not used) and compared every output of
# the timed run with the reported run (verdicts in results/paper_tables/table9/p2_runtime.csv).
set -euo pipefail
: "${DOMAIN:?DOMAIN must be set}"; : "${RUN:?RUN must be set}"
: "${DITTO_CKPT:?DITTO_CKPT must be set}"; : "${SCB_CKPT:?SCB_CKPT must be set}"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY is not set: the P2 SM stage needs it" >&2; exit 3; }
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python3}"
REF_RUN="$REPO/results/best of breeds/$DOMAIN/baseline"
DITTO_CKPT="$(cd "$DITTO_CKPT" && pwd)"; SCB_CKPT="$(cd "$SCB_CKPT" && pwd)"

mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
RUN="$(cd "$RUN" && pwd)"
cd "$RUN"
{ echo "JOB=${SLURM_JOB_ID:-local$$}"; echo "NODE=$(hostname)"; echo "DOMAIN=$DOMAIN"; echo "CODE=$REPO"
  echo "DITTO_CKPT=$DITTO_CKPT"; echo "SCB_CKPT=$SCB_CKPT"; echo "STARTED=$(date -u +%FT%TZ)"; } | tee job.env

"$PY" "$HERE/build_committees.py" --domain "$DOMAIN" --code "$REPO" --ref-run "$REF_RUN" \
    --out "$RUN/best_only" --ditto-cache off

s=$(date +%s.%N); rc=0
"$PY" "$HERE/timed_bob.py" --code "$REPO" --timing-json "$RUN/timing_pipeline.json" \
    --per-pair-blockers "$RUN/best_only/per_pair_blockers.json" -- \
    --config "$RUN/best_only/$DOMAIN.yaml" --committee-dir "$RUN/best_only/committees" \
    --variant baseline --mode replay --out "$RUN/pipeline" \
    --llm-sm --no-llm-em --no-llm-fusion \
    --ditto-checkpoint-override "$DITTO_CKPT" --sc-block-checkpoint-override "$SCB_CKPT" || rc=$?
e=$(date +%s.%N)
echo "pipeline $s $e $rc" >> "$RUN/phases.tsv"
echo "FINISHED=$(date -u +%FT%TZ)" >> job.env
# the pipeline's per-cluster panel files hold gold values; keep them out of the run folder
find "$RUN/pipeline" -name "*_gold.csv" -delete 2>/dev/null || true
[ "$rc" = 0 ] || exit "$rc"
TRAIN_ARGS=()
for t in ${TRAINING:-}; do TRAIN_ARGS+=(--training "$t"); done
"$PY" "$HERE/timing_summary.py" --domain "$DOMAIN" --run "$RUN" "${TRAIN_ARGS[@]}"
