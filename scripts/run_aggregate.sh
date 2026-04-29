#!/bin/bash
# Aggregate per-model prediction files into all_results.json + paper tables. CPU only.
set -o pipefail

PROJECT=/staging/rwu246/rl-memory-curriculum
ENV_TAR=/staging/rwu246/rl_mem_env.tar.gz

SCRATCH=${_CONDOR_SCRATCH_DIR:-/tmp/rwu246_$$}
ENV_DIR="$SCRATCH/env"
mkdir -p "$ENV_DIR"

echo "[$(date '+%F %T')] unpacking env to $ENV_DIR ..."
tar -xzf "$ENV_TAR" -C "$ENV_DIR" || { echo "tar extract FAILED"; exit 10; }
"$ENV_DIR/bin/conda-unpack" || { echo "conda-unpack FAILED"; exit 11; }

export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

cd "$PROJECT"

echo "[$(date '+%F %T')] aggregating eval results"
python eval/run_eval.py --config configs/eval.yaml --aggregate-only

echo "[$(date '+%F %T')] generating paper tables"
mkdir -p paper/tables
python eval/analyze_results.py \
    --results results/all_results.json \
    --output paper/tables/

echo "[$(date '+%F %T')] aggregation done"
