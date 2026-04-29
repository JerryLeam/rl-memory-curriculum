#!/bin/bash
# HTCondor wrapper for a subset of eval models on a single GPU.
#
# Args:
#   $@ = list of model names (e.g., baseline_no_rl config_a_aa_only config_a_full)
set -o pipefail

if [ "$#" -lt 1 ]; then
    echo "ERROR: must provide at least one model name"
    exit 2
fi

PROJECT=/staging/rwu246/rl-memory-curriculum
ENV_TAR=/staging/rwu246/rl_mem_env.tar.gz

echo "[$(date '+%F %T')] host=$(hostname) models=$*"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

SCRATCH=${_CONDOR_SCRATCH_DIR:-/tmp/rwu246_$$}
ENV_DIR="$SCRATCH/env"
mkdir -p "$ENV_DIR"

echo "[$(date '+%F %T')] unpacking env to $ENV_DIR ..."
tar -xzf "$ENV_TAR" -C "$ENV_DIR" || { echo "tar extract FAILED"; exit 10; }
"$ENV_DIR/bin/conda-unpack" || { echo "conda-unpack FAILED"; exit 11; }

export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/staging/rwu246/hf_cache
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

# vLLM can't parse UUID-format CUDA_VISIBLE_DEVICES; cgroup isolation still restricts.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* || "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    echo "[$(date '+%F %T')] translating CUDA_VISIBLE_DEVICES UUID '$CUDA_VISIBLE_DEVICES' -> '0'"
    export CUDA_VISIBLE_DEVICES=0
fi

cd "$PROJECT"

echo "[$(date '+%F %T')] launching run_eval.py for: $*"
python eval/run_eval.py \
    --config configs/eval.yaml \
    --backend vllm \
    --gpus 1 \
    --skip-judge \
    --models "$@"
rc=$?
echo "[$(date '+%F %T')] eval rc=$rc models=$*"
exit $rc
