#!/bin/bash
# Debug variant of eval wrapper: verbose vLLM logging + Python faulthandler.
# Args: model name(s) to evaluate.
set -o pipefail
PROJECT=/staging/rwu246/rl-memory-curriculum
ENV_TAR=/staging/rwu246/rl_mem_env.tar.gz
echo "[$(date '+%F %T')] DEBUG host=$(hostname) models=$*"
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
# Verbose vLLM diagnostics — try to capture what's killing the process
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION=0   # too verbose; only enable if DEBUG isn't enough
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0
# Python faulthandler — dumps stack on segfault/abort
export PYTHONFAULTHANDLER=1

if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* || "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    echo "[$(date '+%F %T')] translating CUDA_VISIBLE_DEVICES UUID '$CUDA_VISIBLE_DEVICES' -> '0'"
    export CUDA_VISIBLE_DEVICES=0
fi

cd "$PROJECT"
echo "[$(date '+%F %T')] launching debug eval"
python -X faulthandler eval/run_eval.py \
    --config configs/eval.yaml \
    --backend vllm \
    --gpus 1 \
    --skip-judge \
    --models "$@"
rc=$?
echo "[$(date '+%F %T')] debug eval rc=$rc models=$*"
exit $rc
