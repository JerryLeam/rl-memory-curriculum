#!/bin/bash
# HTCondor wrapper for one training config.
# Project source + data + checkpoints live on /staging (mounted on isyegpu4000).
# Conda-packed env is unpacked into the per-job scratch dir at start.
#
# Args:
#   $1 = path to config YAML, relative to project root (e.g., configs/full_ft/train_locomo_only.yaml)
set -o pipefail

CONFIG="${1:?must provide config path}"
PROJECT=/staging/rwu246/rl-memory-curriculum
ENV_TAR=/staging/rwu246/rl_mem_env.tar.gz

echo "[$(date '+%F %T')] host=$(hostname) config=$CONFIG"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vLLM 0.17.x can't parse UUID-format CUDA_VISIBLE_DEVICES that Condor sets.
# Cgroup isolation still restricts the process to the assigned physical GPU,
# so renaming it to index 0 inside this process is safe.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* || "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    echo "[$(date '+%F %T')] translating CUDA_VISIBLE_DEVICES UUID '$CUDA_VISIBLE_DEVICES' -> '0'"
    export CUDA_VISIBLE_DEVICES=0
fi

cd "$PROJECT"

echo "[$(date '+%F %T')] env ready; gpu sanity check"
python -c "import torch; print(f'torch={torch.__version__} cuda_avail={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo "[$(date '+%F %T')] launching train_grpo.py --config $CONFIG --agent both"
python src/train_grpo.py --config "$CONFIG" --agent both
rc=$?
echo "[$(date '+%F %T')] train rc=$rc config=$CONFIG"
exit $rc
