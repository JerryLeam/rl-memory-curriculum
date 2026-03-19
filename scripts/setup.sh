#!/bin/bash
# Generic setup: create venv, install deps, download tokenizer.
# Works on any machine with Python 3.10+ and a GPU.
set -e

echo "=== rl-memory-curriculum setup ==="

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Pre-download tokenizer (model weights download on first training run)
echo "Pre-downloading tokenizer..."
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print(f'Tokenizer ready: vocab_size={tok.vocab_size}')
"

# Verify GPU
echo ""
echo "Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'GPU: {name} ({vram:.1f} GB)')
    if vram < 48:
        print('WARNING: <48GB VRAM. Default configs need ≥48GB. See configs/full_ft/ for ≥80GB.')
else:
    print('WARNING: No GPU detected. Training requires a CUDA GPU.')
"

echo ""
echo "Checking data..."
if [ -f "data/processed/locomo_train.jsonl" ]; then
    echo "Processed data found. Ready to train."
elif [ -f "data/raw/locomo/data/locomo10.json" ]; then
    echo "Raw data found. run_all.sh will process it automatically."
else
    echo "NOTE: No data found. Clone raw data before training:"
    echo "  git clone https://github.com/snap-research/locomo data/raw/locomo"
    echo "  git clone https://github.com/xiaowu0162/LongMemEval data/raw/longmemeval"
fi

echo ""
echo "Setup complete. Run: bash scripts/run_all.sh"
