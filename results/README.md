# Phase 1 Reference Results

These results are from Phase 1 (Qwen-2.5-3B + LoRA + AA-only, no Memory Manager).
They serve as a reference baseline. New training will produce results in this same
directory, overwriting these files.

## Setup

- Model: Qwen-2.5-3B-Instruct
- Fine-tuning: LoRA (r=16, alpha=32)
- Agents trained: Answer Agent only (heuristic memories, no MM)
- Reward: Token-level F1
- Hardware: 1× A10G 24GB

## Summary (F1)

| Model | LoCoMo | LongMemEval |
|-------|--------|-------------|
| Baseline (no RL) | 0.112 | 0.116 |
| Config A (LoCoMo only) | 0.126 | 0.124 |
| Config B (Mixed) | 0.113 | 0.124 |
| Config C (LongMemEval only) | 0.112 | 0.116 |

Absolute F1 is low (~12%) because Phase 1 only trained the Answer Agent with
heuristic memories. Phase 2 (7B + LoRA/Full FT + MM) targets 25-40 F1.

## Files

- `all_results.json` — full metrics (F1, BLEU-1, EM) per model, benchmark, and question type
- `config_{a,b,c}_training_meta.json` — hyperparameters used for each config
- `*_predictions.jsonl` — per-example predictions with gold answers
