# Future Work & Known Limitations

## Current Status

Phase 2: Qwen-2.5-7B + LoRA + AA+MM for all 3 curriculum configs.
Default configs run on any GPU ≥48GB. Full FT configs available for ≥80GB GPUs.

## Known Limitations

- Single-turn MM training (one CRUD decision per prompt, not full multi-turn episodes)
- Single GPU training (smaller effective batch size than paper's 4×H100 setup)
- LoRA by default (full FT configs provided for larger GPUs)

## Future Extensions

### Multi-Turn Memory Manager Training (highest priority)
- Current MM training is single-turn. The full Memory-R1 approach processes
  entire conversations sequentially with delayed reward after all turns + QA.
- Estimated +3 F1 points over single-turn MM.
- Fits on L40S 48GB with sequential rollout approach.
- ~10-20× more GPU time per config.

### Full Fine-Tuning
- Full FT configs provided in `configs/full_ft/` (needs ≥80GB GPU).
- Expected +3-5 F1 over LoRA.

### MSC Benchmark
- MSC test set not yet prepared.
- Need to download from ParlAI and convert to JSONL format.

### EverMemBench
- 326M tokens, 1M+ token conversations. Tests memory at extreme scale.

### Model Size Sweep
- Run 3B/7B/14B to show scaling behavior (matches paper's Figure 3).

### LLaMA Backbone
- Paper reports results on both LLaMA-3.1-8B and Qwen-2.5-7B.
- Adding LLaMA would strengthen cross-architecture generalization claims.

### LoRA vs Full FT Ablation
- Compare LoRA and full FT directly to isolate the effect.
- All configs within each setup use identical training, so curriculum comparison is valid regardless.

### Distributed Training (4× GPU)
- Paper uses 4× H100 with VERL, batch size 128.
- Distributed path (VERL + FSDP + vLLM) available in archived branch.

### Reviewer Anticipated Questions
1. "Why single-turn MM?" → Noted limitation, multi-turn is future work.
2. "Why only Qwen?" → Add LLaMA-3.1-8B in future.
3. "Is 60 LongMemEval examples enough for Config C?" → Data scaling curve (10, 20, 40, 60).
4. "How does this compare to heuristic methods?" → Add Mem0 baseline (inference-only).
