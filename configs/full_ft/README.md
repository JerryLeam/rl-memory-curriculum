# Full Fine-Tuning Configs

Use these configs instead of the default ones if you have a GPU with ≥80GB VRAM
(H100 80GB, A100 80GB).

Full FT trains all model parameters instead of LoRA adapters, which typically
yields +3-5 F1 points over LoRA at the cost of higher memory usage.

## Usage

```bash
bash scripts/run_all.sh --config-dir configs/full_ft
```

## Requirements

- 1× GPU with ≥80GB VRAM (H100 80GB or A100 80GB)
- Gradient checkpointing is enabled by default (required for single-GPU full FT)
- ~50GB disk for checkpoints (full model weights saved, not just adapters)

## Differences from Default (LoRA) Configs

| Parameter | Default (LoRA) | Full FT |
|-----------|---------------|---------|
| `use_lora` | true | false |
| `batch_size` | 1 | 2 |
| `learning_rate` | 5e-6 | 1e-6 |
| `gradient_accumulation_steps` | 4 | 2 |
| Min VRAM | 48 GB | 80 GB |
