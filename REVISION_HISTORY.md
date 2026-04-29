# Revision History — Phase 2 H200 Run (CHTC isyegpu4000)

**Run dates:** 2026-04-26 → 2026-04-29
**Hardware:** 1× NVIDIA H200 NVL (143 GB HBM3e), 16 cores, 200 GB RAM per job, on `isyegpu4000.chtc.wisc.edu`
**Model:** Qwen/Qwen2.5-7B-Instruct, full fine-tuning, bf16
**Result:** All 7 models trained & evaluated on both LoCoMo (1307 Q) and LongMemEval (415 Q). Final tables in `paper/tables/`.

---

## Timeline of changes

### 1. Submission infrastructure for CHTC (commit `f4ac5ae`)

CHTC's `isyegpu4000` has unusual filesystem semantics that broke the project's `run_all.sh`:
- `/staging` (CephFS) is mounted on the execute slot, but `/home` is NOT.
- The CHTC AP rejects `initialdir = /staging/...` as invalid.
- The conda-pack tarball (~5.4 GB) is the only practical way to ship the Python env to the execute slot inside CephFS's 40 000-file inode quota on `/staging/<user>`.

Files added:
- `scripts/run_train_one.sh` — invoked by HTCondor for one training config; unpacks env tarball into `$_CONDOR_SCRATCH_DIR`, runs `conda-unpack`, translates UUID-format `CUDA_VISIBLE_DEVICES` to `0` (vLLM 0.17.x can't parse the UUID; cgroup isolation still pins to the right physical GPU), then `cd /staging/.../rl-memory-curriculum && python src/train_grpo.py`.
- `scripts/run_eval_subset.sh`, `run_eval_debug.sh`, `run_aggregate.sh` — same pattern for eval and aggregation phases.
- `submit/{train_a,train_b,train_c,eval_a,eval_b,eval_c,aggregate}.sub` + `submit/run.dag` — HTCondor submit files matching CHTC policy: `initialdir` on `/home`, `executable` absolute on `/staging`, `should_transfer_files=YES`, `request_gpus=1`, `require_gpus=NVIDIA H200 NVL`.

### 2. Three crash fixes (commit `1595347`)

Three distinct crashes during the run; fixed in order they appeared.

| # | Crash | Root cause | Fix |
|---|---|---|---|
| 1 | `AssertionError: dtype in (torch.float16, torch.bfloat16, torch.float32)` at `FastLanguageModel.from_pretrained` | Unsloth 2026.3.18 dropped support for the legacy `dtype="auto"` string. | `dtype=torch.bfloat16` in `src/train_grpo.py` (both AA and MM call sites). |
| 2 | `vllm.exceptions.VLLMValidationError: input 16385 tokens > max model length 16384` mid-eval | `configs/eval.yaml` had `max_model_len: 16384` but the MM-build prompt grows linearly with conversation turns and exceeded the cap. | Bump to `max_model_len: 32768` (Qwen2.5-7B's native max). |
| 3 | Same overflow, just at the new ceiling — eval crashed at 76-78% completion when prompts crossed 32 769 tokens. | Memory bank in long conversations grew past *any* fixed `max_model_len`. | Added `build_mm_text_with_budget()` in `src/memory_manager.py`: renders prompt with all memories, falls back to most-recent-N entries (binary halving) if rendered text exceeds `max_prompt_tokens=30000`. Both vLLM MM-build sites in `eval/run_eval.py` use it. |

Also documented `fast_inference=False` as required for `full_finetuning=True` (Unsloth raises `NotImplementedError` otherwise).

### 3. Hyperparameter tuning for H200 (in commit `1595347`)

| Setting | Old | New | Rationale |
|---|---|---|---|
| `batch_size` | 2 | 4 | README: "can go to 4 on H100"; H200 has same headroom |
| `gradient_checkpointing` | true | false | Unsloth still does its own gradient offload for full FT; turning off TRL's checkpoint saves activation-recompute cost when the model fits anyway |
| `wandb.enabled` | true | false | No wandb credentials on the compute node |
| `eval.yaml max_model_len` | 16384 | 32768 | See crash #2 above |

### 4. Results (commit `833f49c`)

`results/all_results.json` aggregates per-question metrics into per-model and per-question-type means. Per-row `*_predictions.jsonl` are gitignored by the project (consistent with existing `.gitignore` rule) — regenerable from checkpoints + test data + this code.

### 5. Paper tables (commit `9ba9719`)

`paper/tables/table{1,2,3,4}_*.md`. See `RESULTS_ANALYSIS.md` for interpretation.

---

## Operational lessons

These shaped the actual run path; future Phase-2-on-CHTC runs should pre-empt them:

1. **`max_model_len` must be set explicitly**, and even Qwen2.5-7B's 32 768 max is not always enough — always pair with prompt-truncation logic.

2. **`NumSystemHolds` does not increment on user holds** (`on_exit_hold`). The naive `periodic_release ... NumSystemHolds < 2` is effectively unlimited retry on user-induced holds. We had two evals each restart 4 times before we noticed, losing ~6 hours of cumulative MM-build progress per job.

3. **eval MM-build phase has no checkpointing** — every crash restarts at batch step 0. Combined with #2 above, this turned a 12 h eval into a never-completing one until the root cause was patched. Future infra work: add per-conversation checkpointing in `run_mm_all_conversations_vllm`.

4. **CephFS inode quota (40 000) is the binding constraint on `/staging`**, not bytes. A naïve uv-installed `.venv` would need ~80 000 inodes; conda-pack tarball is the correct pattern.

5. **`should_transfer_files = YES`** is required even when `/staging` is shared, because Condor's `FileSystemDomain` matching fails on isyegpu4000 from this AP.

6. **`CUDA_VISIBLE_DEVICES=GPU-<uuid>`** (Condor's default for assigned GPUs) breaks vLLM 0.17.x's `int(...)` parser. The wrapper must override to `0`; cgroup isolation still pins to the right physical GPU.

---

## Files in this commit history

```
src/memory_manager.py       — added entries= param + build_mm_text_with_budget()
src/train_grpo.py           — dtype fix, fast_inference comment
eval/run_eval.py            — 2 vLLM sites use the budget-aware helper
configs/eval.yaml           — max_model_len 32768
configs/full_ft/*.yaml      — bs=4, grad_ckpt=false, wandb=false
scripts/run_*.sh            — 4 new HTCondor wrappers
submit/*                    — 8 .sub files + run.dag
paper/tables/*.md           — 4 tables (table1-table4)
results/all_results.json    — aggregated F1/BLEU/EM
.gitignore                  — added rules for mm_memories caches and submit_backup_*
```
