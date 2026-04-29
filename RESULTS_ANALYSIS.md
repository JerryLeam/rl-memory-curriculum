# Results Analysis — Phase 2 (Qwen-2.5-7B Full FT, single H200)

## Headline numbers

F1 across both benchmarks:

| Model | LoCoMo F1 | LongMemEval F1 | Δ from baseline |
|---|---|---|---|
| Qwen-2.5-7B (no RL) | 0.119 | 0.131 | — |
| Config A AA-only (LoCoMo) | 0.125 | 0.130 | +0.006 / -0.001 |
| Config B AA-only (Mixed) | 0.123 | 0.133 | +0.004 / +0.002 |
| Config C AA-only (LongMemEval) | 0.119 | 0.133 | 0.000 / +0.002 |
| **Config A AA+MM (LoCoMo)** | **0.166** | **0.147** | **+0.047 / +0.016** |
| **Config B AA+MM (Mixed)** | **0.164** | **0.140** | **+0.045 / +0.009** |
| **Config C AA+MM (LongMemEval)** | **0.160** | **0.148** | **+0.041 / +0.017** |

## MM ablation (the interesting comparison)

| Config | LoCoMo Δ from MM | LongMemEval Δ from MM |
|---|---|---|
| Config A (LoCoMo) | **+0.041** | **+0.017** |
| Config B (Mixed) | **+0.041** | **+0.007** |
| Config C (LongMemEval) | **+0.040** | **+0.016** |

## Three takeaways

### 1. AA-only training barely helps

Across all three curriculum configurations, training the Answer Agent alone moves LoCoMo F1 by 0.000-0.006 over baseline — within run-to-run noise. On LongMemEval the AA-only Δ is at most +0.002. **The AA is not the bottleneck.** This matches the prediction in the friend's monitoring doc: "If task reward is dead, everything downstream is wasted" — but we observed the opposite, the task reward signal is alive (formed correct XML, gold-answer F1 reward firing healthily) but the AA's improvements are masked because retrieval quality dominates the answer.

### 2. Adding the trained MM is what moves the needle

AA+MM full pipeline gains **+0.040 F1 on LoCoMo across all three configs**, and +0.007-0.017 on LongMemEval. This is a real, consistent effect across curricula. The MM's CRUD operations on the memory bank during eval-time memory construction give the AA cleaner / better-organized retrieved memories than the heuristic baseline, and the AA's answers benefit accordingly.

### 3. The curriculum effect is tiny

Configs A, B, and C — three different training data mixtures (LoCoMo-only, Mixed, LongMemEval-only) — all converge to nearly the same final F1 (LoCoMo ≈ 0.16-0.17, LongMemEval ≈ 0.14-0.15). **The choice of training mix matters less than having MM at all.** This is consistent with the friend's UPGRADE_PLAN_2xH200.md claim that the bottleneck for AA+MM gains is `group_size=4` (high variance in GRPO advantages) rather than data-mix design.

## Comparison to the paper's reference

Memory-R1 paper reports ~45 F1 on LoCoMo with Qwen-2.5-7B + 4× H100 + group_size=128 + multi-turn MM training with QA-outcome reward. We achieved 0.166 on LoCoMo. The ~28-point gap is consistent with the friend's predicted upgrade-impact ordering:

| Upgrade we are missing | Predicted lift | Notes |
|---|---|---|
| group_size 4 → 16-32 | +5-10 F1 | Highest priority — single config change |
| Multi-turn MM training | +5-8 F1 | Architectural change, biggest engineering lift |
| QA-based MM reward (vs format-only) | +3-5 F1 | What our MM-quality-reward graphs show is happening — quality reward stays at 0.005-0.030, well below the friend's "ideal 0.02-0.10". The reward signal is too sparse/noisy with format-only |
| Cross-encoder reranking at retrieval | +1-3 F1 | Independent of training; can be added at eval time |
| Total predicted | +14-26 F1 | Brings us to ~30-42, within striking distance of paper |

## Per-question-type breakdown (LongMemEval, AA-only)

`paper/tables/table2_per_type.md`. Notable:
- `single-session-assistant` and `temporal-reasoning` have the highest F1 across all configs (0.14-0.20) — these question types require less memory aggregation, so the heuristic memory bank is more useful out of the box.
- `multi-session` and `multi_hop` have the lowest F1 (0.07-0.10) — these specifically require connecting information across turns/sessions, which is exactly what a good MM should help with. Worth re-running the per-type analysis for AA+MM full to see if the lift is concentrated in these multi-step types.

## Reward signal sanity (per the friend's monitoring doc)

Final-step training metrics for the AA phase:

| Job | rewards/reward_func/mean | format_reward/mean | reward_std | grad_norm | kl |
|---|---|---|---|---|---|
| TRAIN_A (LoCoMo) | 0.20-0.27 (healthy, > 0.10) | 0.17-0.18 (saturated as expected) | 0.05-0.08 (well above 0.03 threshold) | 4-7 (yellow per doc) | 0.005-0.008 |
| TRAIN_B (Mixed) | similar | similar | similar | 5-9 | similar |
| TRAIN_C (LongMemEval) | similar | similar | similar | 5-7 | similar |

✅ Task reward alive (well above 0.10). ✅ Reward variance well above the early-stop threshold (0.01). The yellow flag was sustained `grad_norm ≈ 5-9` — within the friend's "occasional spikes are fine" range, not the "sustained 10+ = unstable" zone. Loss did not diverge.

For the MM phase: `mm_format_reward` saturated at 0.78-0.80 within ~10 steps (expected); `mm_quality_reward` averaged 0.005-0.030 (below the friend's 0.02-0.10 ideal). `frac_reward_zero_std` hovered at 0.5-0.7 — i.e. half the GRPO groups gave zero gradient signal in any given step. **This is the diagnosed mechanism for the small MM contribution: the format reward saturates fast and the quality reward isn't producing enough variance for GRPO to learn from.** The `RewardVarianceEarlyStopCallback` did not fire, so training continued, but the effective signal was weak.

The friend's recommendation: switch to a frozen-AA outcome-based MM reward in a future run. That's the single highest-priority change to close the gap to the paper.

## Time / cost

| Phase | Wall clock | Notes |
|---|---|---|
| Data prep | ~5 min | Done once on AP |
| Conda env build + tarball | ~10 min | One-time setup |
| Training (3 configs, sequential AA→MM each, all running in parallel on 3 GPUs) | ~5.5h longest config (TRAIN_B Mixed) | TRAIN_A 3.5h, TRAIN_B 5.85h, TRAIN_C 3.16h |
| Eval (3 × `_full` configs, 3 GPUs in parallel after the patch) | ~26.5h longest | Each eval does MM memory construction over both benchmarks (~210 K turns), then inference. The slow phase is the per-turn MM batch loop, not inference. |
| Aggregate + paper tables | ~30s | Local on AP |
| **Total** | **~36 hours wall-clock** | Of which ~15 hours was lost to crash diagnosis on the unfixed code path |

After the patch, a clean re-eval would be ~26 hours. With the friend's UPGRADE_PLAN_2xH200.md changes (G=16 + multi-turn MM + QA-reward + reranking), expected ~50-80 hours total wall-clock.
