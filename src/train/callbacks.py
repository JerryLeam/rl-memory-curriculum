"""
Training callbacks for GRPO training.
"""
import json
import logging
from pathlib import Path
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class RewardLoggingCallback(TrainerCallback):
    """Print reward statistics to stdout for real-time monitoring."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        reward_keys = [k for k in logs if "reward" in k.lower()]
        if reward_keys:
            stats = {k: f"{logs[k]:.4f}" for k in reward_keys}
            logger.info(f"Step {state.global_step} rewards: {stats}")


class TrainingLogCallback(TrainerCallback):
    """Write structured JSONL training logs for paper figures.

    Captures per-step: loss, reward stats, grad norm, learning rate, wall time.
    For MM training, also captures CRUD operation distribution from completions.

    Output: one JSONL file per training run at `logs/{run_name}_training.jsonl`.
    Each line is a self-contained JSON object — crash-safe, easy to parse with pandas.
    Also logs to wandb if available.
    """

    def __init__(self, log_dir="logs", agent_type="aa", training_meta=None):
        self.log_dir = Path(log_dir)
        self.agent_type = agent_type
        self.training_meta = training_meta
        self._file = None
        self._start_time = None
        self._wandb_available = False

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{args.run_name}_training.jsonl"
        self._file = open(log_path, "a", encoding="utf-8")
        self._start_time = time.time()
        logger.info(f"Training log: {log_path}")

        # Log training config to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                self._wandb_available = True
                if self.training_meta:
                    wandb.config.update(self.training_meta, allow_val_change=True)
        except ImportError:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        import time
        if logs is None or self._file is None:
            return

        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 3) if state.epoch else 0,
            "wall_time_s": round(time.time() - self._start_time, 1),
            "agent": self.agent_type,
        }

        # Standard training metrics
        for key in ("loss", "grad_norm", "learning_rate"):
            if key in logs:
                record[key] = round(logs[key], 6)

        # Reward metrics (TRL logs these with various key patterns)
        for key, val in logs.items():
            if "reward" in key.lower():
                record[key.replace("/", "_")] = round(val, 6)

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._file is not None:
            self._file.close()
            self._file = None


class RewardVarianceEarlyStopCallback(TrainerCallback):
    """Stop training when reward variance collapses (all GRPO samples score the same).

    When std(rewards) < threshold for `patience` consecutive logging steps,
    GRPO advantages are effectively zero and further training is wasted compute.
    """

    def __init__(self, std_threshold=0.01, patience=20):
        self.std_threshold = std_threshold
        self.patience = patience
        self.low_variance_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # TRL >= 0.29 logs reward/std; some versions use rewards/std
        reward_std = None
        for key in ("reward/std", "rewards/std", "reward_std"):
            if key in logs:
                reward_std = logs[key]
                break
        if reward_std is not None and reward_std < self.std_threshold:
            self.low_variance_count += 1
            if self.low_variance_count >= self.patience:
                logger.warning(
                    f"Reward variance collapsed (std={reward_std:.4f} for "
                    f"{self.patience} consecutive steps). Stopping early."
                )
                control.should_training_stop = True
        else:
            self.low_variance_count = 0


class WandbSampleTableCallback(TrainerCallback):
    """Log a wandb Table with the best-reward sample every N training steps.

    Accumulates the best sample (highest total reward) from each batch of
    reward evaluations.  Every ``log_every`` training steps the full
    accumulated table is pushed to wandb under ``"reward_samples"``.

    Columns: step, completion, extracted_answer, ground_truth,
    one column per reward function, total_reward.

    Parameters
    ----------
    sample_logger : SampleLogger
        Shared buffer populated by ``wrap_reward_func`` wrappers.
    reward_names : list[str]
        Column names for each reward function (order must match).
    log_every : int
        Push the table to wandb every this many training steps.
    """

    def __init__(self, sample_logger, reward_names, log_every=10):
        self.sample_logger = sample_logger
        self.reward_names = reward_names
        self.log_every = log_every
        self._columns = ["step", "prompt", "completion", "extracted_answer", "ground_truth"] + \
                        list(reward_names) + ["total_reward"]
        self._rows: list[list] = []
        self._last_logged_step = -1
        self._sample_log_file = None

    def _pick_best_from_buffer(self, step):
        """Drain the sample logger and append the best sample to _rows."""
        records = self.sample_logger.drain()
        if not records:
            return

        per_reward = {}
        for rec in records:
            per_reward.setdefault(rec["reward_name"], []).append(rec)

        n_samples = max(len(v) for v in per_reward.values()) if per_reward else 0
        if n_samples == 0:
            return

        best_row = None
        best_total = -1.0
        best_rec_full = None
        for i in range(n_samples):
            first_rec = None
            for rname in self.reward_names:
                batch = per_reward.get(rname, [])
                if i < len(batch):
                    first_rec = batch[i]
                    break
            if first_rec is None:
                continue

            reward_scores = []
            for rname in self.reward_names:
                batch = per_reward.get(rname, [])
                score = batch[i]["score"] if i < len(batch) else 0.0
                reward_scores.append(round(score, 4))

            total = round(sum(reward_scores), 4)
            if total > best_total:
                best_total = total
                best_rec_full = first_rec
                best_row = [
                    step,
                    first_rec.get("prompt", "")[:1000],
                    first_rec["completion"][:1000],
                    first_rec.get("extracted_answer", "") or "",
                    str(first_rec.get("gold", ""))[:500],
                    *reward_scores,
                    total,
                ]

        if best_row is not None:
            self._rows.append(best_row)
            # Write full (untruncated) sample to plain text log file
            self._write_sample_log(best_row, best_rec_full)

    def _write_sample_log(self, row, full_rec):
        """Append best sample details to logs/reward_samples.log.

        Uses full_rec for untruncated prompt/completion; row for scores.
        """
        if self._sample_log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            self._sample_log_file = open(log_dir / "reward_samples.log", "a", encoding="utf-8")
        f = self._sample_log_file
        f.write(f"{'='*80}\n")
        f.write(f"Step: {row[0]}\n")
        f.write(f"Total Reward: {row[-1]}\n")
        f.write(f"Rewards: {dict(zip(self.reward_names, row[5:-1]))}\n")
        f.write(f"--- PROMPT (FULL) ---\n{full_rec.get('prompt', '')}\n")
        f.write(f"--- COMPLETION (FULL) ---\n{full_rec.get('completion', '')}\n")
        f.write(f"--- EXTRACTED ANSWER ---\n{full_rec.get('extracted_answer', '')}\n")
        f.write(f"--- GROUND TRUTH ---\n{full_rec.get('gold', '')}\n")
        f.write(f"{'='*80}\n\n")
        f.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._pick_best_from_buffer(state.global_step)

        if state.global_step - self._last_logged_step < self.log_every:
            return
        if not self._rows:
            return

        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        self._last_logged_step = state.global_step
        table = wandb.Table(columns=self._columns, data=self._rows)
        # NOTE: wandb.log without commit=False may interfere with TRL's
        # own wandb logging (step counters, metric grouping).  We omit the
        # step= arg and let wandb auto-increment to avoid conflicts.
        wandb.log({"reward_samples": table})
        logger.info(f"Logged reward_samples table with {len(self._rows)} rows at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        """Flush any remaining rows at the end of training."""
        self._pick_best_from_buffer(state.global_step)
        if not self._rows:
            return
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return
        table = wandb.Table(columns=self._columns, data=self._rows)
        # NOTE: see on_log comment — omit step= to avoid conflicts with TRL.
        wandb.log({"reward_samples": table})
        logger.info(f"Final reward_samples table: {len(self._rows)} rows")
        if self._sample_log_file is not None:
            self._sample_log_file.close()
            self._sample_log_file = None
