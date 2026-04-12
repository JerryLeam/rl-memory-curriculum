"""
GRPO training for Memory-R1 Answer Agent and Memory Manager using TRL + Unsloth.

Uses Unsloth's FastLanguageModel for efficient training of Qwen2.5-7B-Instruct.
Supports both LoRA and full fine-tuning, controlled by YAML config.
No code changes needed between configurations.

Usage:
    # Train Answer Agent with LoRA (default, ~10GB VRAM)
    python -m src.train.grpo --config configs/train_locomo_only.yaml --agent answer_agent

    # Train both agents with full FT (≥40GB GPU)
    python -m src.train.grpo --config configs/full_ft/train_locomo_only.yaml --agent both
"""
import argparse
import json
import logging

import unsloth  # Must be imported before trl/transformers/peft (applies optimizations)  # noqa: F401

from src.common.config import load_config
from src.train.callbacks import (
    RewardLoggingCallback,
    TrainingLogCallback,
    RewardVarianceEarlyStopCallback,
)
from src.train.model import load_model_unsloth
from src.train.rewards import (
    make_aa_reward_func,
    format_reward_func,
    mm_format_reward,
    make_mm_quality_reward,
)
from src.train.datasets import load_training_data, prepare_aa_dataset, prepare_mm_dataset

logger = logging.getLogger(__name__)


def _configure_wandb(wandb_cfg: dict) -> bool:
    """Set wandb env vars and verify the API key.

    Returns True if wandb is ready to use, False otherwise (so callers can
    fall back to tensorboard without crashing the training run).
    """
    import os
    if not wandb_cfg.get("enabled", False):
        return False

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — falling back to tensorboard. "
                       "Install with: uv add wandb")
        return False

    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key or api_key == "your_wandb_api_key_here":
        logger.warning(
            "WANDB_API_KEY is missing or still the placeholder value. "
            "Set it in .env (copy .env.example) — falling back to tensorboard."
        )
        return False

    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "rl-memory-curriculum")
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = wandb_cfg["entity"]

    # Verify the key is valid before handing control to TRL
    try:
        api = wandb.Api(api_key=api_key)
        _ = api.viewer  # lightweight authenticated request
    except Exception as e:
        logger.warning(f"wandb authentication failed ({e}) — falling back to tensorboard.")
        return False

    return True


def train_answer_agent(config: dict):
    """Train Answer Agent with GRPO via TRL + Unsloth."""
    from trl import GRPOConfig, GRPOTrainer

    model_name = config["model"]["name"]
    exp_name = config["experiment"]["name"]
    data_path = config["data"]["train_file"]
    seed = config["experiment"]["seed"]
    reward_type = config.get("reward", {}).get("type", "f1")
    retrieval_top_k = config.get("retrieval", {}).get("top_k", 20)

    logger.info(f"=== Training Answer Agent: {exp_name} ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Reward: {reward_type}")
    logger.info(f"Retrieval top-k: {retrieval_top_k}")

    # Load data and prepare dataset
    train_data = load_training_data(data_path)
    dataset = prepare_aa_dataset(train_data, max_memories=retrieval_top_k)
    logger.info(f"Prepared {len(dataset)} training prompts")

    # GRPO training config
    aa_epochs = config["training"].get("aa_epochs", 2)
    group_size = config["training"].get("group_size", 4)
    batch_size = config["training"].get("batch_size", 1)
    grad_accum = config["training"].get("gradient_accumulation_steps", 4)
    lr = config["training"].get("learning_rate", 5e-6)
    max_completion = config["training"].get("aa_max_completion_length", 512)
    max_seq_length = config["training"].get("max_seq_length", 2048)

    output_dir = f"checkpoints/{exp_name}/answer_agent"

    # Validate: generation_batch_size = batch_size * grad_accum must be divisible by num_generations
    gen_batch = batch_size * grad_accum
    if gen_batch % group_size != 0:
        logger.warning(f"generation_batch_size ({gen_batch}) not divisible by group_size ({group_size}). "
                       f"Adjusting group_size to {gen_batch}.")
        group_size = gen_batch

    wandb_cfg = config.get("wandb", {})
    report_to = "wandb" if _configure_wandb(wandb_cfg) else "tensorboard"

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f"{exp_name}_aa",
        learning_rate=lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        num_generations=group_size,
        max_completion_length=max_completion,
        num_train_epochs=aa_epochs,
        save_steps=200,
        save_total_limit=2,
        max_grad_norm=1.0,
        seed=seed,
        report_to=report_to,
    )

    # Load model via Unsloth
    model, tokenizer = load_model_unsloth(config, max_seq_length=max_seq_length)

    # Build reward functions
    aa_reward = make_aa_reward_func(reward_type)

    aa_training_meta = {
        "model": model_name,
        "experiment": exp_name,
        "agent": "answer_agent",
        "epochs": aa_epochs,
        "group_size": group_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_examples": len(dataset),
        "learning_rate": lr,
        "reward_type": reward_type,
        "retrieval_top_k": retrieval_top_k,
        "max_completion_length": max_completion,
        "use_lora": config["training"].get("use_lora", True),
    }

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[aa_reward, format_reward_func],
        args=training_args,
        train_dataset=dataset,
        callbacks=[
            RewardLoggingCallback(),
            TrainingLogCallback(agent_type="aa", training_meta=aa_training_meta),
        ],
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    use_lora = config["training"].get("use_lora", True)
    lora_r = config["training"].get("lora_rank", 16)
    lora_alpha = config["training"].get("lora_alpha", 16)
    meta = {
        "model": model_name,
        "experiment": exp_name,
        "agent": "answer_agent",
        "epochs": aa_epochs,
        "group_size": group_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_examples": len(dataset),
        "learning_rate": lr,
        "reward_type": reward_type,
        "retrieval_top_k": retrieval_top_k,
        "max_completion_length": max_completion,
        "max_seq_length": max_seq_length,
        "use_lora": use_lora,
        "lora_r": lora_r if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "unsloth": True,
    }
    with open(f"{output_dir}/training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"AA training complete. Saved to {output_dir}")


def train_memory_manager(config: dict):
    """
    Train Memory Manager with GRPO via TRL + Unsloth.

    Single-turn: each prompt is a dialogue turn + current memory state.
    Reward = format correctness of CRUD operation.
    Full multi-turn training with downstream QA reward is future work.
    """
    from trl import GRPOConfig, GRPOTrainer

    model_name = config["model"]["name"]
    exp_name = config["experiment"]["name"]
    data_path = config["data"]["train_file"]
    seed = config["experiment"]["seed"]

    logger.info(f"=== Training Memory Manager: {exp_name} ===")

    # Load training data and prepare MM dataset
    train_data = load_training_data(data_path)
    dataset = prepare_mm_dataset(train_data, config)

    # Build reward functions
    mm_quality_reward = make_mm_quality_reward()

    mm_epochs = config["training"].get("mm_epochs", 2)
    group_size = config["training"].get("group_size", 4)
    batch_size = config["training"].get("batch_size", 1)
    grad_accum = config["training"].get("gradient_accumulation_steps", 4)
    lr = config["training"].get("learning_rate", 5e-6)
    max_completion = config["training"].get("mm_max_completion_length", 256)
    max_seq_length = config["training"].get("max_seq_length", 2048)
    output_dir = f"checkpoints/{exp_name}/memory_manager"

    gen_batch = batch_size * grad_accum
    if gen_batch % group_size != 0:
        logger.warning(f"generation_batch_size ({gen_batch}) not divisible by group_size ({group_size}). "
                       f"Adjusting group_size to {gen_batch}.")
        group_size = gen_batch

    wandb_cfg = config.get("wandb", {})
    report_to = "wandb" if _configure_wandb(wandb_cfg) else "tensorboard"

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f"{exp_name}_mm",
        learning_rate=lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        num_generations=group_size,
        max_completion_length=max_completion,
        num_train_epochs=mm_epochs,
        save_steps=300,
        save_total_limit=2,
        max_grad_norm=1.0,
        beta=0.04,
        seed=seed,
        report_to=report_to,
    )

    # Load model via Unsloth
    model, tokenizer = load_model_unsloth(config, max_seq_length=max_seq_length)

    mm_training_meta = {
        "model": model_name,
        "experiment": exp_name,
        "agent": "memory_manager",
        "epochs": mm_epochs,
        "group_size": group_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_examples": len(dataset),
        "learning_rate": lr,
        "max_completion_length": max_completion,
        "use_lora": config["training"].get("use_lora", True),
    }

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[mm_format_reward, mm_quality_reward],
        args=training_args,
        train_dataset=dataset,
        callbacks=[
            RewardLoggingCallback(),
            RewardVarianceEarlyStopCallback(),
            TrainingLogCallback(agent_type="mm", training_meta=mm_training_meta),
        ],
    )

    logger.info("Starting MM GRPO training...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    use_lora = config["training"].get("use_lora", True)
    lora_r = config["training"].get("lora_rank", 16)
    lora_alpha = config["training"].get("lora_alpha", 16)
    meta = {
        "model": model_name,
        "experiment": exp_name,
        "agent": "memory_manager",
        "epochs": mm_epochs,
        "group_size": group_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_examples": len(dataset),
        "learning_rate": lr,
        "max_completion_length": max_completion,
        "max_seq_length": max_seq_length,
        "use_lora": use_lora,
        "lora_r": lora_r if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "unsloth": True,
    }
    with open(f"{output_dir}/training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"MM training complete. Saved to {output_dir}")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train Memory-R1 with GRPO")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent", type=str, default="both",
                        choices=["memory_manager", "answer_agent", "both"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    config = load_config(args.config)

    exp_name = config["experiment"]["name"]
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"LoRA: {config['training'].get('use_lora', True)}")
    logger.info(f"Reward: {config.get('reward', {}).get('type', 'f1')}")

    if args.agent in ("answer_agent", "both"):
        logger.info("=== Training Answer Agent ===")
        train_answer_agent(config)

    if args.agent in ("memory_manager", "both"):
        logger.info("=== Training Memory Manager ===")
        train_memory_manager(config)

    logger.info(f"Training complete for {exp_name}")


if __name__ == "__main__":
    main()
