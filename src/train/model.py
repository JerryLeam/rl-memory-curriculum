"""
Model loading for GRPO training.

Loads the model via HuggingFace Transformers and optionally applies LoRA via PEFT.
"""
import logging

logger = logging.getLogger(__name__)


def load_model(config: dict):
    """Load model and tokenizer. Applies LoRA if configured, otherwise full FT.

    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model"]["name"]
    use_lora = config["training"].get("use_lora", True)

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Mode: {'LoRA' if use_lora else 'Full fine-tuning'}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_r = config["training"].get("lora_rank", 16)
        lora_alpha = config["training"].get("lora_alpha", 16)
        logger.info(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})")

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # PeftModel doesn't inherit warnings_issued from PreTrainedModel,
    # but TRL's Trainer.__init__ expects it. Set it to avoid AttributeError.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    return model, tokenizer
