"""
Eager tester — quickly run the pipeline on a few examples and print results.

Usage:
    python -m src.tester                          # baseline, 5 locomo examples
    python -m src.tester -n 10                    # 10 examples
    python -m src.tester --checkpoint path/to/ckpt
    python -m src.tester --mm-checkpoint path/to/mm_ckpt  # use trained Memory Manager
"""
import argparse
import json
import logging
import sys
from collections import defaultdict

from src.common.scoring import token_f1, bleu1, exact_match
from src.eval.inference import format_aa_prompt, generate_answer, run_mm_on_sessions
from src.eval.model_loader import load_model_and_tokenizer, load_mm_model
from src.memory.heuristic import build_heuristic_memories, retrieve_memories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# DEFAULT_MODEL = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="Eager tester for Memory-R1 pipeline")
    p.add_argument("--checkpoint", default=DEFAULT_MODEL,
                   help="Model checkpoint path or HF model name (default: %(default)s)")
    p.add_argument("--base-model", default=DEFAULT_MODEL,
                   help="Base model for LoRA adapters (default: %(default)s)")
    p.add_argument("-n", "--num-examples", type=int, default=5,
                   help="Number of examples to run (default: %(default)s)")
    p.add_argument("--data-file", default="data/processed/locomo_test.jsonl",
                   help="Input JSONL file (default: %(default)s)")
    p.add_argument("--top-k", type=int, default=20,
                   help="Retrieval top-k memories (default: %(default)s)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max new tokens for generation (default: %(default)s)")
    p.add_argument("--mm-checkpoint", default=None,
                   help="Trained Memory Manager checkpoint (omit for heuristic memory)")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Generation temperature (default: %(default)s)")
    return p.parse_args()


def load_data(path, n):
    """Load the first *n* examples from a JSONL file."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
                if len(examples) >= n:
                    break
    return examples


def main():
    args = parse_args()

    # ── Load data ───────────────────────────────────────────────────────
    logger.info("Loading %d examples from %s", args.num_examples, args.data_file)
    examples = load_data(args.data_file, args.num_examples)
    if not examples:
        logger.error("No examples loaded — check the data file path.")
        sys.exit(1)
    logger.info("Loaded %d examples", len(examples))

    # ── Load model ──────────────────────────────────────────────────────
    is_baseline = (args.checkpoint == DEFAULT_MODEL)
    model_config = {
        "name": "eager_test",
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "is_baseline": is_baseline,
    }
    logger.info("Loading answer-agent model: %s", args.checkpoint)
    model, tokenizer = load_model_and_tokenizer(model_config)

    # Optional: load trained Memory Manager
    mm_model, mm_tokenizer = None, None
    if args.mm_checkpoint:
        mm_config = {
            "name": "eager_test_mm",
            "mm_checkpoint": args.mm_checkpoint,
            "base_model": args.base_model,
        }
        logger.info("Loading Memory Manager: %s", args.mm_checkpoint)
        mm_model, mm_tokenizer = load_mm_model(mm_config)

    use_mm = mm_model is not None

    # ── Group by conversation ───────────────────────────────────────────
    conv_groups = defaultdict(list)
    for ex in examples:
        conv_groups[ex["conversation_id"]].append(ex)

    # ── Run inference ───────────────────────────────────────────────────
    results = []
    example_idx = 0

    for conv_id, conv_examples in conv_groups.items():
        sessions = conv_examples[0].get("sessions", [])

        # Build memories
        if use_mm:
            logger.info("  Conv %s: building memories with trained MM...", conv_id)
            memories = run_mm_on_sessions(mm_model, mm_tokenizer, sessions)
        else:
            memories = build_heuristic_memories(sessions)
        logger.info("  Conv %s: %d memories, %d questions",
                     conv_id, len(memories), len(conv_examples))

        for ex in conv_examples:
            example_idx += 1
            question = ex["question"]
            gold = ex["answer"]

            retrieved = retrieve_memories(question, memories, top_k=args.top_k)
            predicted = generate_answer(
                model, tokenizer, question, retrieved,
                max_new_tokens=args.max_new_tokens,
            )

            f1 = token_f1(predicted, gold)
            b1 = bleu1(predicted, gold)
            em = exact_match(predicted, gold)

            results.append({"f1": f1, "bleu1": b1, "em": em})

            # ── Print ───────────────────────────────────────────────────
            print(f"\n{'─' * 4} Example {example_idx}/{len(examples)} {'─' * 40}")
            print(f"Conversation : {conv_id}")
            print(f"Type         : {ex.get('question_type', 'unknown')}")
            print(f"Question     : {question}")
            print(f"Gold         : {gold}")
            print(f"Predicted    : {predicted}")
            print(f"F1: {f1:.4f} | BLEU-1: {b1:.4f} | EM: {em:.0f}")

    # ── Aggregate ───────────────────────────────────────────────────────
    n = len(results)
    if n:
        avg_f1 = sum(r["f1"] for r in results) / n
        avg_b1 = sum(r["bleu1"] for r in results) / n
        avg_em = sum(r["em"] for r in results) / n
        print(f"\n{'═' * 56}")
        print(f"  Aggregate ({n} examples)")
        print(f"  F1: {avg_f1:.4f} | BLEU-1: {avg_b1:.4f} | EM: {avg_em:.4f}")
        print(f"  Memory mode: {'trained MM' if use_mm else 'heuristic'}")
        print(f"  Model: {args.checkpoint}")
        print(f"{'═' * 56}")


if __name__ == "__main__":
    main()
