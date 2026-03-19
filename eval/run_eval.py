"""
Run evaluation across all configs and benchmarks.

Behavior controlled by eval YAML config.

Usage:
    python eval/run_eval.py --config configs/eval.yaml
    python eval/run_eval.py --config configs/eval.yaml --skip-judge
    python eval/run_eval.py --config configs/eval.yaml --judge-only
    python eval/run_eval.py --config configs/eval.yaml --models config_a_full
"""
import argparse
import json
import logging
import re
import sys
import yaml
from pathlib import Path
from collections import defaultdict

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import evaluate_predictions, format_results_table, save_results
from eval.judge import judge_batch

logger = logging.getLogger(__name__)


# ============================================================
# Model loading (torch imported lazily - GPU only)
# ============================================================

def load_model_and_tokenizer(model_config):
    """Load model + tokenizer. Handles base model, LoRA, and full FT checkpoints."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = model_config["checkpoint"]
    # Auto-detect checkpoint type from training_meta.json if available
    is_lora = model_config.get("lora", False)
    is_full_ft = model_config.get("full_ft", False)

    if not is_lora and not is_full_ft and not model_config.get("is_baseline", False):
        # Auto-detect from training_meta.json in checkpoint dir
        meta_path = Path(checkpoint) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("use_lora", False):
                is_lora = True
            elif meta.get("use_lora") is False:
                is_full_ft = True
            logger.info(f"Auto-detected from training_meta.json: lora={is_lora}, full_ft={is_full_ft}")

    if is_lora:
        from peft import PeftModel
        base_model_name = model_config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
        logger.info(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        logger.info(f"Loading LoRA adapter: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    elif is_full_ft:
        # Full fine-tuned checkpoint: load directly from saved directory
        logger.info(f"Loading full FT checkpoint: {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    else:
        # Base model (no training)
        logger.info(f"Loading base model: {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


# ============================================================
# Heuristic memory builder (no GPU needed)
# ============================================================

SKIP_WORDS = {"hi", "hello", "hey", "thanks", "bye", "ok", "okay", "yes", "no"}


def build_heuristic_memories(sessions, max_memories=200):
    """Build memory entries from sessions using heuristics."""
    memories = []
    for session in sessions:
        sid = session.get("session_id", 0)
        dt = session.get("date_time", "")
        for turn in session.get("turns", []):
            text = turn.get("text", "").strip()
            speaker = turn.get("speaker", "")
            words = text.lower().split()
            if len(words) > 5 and not any(w in SKIP_WORDS for w in words[:2]):
                mem = f"{speaker}: {text[:300]}"
                if dt:
                    mem += f" (session {sid}, {dt})"
                memories.append(mem)
    if len(memories) > max_memories:
        step = len(memories) / max_memories
        memories = [memories[int(i * step)] for i in range(max_memories)]
    return memories


def retrieve_memories(question, memories, top_k=20):
    """Retrieve top-k memories. Embedding search with keyword fallback."""
    if not memories:
        return []
    try:
        from src.retriever import embed_texts, search_numpy_fallback
        corpus_emb = embed_texts(memories)
        if corpus_emb is not None:
            query_emb = embed_texts([question])
            if query_emb is not None:
                _, indices = search_numpy_fallback(
                    query_emb[0], corpus_emb, top_k=min(top_k, len(memories)),
                )
                return [memories[i] for i in indices if i < len(memories)]
    except Exception:
        pass
    q_words = set(question.lower().split())
    scored = [(len(q_words & set(m.lower().split())), m) for m in memories]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


# ============================================================
# Answer generation (GPU) — single and batched
# ============================================================

AA_SYSTEM_PROMPT = """You are an Answer Agent for a conversational AI assistant.
You have access to a memory bank containing facts from past conversations.

Given a question and retrieved memories, you must:
1. Select the most relevant memories for answering the question.
2. Reason step-by-step using the selected memories.
3. Provide a concise, accurate answer.

Output format:
<selected_memories>
[list the memory IDs or snippets you're using]
</selected_memories>
<reasoning>
[your step-by-step reasoning]
</reasoning>
<answer>
[your final answer - be concise]
</answer>"""

AA_USER_TEMPLATE = """## Retrieved Memories (top {num_retrieved})
{memories}

## Question
{question}

## Your Response:"""


def format_aa_prompt(question, retrieved_memories):
    """Format a single AA prompt (returns chat messages list)."""
    mem_str = "\n".join(f"- {m}" for m in retrieved_memories) \
        if retrieved_memories else "No relevant memories found."
    return [
        {"role": "system", "content": AA_SYSTEM_PROMPT},
        {"role": "user", "content": AA_USER_TEMPLATE.format(
            num_retrieved=len(retrieved_memories),
            memories=mem_str, question=question,
        )},
    ]


def extract_answer(raw_output):
    """Extract answer from model output."""
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_output, re.DOTALL)
    if ans_match:
        return ans_match.group(1).strip()
    lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
    return lines[-1] if lines else raw_output.strip()


def generate_answer(model, tokenizer, question, retrieved_memories,
                    max_new_tokens=512):
    """Generate an answer for a single question (fallback for non-batched)."""
    import torch
    messages = format_aa_prompt(question, retrieved_memories)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True, top_p=0.9,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_answer(raw_output)


def generate_answers_batched(model, tokenizer, prompts_list,
                             max_new_tokens=1024, batch_size=8):
    """
    Batched answer generation. Significantly faster than one-at-a-time.
    prompts_list: list of chat message lists (one per question).
    Returns list of extracted answer strings.
    """
    import torch

    # Convert chat messages to text
    texts = []
    for messages in prompts_list:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    answers = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.3, do_sample=True, top_p=0.9,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            new_tokens = output[input_len:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answers.append(extract_answer(raw_output))

    return answers


# ============================================================
# Inference orchestration
# ============================================================

def load_test_data(test_file):
    """Load test JSONL file."""
    data = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_mm_model(model_config):
    """Load Memory Manager model for use_mm inference. Returns (model, tokenizer) or (None, None)."""
    mm_checkpoint = model_config.get("mm_checkpoint")
    if not mm_checkpoint:
        logger.warning("use_mm=true but no mm_checkpoint specified, falling back to heuristic")
        return None, None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_full_ft = model_config.get("full_ft", False)
    is_lora = model_config.get("lora", False)

    if not is_lora and not is_full_ft:
        # Auto-detect from training_meta.json in MM checkpoint dir
        meta_path = Path(mm_checkpoint) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("use_lora", False):
                is_lora = True
            elif meta.get("use_lora") is False:
                is_full_ft = True
            logger.info(f"MM auto-detected: lora={is_lora}, full_ft={is_full_ft}")

    if is_full_ft:
        logger.info(f"Loading MM full FT checkpoint: {mm_checkpoint}")
        mm_model = AutoModelForCausalLM.from_pretrained(
            mm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mm_tokenizer = AutoTokenizer.from_pretrained(mm_checkpoint)
    elif is_lora:
        from peft import PeftModel
        base_model_name = model_config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
        mm_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mm_model = PeftModel.from_pretrained(mm_model, mm_checkpoint)
        mm_model = mm_model.merge_and_unload()
        mm_tokenizer = AutoTokenizer.from_pretrained(mm_checkpoint)
    else:
        logger.info(f"Loading MM base model: {mm_checkpoint}")
        mm_model = AutoModelForCausalLM.from_pretrained(
            mm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mm_tokenizer = AutoTokenizer.from_pretrained(mm_checkpoint)

    if mm_tokenizer.pad_token is None:
        mm_tokenizer.pad_token = mm_tokenizer.eos_token
    mm_model.eval()
    return mm_model, mm_tokenizer


def run_mm_on_sessions(mm_model, mm_tokenizer, sessions, max_new_tokens=256):
    """
    Run trained Memory Manager on all dialogue turns to build a memory bank.
    Returns list of memory strings (same format as heuristic memories).
    """
    import torch
    from src.memory_bank import MemoryBank
    from src.memory_manager import build_mm_prompt, parse_mm_output, execute_mm_operation

    bank = MemoryBank(use_embeddings=False)

    for session in sessions:
        sid = session.get("session_id", 0)
        for i, turn in enumerate(session.get("turns", [])):
            messages = build_mm_prompt(
                bank, session_id=sid, turn_id=i,
                speaker=turn["speaker"], message=turn["text"][:500],
            )
            text = mm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = mm_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=4096,
            ).to(mm_model.device)

            with torch.no_grad():
                outputs = mm_model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=0.3, do_sample=True, top_p=0.9,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = mm_tokenizer.decode(new_tokens, skip_special_tokens=True)

            operation = parse_mm_output(raw_output)
            execute_mm_operation(operation, bank, session_id=sid)
            bank.advance_turn()

    # Convert bank entries to string list (same format as heuristic)
    return [e.content for e in bank.get_all()]


def run_inference(model, tokenizer, model_config, benchmark_config,
                  retrieval_top_k=20, max_examples=None, use_batched=True,
                  inference_batch_size=8, max_new_tokens=1024,
                  mm_model=None, mm_tokenizer=None):
    """
    Run inference for a loaded model on a benchmark.
    Groups questions by conversation_id.
    Supports batched generation for speed.

    If mm_model is provided (use_mm=true), uses trained MM to build memories
    instead of heuristic memory construction.
    """
    model_name = model_config["name"]
    benchmark_name = benchmark_config["name"]
    test_file = benchmark_config["test_file"]
    use_mm = model_config.get("use_mm", False) and mm_model is not None

    logger.info(f"Running {model_name} on {benchmark_name} ({test_file})")
    logger.info(f"Memory construction: {'trained MM' if use_mm else 'heuristic'}")
    test_data = load_test_data(test_file)
    if max_examples and len(test_data) > max_examples:
        test_data = test_data[:max_examples]
        logger.info(f"Capped to {max_examples} examples")
    logger.info(f"Loaded {len(test_data)} test examples")

    conv_groups = defaultdict(list)
    for ex in test_data:
        conv_groups[ex["conversation_id"]].append(ex)
    logger.info(f"Grouped into {len(conv_groups)} unique conversations")

    conv_memories = {}
    all_prompts = []
    all_metadata = []

    for conv_idx, (conv_id, examples) in enumerate(conv_groups.items()):
        if conv_id not in conv_memories:
            sessions = examples[0].get("sessions", [])
            if use_mm:
                conv_memories[conv_id] = run_mm_on_sessions(
                    mm_model, mm_tokenizer, sessions,
                )
            else:
                conv_memories[conv_id] = build_heuristic_memories(sessions)
        memories = conv_memories[conv_id]
        logger.info(
            f"  Conv {conv_idx+1}/{len(conv_groups)} ({conv_id}): "
            f"{len(memories)} memories, {len(examples)} questions"
        )
        for ex in examples:
            question = ex["question"]
            retrieved = retrieve_memories(question, memories, top_k=retrieval_top_k)
            prompt = format_aa_prompt(question, retrieved)
            all_prompts.append(prompt)
            all_metadata.append({
                "question": question,
                "gold_answer": ex["answer"],
                "question_type": ex.get("question_type", "unknown"),
                "source_benchmark": ex.get("source_benchmark", benchmark_name),
                "model": model_name,
                "conversation_id": conv_id,
                "memory_source": "trained_mm" if use_mm else "heuristic",
            })

    # Generate answers (batched or sequential)
    if use_batched and len(all_prompts) > 1:
        logger.info(f"Generating answers (batched, batch_size={inference_batch_size})...")
        answers = generate_answers_batched(
            model, tokenizer, all_prompts,
            max_new_tokens=max_new_tokens,
            batch_size=inference_batch_size,
        )
    else:
        logger.info("Generating answers (sequential)...")
        answers = []
        for i, prompt in enumerate(all_prompts):
            answer = generate_answer(model, tokenizer,
                                     all_metadata[i]["question"],
                                     [],  # memories already in prompt
                                     max_new_tokens=max_new_tokens)
            answers.append(answer)
            if (i + 1) % 50 == 0:
                logger.info(f"    Progress: {i+1}/{len(all_prompts)}")

    # Combine
    predictions = []
    for meta, answer in zip(all_metadata, answers):
        meta["answer"] = answer
        predictions.append(meta)

    logger.info(f"Completed {len(predictions)} predictions")
    return predictions


# ============================================================
# Config + Main
# ============================================================

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_comparison_table(all_results):
    """Print a cross-model F1 comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Cross-Model Comparison (F1)")
    print("=" * 80)
    benchmarks = sorted({b for r in all_results.values() for b in r})
    header = f"{'Model':<30}" + "".join(f"{b:>15}" for b in benchmarks)
    print(header)
    print("-" * len(header))
    for model_name, model_results in all_results.items():
        row = f"{model_name:<30}"
        for bench in benchmarks:
            if bench in model_results:
                row += f"{model_results[bench]['overall']['f1']:>15.3f}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-only", action="store_true",
                        help="Run judge on existing prediction files (no inference)")
    parser.add_argument("--models", type=str, nargs="*")
    parser.add_argument("--benchmarks", type=str, nargs="*")
    parser.add_argument("--retrieval-top-k", type=int, default=None,
                        help="Override retrieval top-k from config")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batched inference")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    config = load_config(args.config)
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retrieval top-k: CLI > config > default 20
    retrieval_top_k = (args.retrieval_top_k
                       or config["evaluation"].get("retrieval", {}).get("top_k", 20))
    inference_batch_size = config["evaluation"].get("hardware", {}).get("batch_size", 8)
    max_new_tokens = 1024  # Phase 2 default

    # --judge-only: read existing predictions, run judge, recompute metrics
    if args.judge_only:
        judge_cfg = config["evaluation"].get("llm_judge", {})
        model_id = judge_cfg.get("model", "gpt-4o-mini")
        all_results = {}
        for model_cfg in config["evaluation"]["models"]:
            model_name = model_cfg["name"]
            if args.models and model_name not in args.models:
                continue
            model_results = {}
            for bench_cfg in config["evaluation"]["benchmarks"]:
                bench_name = bench_cfg["name"]
                if args.benchmarks and bench_name not in args.benchmarks:
                    continue
                pred_path = output_dir / f"{model_name}_{bench_name}_predictions.jsonl"
                if not pred_path.exists():
                    logger.warning(f"No predictions found: {pred_path}, skipping")
                    continue
                predictions = []
                with open(pred_path) as f:
                    for line in f:
                        if line.strip():
                            predictions.append(json.loads(line))
                logger.info(f"Judging {len(predictions)} predictions for {model_name}/{bench_name}")
                predictions = judge_batch(predictions, model=model_id)
                results = evaluate_predictions(predictions)
                model_results[bench_name] = results
                with open(pred_path, "w") as f:
                    for p in predictions:
                        f.write(json.dumps(p) + "\n")
                print(format_results_table(results, f"{model_name} / {bench_name}"))
                print()
            all_results[model_name] = model_results
        save_results(all_results, output_dir / "all_results.json")
        logger.info(f"Judge results saved to {output_dir}")
        print_comparison_table(all_results)
        return

    all_results = {}

    for model_cfg in config["evaluation"]["models"]:
        model_name = model_cfg["name"]
        if args.models and model_name not in args.models:
            continue

        logger.info(f"Loading model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_cfg)

        # Load MM model if use_mm is set
        mm_model, mm_tokenizer = None, None
        if model_cfg.get("use_mm", False):
            logger.info(f"Loading MM model for {model_name}")
            mm_model, mm_tokenizer = load_mm_model(model_cfg)

        model_results = {}

        for bench_cfg in config["evaluation"]["benchmarks"]:
            bench_name = bench_cfg["name"]
            if args.benchmarks and bench_name not in args.benchmarks:
                continue
            test_file = bench_cfg["test_file"]
            if not Path(test_file).exists():
                logger.warning(f"Test file not found: {test_file}, skipping")
                continue

            predictions = run_inference(
                model, tokenizer, model_cfg, bench_cfg,
                retrieval_top_k=retrieval_top_k,
                max_examples=args.max_examples,
                use_batched=not args.no_batch,
                inference_batch_size=inference_batch_size,
                max_new_tokens=max_new_tokens,
                mm_model=mm_model,
                mm_tokenizer=mm_tokenizer,
            )

            if not args.skip_judge and "llm_judge" in config["evaluation"]["metrics"]:
                judge_cfg = config["evaluation"].get("llm_judge", {})
                predictions = judge_batch(
                    predictions,
                    model=judge_cfg.get(
                        "model", "gpt-4o-mini"),
                )

            results = evaluate_predictions(predictions)
            model_results[bench_name] = results

            pred_path = output_dir / f"{model_name}_{bench_name}_predictions.jsonl"
            with open(pred_path, "w") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")
            print(format_results_table(results, f"{model_name} / {bench_name}"))
            print()

        all_results[model_name] = model_results

        del model, tokenizer
        if mm_model is not None:
            del mm_model, mm_tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    save_results(all_results, output_dir / "all_results.json")
    logger.info(f"All results saved to {output_dir}")
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
