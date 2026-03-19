"""
Answer Agent for Memory-R1.

Given a question and retrieved memories, the Answer Agent:
1. Pre-selects the most relevant memories (distillation from 60 → ~5)
2. Reasons over selected memories to generate an answer

During training, fine-tuned with GRPO where reward = F1(answer, gold_answer).
"""

# ---- Prompt Templates ----

ANSWER_AGENT_SYSTEM = """You are an Answer Agent for a conversational AI assistant.
You have access to a memory bank containing facts from past conversations.

Given a question and retrieved memories, you must:
1. Select the most relevant memories for answering the question.
2. Reason step-by-step using the selected memories.
3. Provide a concise, accurate answer.

Output format:
<selected_memories>
[list the entry IDs you're using, e.g., "a1b2c3, d4e5f6"]
</selected_memories>
<reasoning>
[your step-by-step reasoning]
</reasoning>
<answer>
[your final answer]
</answer>"""

ANSWER_AGENT_USER_TEMPLATE = """## Retrieved Memories (top {num_retrieved})
{memories}

## Question
{question}

## Your Response:"""


import re
from src.memory_bank import MemoryBank, MemoryEntry


def build_aa_prompt(question: str, retrieved_memories: list[MemoryEntry]) -> list[dict]:
    """Build the prompt for the Answer Agent."""
    if retrieved_memories:
        mem_lines = []
        for e in retrieved_memories:
            meta = f"(session {e.source_session}"
            if e.timestamp:
                meta += f", {e.timestamp}"
            meta += ")"
            mem_lines.append(f"- [{e.entry_id}] {e.content} {meta}")
        memories_str = "\n".join(mem_lines)
    else:
        memories_str = "No relevant memories found."

    user_msg = ANSWER_AGENT_USER_TEMPLATE.format(
        num_retrieved=len(retrieved_memories),
        memories=memories_str,
        question=question,
    )
    return [
        {"role": "system", "content": ANSWER_AGENT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def parse_aa_output(raw_output: str) -> dict:
    """Parse Answer Agent output into structured components."""
    result = {
        "selected_memories": [],
        "reasoning": "",
        "answer": "",
    }

    # Extract selected memories
    sel_match = re.search(
        r"<selected_memories>\s*(.*?)\s*</selected_memories>",
        raw_output, re.DOTALL
    )
    if sel_match:
        ids_str = sel_match.group(1).strip()
        result["selected_memories"] = [
            s.strip() for s in ids_str.split(",") if s.strip()
        ]

    # Extract reasoning
    reason_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>",
        raw_output, re.DOTALL
    )
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    # Extract answer
    ans_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        raw_output, re.DOTALL
    )
    if ans_match:
        result["answer"] = ans_match.group(1).strip()
    else:
        # Fallback: use the last non-empty line
        lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
        if lines:
            result["answer"] = lines[-1]

    return result
