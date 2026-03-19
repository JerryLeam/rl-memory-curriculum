"""
Memory Manager agent for Memory-R1.

Processes each dialogue turn and decides which CRUD operation to perform
on the memory bank: ADD, UPDATE, DELETE, or NOOP.

During training, this agent is fine-tuned with GRPO to learn optimal
memory management policies. During inference, it generates structured
JSON operations.
"""
from typing import Optional

# ---- Prompt Templates ----

MEMORY_MANAGER_SYSTEM = """You are a Memory Manager for a conversational AI assistant.
Your job is to maintain an external memory bank by deciding what information to store,
update, or remove after each dialogue turn.

Given the current dialogue turn and existing memories, output a JSON operation:

Operations:
- ADD: Store new important information. Output: {"op": "ADD", "content": "<fact to store>"}
- UPDATE: Modify an existing memory with new info. Output: {"op": "UPDATE", "entry_id": "<id>", "content": "<updated fact>"}
- DELETE: Remove outdated/incorrect memory. Output: {"op": "DELETE", "entry_id": "<id>"}
- NOOP: No memory change needed. Output: {"op": "NOOP"}

Rules:
1. Only ADD facts that are important for future conversations (preferences, events, relationships).
2. UPDATE when the user corrects or changes a previously stated fact.
3. DELETE when information is explicitly retracted or clearly wrong.
4. NOOP when the turn is casual/greeting or contains no memorable information.
5. Output exactly ONE operation per turn as valid JSON."""

MEMORY_MANAGER_USER_TEMPLATE = """## Current Memories
{memories}

## Current Dialogue Turn
Session {session_id}, Turn {turn_id}:
Speaker: {speaker}
Message: {message}

## Your Decision (output valid JSON):"""


import json
import re
from src.memory_bank import MemoryBank


def build_mm_prompt(memory_bank: MemoryBank, session_id: int,
                    turn_id: int, speaker: str, message: str) -> list[dict]:
    """Build the prompt for the Memory Manager."""
    memories_str = memory_bank.format_for_prompt()
    user_msg = MEMORY_MANAGER_USER_TEMPLATE.format(
        memories=memories_str,
        session_id=session_id,
        turn_id=turn_id,
        speaker=speaker,
        message=message,
    )
    return [
        {"role": "system", "content": MEMORY_MANAGER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def parse_mm_output(raw_output: str) -> dict:
    """
    Parse Memory Manager output into a structured operation.
    Handles common LLM output quirks (extra text, markdown fences).
    """
    # Try to extract JSON from the output
    # Strip markdown code fences if present
    cleaned = re.sub(r"```json?\s*", "", raw_output)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # Find the first JSON object
    match = re.search(r"\{[^}]+\}", cleaned)
    if not match:
        return {"op": "NOOP"}

    try:
        parsed = json.loads(match.group())
        op = parsed.get("op", "NOOP").upper()
        if op not in ("ADD", "UPDATE", "DELETE", "NOOP"):
            return {"op": "NOOP"}
        return parsed
    except json.JSONDecodeError:
        return {"op": "NOOP"}


def execute_mm_operation(operation: dict, memory_bank: MemoryBank,
                         session_id: int) -> str:
    """Execute a parsed Memory Manager operation on the memory bank."""
    op = operation.get("op", "NOOP").upper()

    if op == "ADD":
        content = operation.get("content", "")
        if content:
            entry_id = memory_bank.add(content, source_session=session_id)
            return f"ADD: created {entry_id}"
        return "ADD: skipped (empty content)"

    elif op == "UPDATE":
        entry_id = operation.get("entry_id", "")
        content = operation.get("content", "")
        if entry_id and content:
            success = memory_bank.update(entry_id, content)
            return f"UPDATE: {'ok' if success else 'failed (id not found)'}"
        return "UPDATE: skipped (missing id or content)"

    elif op == "DELETE":
        entry_id = operation.get("entry_id", "")
        if entry_id:
            success = memory_bank.delete(entry_id)
            return f"DELETE: {'ok' if success else 'failed (id not found)'}"
        return "DELETE: skipped (missing id)"

    else:  # NOOP
        memory_bank.noop()
        return "NOOP"
