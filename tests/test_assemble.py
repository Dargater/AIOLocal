from __future__ import annotations

import pytest

from aiolocal.config import Settings
from aiolocal.memory.assemble import InputTooLarge, assemble
from aiolocal.memory.retrieve import MemoryItem
from aiolocal.memory.text import estimate_tokens


def item(kind: str, idx: int, chars: int = 600) -> MemoryItem:
    body = f"item {idx} " + ("x" * chars)
    return MemoryItem(
        kind=kind,
        id=f"item{idx}",
        title=f"Item {idx}",
        body=body,
        score=1.0 - idx * 0.01,
        created_at=1000 + idx,
        tokens=estimate_tokens(body),
    )


def test_assemble_respects_context_budget():
    settings = Settings(context_tokens=2000, max_output_tokens=200)
    items = [item("summary", i) for i in range(10)] + [item("exact", i + 100) for i in range(10)]
    messages, plan = assemble(
        settings,
        user_text="What did we decide?",
        history=[{"role": "user", "content": "older question"}, {"role": "assistant", "content": "older answer"}],
        items=items,
    )
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "What did we decide?"}
    total = estimate_tokens(messages[0]["content"]) + plan["input_tokens"] + plan["tool_tokens"]
    assert total <= settings.context_tokens - settings.max_output_tokens
    assert plan["trims"], "expected the planner to trim something"


def test_current_message_wins_and_oversize_is_rejected():
    settings = Settings(context_tokens=1024, max_output_tokens=200)
    huge = "word " * 5000
    with pytest.raises(InputTooLarge):
        assemble(settings, user_text=huge)


def test_tool_chain_is_preserved():
    settings = Settings(context_tokens=4096, max_output_tokens=256)
    tool_chain = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "tool result", "tool_call_id": "1"},
    ]
    messages, plan = assemble(settings, user_text="continue", tool_chain=tool_chain)
    assert messages[-2:] == tool_chain
    assert plan["tool_tokens"] > 0


def test_memory_hint_can_be_disabled():
    settings = Settings(context_tokens=4096, max_output_tokens=256, memory_hint=False)
    messages, _ = assemble(settings, user_text="hello", items=[])
    assert "nothing relevant was found" not in messages[0]["content"]


def test_medical_mode_has_disclaimer():
    settings = Settings(context_tokens=4096, max_output_tokens=256)
    messages, plan = assemble(settings, user_text="my chest hurts", mode="medical")
    assert plan["mode"] == "medical"
    assert "not a doctor" in messages[0]["content"]


def test_no_overflow_with_large_history():
    settings = Settings(context_tokens=1500, max_output_tokens=200, short_history_turns=20)
    history = [{"role": "assistant", "content": "answer " * 200} for _ in range(20)]
    items = [item("summary", i, chars=1000) for i in range(5)]
    messages, plan = assemble(settings, user_text="continue the plan", history=history, items=items)
    total = estimate_tokens(messages[0]["content"]) + plan["input_tokens"]
    assert total <= settings.context_tokens - settings.max_output_tokens
