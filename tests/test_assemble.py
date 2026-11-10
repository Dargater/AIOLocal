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


def test_document_summary_is_guaranteed_and_not_crowded_out():
    settings = Settings(context_tokens=8192, max_output_tokens=1024)
    document_body = "vector question " + ("detail " * 3200)
    document = MemoryItem(
        kind="summary",
        id="doc1",
        title="Document: Vectors",
        body=document_body,
        score=0.9,
        created_at=5000,
        tokens=estimate_tokens(document_body),
        summary_kind="document",
    )
    others = [item("summary", i, chars=500) for i in range(10)]
    exact = [item("exact", 100 + i, chars=1800) for i in range(4)]
    messages, plan = assemble(
        settings,
        user_text="solve the vector questions from the PDF",
        history=[{"role": "assistant", "content": "earlier answer"}],
        items=exact + [document] + others,
    )
    system = messages[0]["content"]
    assert "Document: Vectors" in system
    assert "detail detail detail" in system
    assert "DOCUMENT MEMORY" in system
    assert document.id in plan["document_ids"]
    assert plan["document_tokens"] > 1000
    assert "document" not in plan["trims"]
    assert len(system) > len(document_body)


def test_oversized_document_is_truncated_not_dropped():
    settings = Settings(context_tokens=4096, max_output_tokens=512)
    huge_body = "very long document " * 9000
    document = MemoryItem(
        kind="summary",
        id="doc2",
        title="Document: Huge",
        body=huge_body,
        score=1.0,
        created_at=9000,
        tokens=estimate_tokens(huge_body),
        summary_kind="document",
    )
    messages, plan = assemble(settings, user_text="summarise the document", items=[document])
    assert "Document: Huge" in messages[0]["content"]
    assert "very long document" in messages[0]["content"]
    assert document.id in plan["document_ids"]
    assert len(messages[0]["content"]) < len(huge_body)


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
