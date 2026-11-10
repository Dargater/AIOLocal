"""Prompt assembly and the token-budget planner.

The planner is the single place that decides what fits in the context window. The
current user message always wins: if it cannot fit, the gateway answers with a clear
413 instead of silently truncating the question.

Trim order (from the design): exact excerpts first, then summaries, then recent
history. Inside exact excerpts the oldest group goes first; inside summaries the
lowest score goes first; inside history the oldest turn goes first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .retrieve import MemoryItem
from .text import estimate_tokens, truncate_tokens

from ..config import Settings

MEDICAL_DISCLAIMER = (
    "MEDICAL SAFETY: This assistant is not a doctor and cannot diagnose, prescribe, or "
    "replace professional medical care. Provide general information only, clearly label "
    "uncertainty, and tell the user to contact a qualified clinician for personal medical "
    "decisions. In an emergency, tell them to contact local emergency services."
)

MODE_INSTRUCTIONS = {
    "chat": "Be direct, accurate and useful. Prefer concrete answers over filler.",
    "code": (
        "You are helping with software. Read the existing code before changing it, keep "
        "changes minimal and consistent, and explain risky operations before running them."
    ),
    "document": (
        "You are working with the user's documents. Quote or cite the provided excerpts, "
        "and say clearly when the documents do not contain the answer."
    ),
    "re": (
        "You are assisting with reverse engineering of software the user is authorised to "
        "analyse. Stay factual and avoid speculation about intent."
    ),
    "medical": (
        "You are providing general health information. Never claim to be a clinician. "
        "Never give a personal diagnosis or a prescription dose."
    ),
}

BASE_SYSTEM = (
    "You are AIO Local, a private AI assistant running entirely on the user's computer. "
    "You have no internet access unless the user explicitly enables a tool. "
    "Your answers are read by one person, so be clear and concrete.\n"
    "Speak in the first person at all times: say \"I\", not \"AIO Local\" or \"the assistant\". "
    "If you refer to the app, say \"this app\" or \"I\". Never describe yourself from the "
    "outside or invent a different identity.\n"
)


class InputTooLarge(Exception):
    """The current user message alone does not fit in the context window."""

    def __init__(self, needed: int, available: int) -> None:
        super().__init__(
            f"current message needs about {needed} tokens but only {available} are available"
        )
        self.needed = needed
        self.available = available


@dataclass
class Budget:
    context: int
    output: int
    system: int
    available: int
    input: int
    tools: int


def _render_system(
    settings: Settings,
    items: list[MemoryItem],
    history: list[dict],
    mode: str,
) -> str:
    parts = [BASE_SYSTEM, MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["chat"])]
    parts.append(
        f"\nYour runtime model is {settings.chat_model}, running locally through OpenVINO. "
        "If the user asks who you are or what model you are, answer in the first person: "
        f"\"I am AIO Local, powered by {settings.chat_model}, running on your computer.\" "
        "Do not say \"AIO Local is\" or \"the assistant is\". Do not describe yourself as a "
        "cloud service or claim a different model.\n"
    )
    if mode == "medical":
        parts.append("\n" + MEDICAL_DISCLAIMER)

    if items:
        has_document = any(i.summary_kind == "document" for i in items)
        if has_document:
            parts.append(
                "\n\nDOCUMENT MEMORY: The document notes below were produced by reading the "
                "attached PDF page by page with the vision model. Treat them as the document "
                "contents. Answer from them and do not claim you cannot see or read the "
                "document when its notes are present.\n"
            )
        parts.append("\n\n## What you remember (retrieved)\n")
        for item in items:
            if item.kind == "summary":
                tag = f"  [{item.topic}]" if item.topic else ""
                parts.append(f"### Summary — {item.title}{tag}\n{item.body}\n")
            else:
                ids = ", ".join(item.message_ids[:6])
                parts.append(f"### {item.title}  (messages {ids})\n{item.body}\n")
    elif settings.memory_hint:
        parts.append("\n\n## What you remember (retrieved)\n(nothing relevant was found)\n")

    if history:
        parts.append("\n## Recent conversation (same chat)\n")
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            parts.append(f"{role}: {msg.get('content', '')}\n")

    parts.append(
        "\n## Rules\n"
        "- The current user message overrides anything remembered.\n"
        "- If memory conflicts with the current message, follow the current message.\n"
        "- Do not invent memories. If the retrieved memory is empty, say you do not know.\n"
        "- Cite memory ids only when the user asks what you remember.\n"
    )
    return "".join(parts)


def _trim_history(history: list[dict], budget: int) -> tuple[list[dict], int]:
    kept: list[dict] = []
    used = 0
    for msg in reversed(history):
        cost = estimate_tokens(msg.get("content", ""))
        if kept and used + cost > budget:
            break
        if not kept and cost > budget:
            text = truncate_tokens(msg.get("content", ""), max(1, budget))
            kept.append({**msg, "content": text})
            used += estimate_tokens(text)
            break
        kept.append(msg)
        used += cost
    kept.reverse()
    return kept, used


def _trim_items(
    items: list[MemoryItem],
    budget: int,
    newest_first: bool,
    *,
    keep_top: bool = False,
) -> tuple[list[MemoryItem], int]:
    ordered = sorted(
        items,
        key=(lambda i: i.created_at) if newest_first else (lambda i: i.score),
        reverse=True,
    )
    kept: list[MemoryItem] = []
    used = 0
    remaining = ordered
    if keep_top and ordered:
        # The highest-ranked item must survive. If it is larger than the whole
        # budget, truncate it rather than skipping it and answering without the
        # one memory the query actually matched (usually a document summary).
        item = ordered[0]
        cost = item.tokens or estimate_tokens(item.body)
        if cost > budget:
            allowed = max(1, budget)
            text = truncate_tokens(item.body, max(1, allowed - estimate_tokens(item.title)))
            item = _clone_item(item, text)
            cost = item.tokens
        if cost <= budget:
            kept.append(item)
            used = cost
        remaining = ordered[1:]
    for item in remaining:
        cost = item.tokens or estimate_tokens(item.body)
        if used + cost > budget:
            continue
        kept.append(item)
        used += cost
    if not kept and ordered:
        item = ordered[0]
        allowed = max(1, budget)
        text = truncate_tokens(item.body, max(1, allowed - estimate_tokens(item.title)))
        clone = _clone_item(item, text)
        kept = [clone]
        used = clone.tokens
    return kept, used


def _clone_item(item: MemoryItem, text: str) -> MemoryItem:
    return MemoryItem(
        kind=item.kind,
        id=item.id,
        title=item.title,
        body=text,
        score=item.score,
        topic=item.topic,
        message_ids=item.message_ids,
        created_at=item.created_at,
        salience=item.salience,
        tokens=estimate_tokens(text),
        summary_kind=item.summary_kind,
    )


def assemble(
    settings: Settings,
    *,
    user_text: str,
    tool_chain: list[dict] | None = None,
    history: list[dict] | None = None,
    items: list[MemoryItem] | None = None,
    extra_context: str = "",
    mode: str = "chat",
    request_id: str | None = None,
) -> tuple[list[dict], dict]:
    tool_chain = list(tool_chain or [])
    history = list(history or [])
    items = list(items or [])

    output_tokens = max(16, settings.max_output_tokens or settings.output_reserve_tokens)
    base_system = _render_system(settings, [], [], mode) + (extra_context or "")
    system_tokens = estimate_tokens(base_system) + 64  # headers + safety margin
    available = settings.context_tokens - output_tokens - system_tokens
    if available < 256:
        raise InputTooLarge(estimate_tokens(user_text), max(0, available))

    input_tokens = estimate_tokens(user_text)
    tool_tokens = sum(estimate_tokens(m.get("content", "") or "") for m in tool_chain)
    if input_tokens + tool_tokens > available:
        raise InputTooLarge(input_tokens + tool_tokens, available)

    remaining = available - input_tokens - tool_tokens

    # Allocation order: history (25%), summaries (35%), exact (remainder).
    history_budget = min(int(available * 0.25), max(0, remaining))
    chosen_history, history_used = _trim_history(history, history_budget)
    remaining -= history_used

    summaries = [i for i in items if i.kind == "summary"]
    exact = [i for i in items if i.kind == "exact"]
    documents = [i for i in summaries if i.summary_kind == "document"]
    other_summaries = [i for i in summaries if i.summary_kind != "document"]

    # Document summaries are the complete analysis of a PDF the user attached.
    # They are large and must not be crowded out by incidental exact excerpts, so
    # they get their own guaranteed slice before the ordinary summaries.
    chosen_documents: list[MemoryItem] = []
    document_used = 0
    if documents:
        document_budget = min(int(available * 0.55), max(0, remaining))
        chosen_documents, document_used = _trim_items(
            documents, document_budget, newest_first=False, keep_top=True
        )
        remaining -= document_used

    summary_budget = min(int(available * 0.35), max(0, remaining))
    chosen_summaries, summary_used = _trim_items(
        other_summaries, summary_budget, newest_first=False, keep_top=True
    )
    remaining -= summary_used

    chosen_exact, exact_used = _trim_items(exact, max(0, remaining), newest_first=True)

    # Final safety pass: render and trim in the documented order if still over.
    trims: list[str] = []
    if len(chosen_history) < len(history):
        trims.append("history")
    if len(chosen_documents) < len(documents):
        trims.append("document")
    if len(chosen_summaries) < len(other_summaries):
        trims.append("summary")
    if len(chosen_exact) < len(exact):
        trims.append("exact")
    chosen_memory = chosen_documents + chosen_summaries + chosen_exact
    for _ in range(24):
        rendered = _render_system(settings, chosen_memory, chosen_history, mode)
        total = estimate_tokens(rendered) + input_tokens + tool_tokens
        if total <= settings.context_tokens - output_tokens:
            break
        if chosen_exact:
            chosen_exact.pop()
            chosen_memory = chosen_documents + chosen_summaries + chosen_exact
            trims.append("exact")
        elif chosen_summaries:
            chosen_summaries.sort(key=lambda i: i.score)
            chosen_summaries.pop(0)
            chosen_memory = chosen_documents + chosen_summaries + chosen_exact
            trims.append("summary")
        elif chosen_documents:
            # Truncate the document note as a last resort instead of dropping it.
            document = chosen_documents[0]
            shorter = truncate_tokens(document.body, max(1, document.tokens // 2))
            chosen_documents = [_clone_item(document, shorter)]
            chosen_memory = chosen_documents + chosen_summaries + chosen_exact
            trims.append("document")
        elif chosen_history:
            chosen_history.pop(0)
            trims.append("history")
        else:
            raise InputTooLarge(input_tokens + tool_tokens, available)

    rendered = _render_system(settings, chosen_memory, chosen_history, mode)

    messages: list[dict] = [{"role": "system", "content": rendered}]
    messages.append({"role": "user", "content": user_text})
    messages.extend(tool_chain)

    plan = {
        "request_id": request_id,
        "mode": mode,
        "context_tokens": settings.context_tokens,
        "output_reserve": output_tokens,
        "available": available,
        "input_tokens": input_tokens,
        "tool_tokens": tool_tokens,
        "history_tokens": history_used,
        "document_tokens": document_used,
        "summary_tokens": summary_used,
        "exact_tokens": exact_used,
        "system_tokens": estimate_tokens(rendered),
        "history_count": len(chosen_history),
        "document_ids": [i.id for i in chosen_documents],
        "summary_ids": [i.id for i in chosen_summaries],
        "exact_ids": [i.id for i in chosen_exact],
        "trims": trims,
        "items": [i.to_dict() for i in chosen_memory],
    }
    return messages, plan
