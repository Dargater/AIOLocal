"""OpenAI-compatible gateway with the memory pipeline in the middle.

Contract with clients: send the current turn; the gateway rebuilds history from its
own append-only database. Client-supplied history is ignored except for the current
user message and an in-progress tool chain. This is what keeps memory consistent and
stops clients from injecting their own RAG blocks.
"""

from __future__ import annotations

import json
import re
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .context import AppContext
from .llm import UpstreamError
from .memory import store, semantic
from .memory.assemble import InputTooLarge, assemble
from .memory.retrieve import retrieve
from .memory.text import content_hash, new_id, normalise, truncate_tokens
from .security import valid_id

router = APIRouter()

_SIDE_HINTS = (
    "generate a title",
    "create a title",
    "suggest a title",
    "summarize the chat",
    "summarise this chat",
    "summarize this conversation",
    "3-5 words",
    "3 to 5 words",
    "short title",
    "tags for this chat",
)


def _error(status: int, message: str, code: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": kind, "code": code, "param": None}},
    )


def _apply_generation_defaults(ctx: AppContext, payload: dict) -> dict:
    """Apply local-model defaults without overriding explicit client choices.

    Qwen 3.5 enables a long reasoning trace by default. On an iGPU that turns a
    one-word greeting into a minute-long wait, so the app defaults thinking off
    and exposes it as a setting. Clients can still opt in per request.
    """
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    kwargs.setdefault("enable_thinking", bool(ctx.settings.enable_thinking))
    payload["chat_template_kwargs"] = kwargs
    return payload


def _last_user(messages: list[dict]) -> tuple[str, int]:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                # Some clients send content parts; keep the text parts only.
                content = " ".join(
                    str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            return str(content or ""), idx
    return "", -1


def _image_parts(messages: list[dict], index: int) -> list[dict]:
    """Extract OpenAI-style image parts from one message, keeping them out of memory.

    The archive only ever stores the text of a turn. Images are forwarded for the
    current request so a vision model can see them, then forgotten.
    """
    if not (0 <= index < len(messages)):
        return []
    content = messages[index].get("content")
    if not isinstance(content, list):
        return []
    parts: list[dict] = []
    total = 0
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind not in ("image_url", "input_image"):
            continue
        image = part.get("image_url", part.get("input_image"))
        if isinstance(image, dict):
            url = str(image.get("url") or image.get("image_url") or "")
        else:
            url = str(image or "")
        if not url:
            continue
        if not (url.startswith("data:image/") or url.startswith("http://") or url.startswith("https://")):
            continue
        # Keep a single turn's image payload bounded before it reaches OVMS.
        total += len(url)
        if total > 20_000_000:
            break
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _first_user(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            return str(content or "")
    return ""


def _build_extra_context(kb_items: list[dict], web_results: list[dict]) -> str:
    parts: list[str] = []
    if kb_items:
        lines = [f"Reference {i+1} (collection excerpt, not conversation memory): {item.get('title') or 'Untitled'}\n{item.get('text', '')[:2000]}" for i, item in enumerate(kb_items)]
        parts.append("## Knowledge base references (opt-in via X-AIO-Knowledge)\n" + "\n---\n".join(lines))
    if web_results:
        lines = [f"Web result {i+1} (untrusted — verify before relying on it):\nTitle: {r.get('title', '')}\nURL: {r.get('url', '')}\nSnippet: {r.get('snippet', '')}" for i, r in enumerate(web_results)]
        parts.append("## Web search results (opt-in via X-AIO-Web; links never followed automatically)\n" + "\n---\n".join(lines))
    if parts:
        return "\n\n".join(parts) + "\n\nNote: the above is injected for this message only; it is not saved into your conversation memory."
    return ""


def classify(body: dict, headers: dict, messages: list[dict]) -> str:
    if headers.get("x-aio-side") == "1":
        return "side"
    if not messages:
        return "probe"
    if headers.get("x-aio-mode") == "document":
        return "doc"

    last_role = str(messages[-1].get("role") or "")
    if last_role == "tool" or (last_role == "assistant" and messages[-1].get("tool_calls")):
        return "tool_loop"

    # Do not treat a small max_tokens as a side request. Some clients (including
    # OpenCode subagents) legitimately ask for short answers, and classifying those
    # as "side" skipped memory for real conversations. Side work must opt in with
    # the X-AIO-Side header or match a known probe hint below.
    haystack = " ".join(
        str(m.get("content") or "") for m in messages if m.get("role") in ("system", "user")
    ).lower()[:4000]
    if any(hint in haystack for hint in _SIDE_HINTS) and len(str(messages[-1].get("content") or "")) < 500:
        return "side"
    return "chat"


def resolve_conversation(
    ctx: AppContext, body: dict, headers: dict, *, source: str = "api"
) -> str:
    header = headers.get("x-aio-conversation")
    if header and valid_id(header):
        store.ensure_conversation(ctx.db, header, source=source)
        return header

    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    chat_id = body.get("chat_id") or metadata.get("chat_id")
    if chat_id is not None and valid_id(str(chat_id)):
        chat_id = str(chat_id)
        found = store.find_conversation_by_external(ctx.db, source, chat_id)
        if found:
            store.touch_conversation(ctx.db, found)
            return found
        cid = new_id()
        store.ensure_conversation(ctx.db, cid, source=source, external_id=chat_id)
        return cid

    messages = body.get("messages") or []
    first = normalise(_first_user(messages)).lower()
    model = str(body.get("model") or "")
    fingerprint = "fp_" + content_hash(f"{source}\x00{model}\x00{first}")[:40]
    found = store.find_conversation_by_external(ctx.db, source, fingerprint)
    if found:
        return found
    cid = new_id()
    store.ensure_conversation(ctx.db, cid, source=source, external_id=fingerprint)
    return cid


def _tool_chain(messages: list[dict], last_user_index: int) -> list[dict]:
    chain: list[dict] = []
    for msg in messages[last_user_index + 1 :]:
        role = msg.get("role")
        if role not in ("assistant", "tool"):
            continue
        clean: dict = {"role": role}
        if role == "assistant":
            clean["content"] = msg.get("content") or ""
            if msg.get("tool_calls"):
                clean["tool_calls"] = msg["tool_calls"]
        else:
            clean["content"] = msg.get("content") or ""
            if msg.get("tool_call_id"):
                clean["tool_call_id"] = msg["tool_call_id"]
        chain.append(clean)
    return chain


def _accumulate_sse(line: str, state: dict) -> None:
    if not line.startswith("data:"):
        return
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return
    choices = obj.get("choices") or []
    if not choices:
        return
    choice = choices[0]
    delta = choice.get("delta") or {}
    if delta.get("content"):
        state["content"].append(str(delta["content"]))
    if delta.get("reasoning_content"):
        state["reasoning"].append(str(delta["reasoning_content"]))
    for tc in delta.get("tool_calls") or []:
        index = int(tc.get("index", 0))
        slot = state["tool_calls"].setdefault(
            index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += str(fn["name"])
        if fn.get("arguments"):
            slot["function"]["arguments"] += str(fn["arguments"])
    if choice.get("finish_reason"):
        state["finish_reason"] = choice["finish_reason"]


def _archive_assistant(
    ctx: AppContext,
    *,
    conversation_id: str,
    content: str,
    tool_calls: list[dict],
    request_id: str,
    model: str,
) -> tuple[str, bool]:
    meta = {"model": model}
    if tool_calls:
        meta["tool_calls"] = tool_calls
    message_id, inserted = store.archive_message(
        ctx.db,
        conversation_id=conversation_id,
        role="assistant",
        content=content,
        source="api",
        request_id=request_id,
        meta=meta,
        embed_dim=ctx.settings.embed_dim,
    )
    if inserted:
        try:
            semantic.index_message(ctx.db, ctx.settings, message_id, content)
        except Exception:
            pass
    return message_id, inserted


def _enqueue_filing(
    ctx: AppContext,
    *,
    conversation_id: str,
    user_message_id: str,
    assistant_message_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    if ctx.worker is None:
        return
    ctx.worker.enqueue_file_turn(
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        user_text=user_text,
        assistant_text=assistant_text,
    )


async def _prepare(
    ctx: AppContext, body: dict, request: Request
) -> tuple[dict, str, str, str, str, list[dict], dict] | JSONResponse:
    """Validate, archive the user turn, retrieve memory, and build the prompt."""
    if not isinstance(body, dict):
        return _error(400, "request body must be a JSON object", "invalid_body")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return _error(400, "messages must be a non-empty array", "invalid_messages")

    classification = classify(body, {k.lower(): v for k, v in request.headers.items()}, messages)
    if classification == "side":
        return _error(400, "side request reached the memory pipeline", "internal_error")

    user_text, last_user_index = _last_user(messages)
    image_parts = _image_parts(messages, last_user_index)
    if image_parts:
        # Refuse early instead of silently dropping the image inside a text model.
        # Only catalog models are checked: an external OpenAI-compatible server may
        # have its own vision model that this app does not know about.
        from .models import catalog_entry

        requested = str(body.get("model") or ctx.settings.chat_model or "")
        entry = catalog_entry(requested)
        if entry is not None and not entry.get("vision"):
            return _error(
                400,
                f"{entry['name']} is a text-only model. Choose a vision model such as "
                "Qwen3.5 9B or Qwen3-VL 4B in Models, then attach the image again.",
                "model_not_vision",
            )
    if not user_text.strip() and image_parts:
        user_text = "Describe this image."
    if not user_text.strip():
        return _error(400, "the last user message is empty", "empty_message")
    if len(user_text) > 200_000:
        return _error(413, "message is too large (200k characters max)", "message_too_large")

    conversation_id = resolve_conversation(ctx, body, request.headers)
    request_id = new_id()

    history_rows = store.recent_messages(
        ctx.db, conversation_id, limit=max(2, ctx.settings.short_history_turns * 2)
    )

    user_msg_id, _ = store.archive_message(
        ctx.db,
        conversation_id=conversation_id,
        role="user",
        content=user_text,
        source="api",
        client_msg_id=str(body.get("client_msg_id") or "") or None,
        request_id=request_id,
        embed_dim=ctx.settings.embed_dim,
    )
    try:
        semantic.index_message(ctx.db, ctx.settings, user_msg_id, user_text)
    except Exception:
        pass

    # Archive new tool results so the memory sees the whole turn, but never re-archive
    # the assistant tool_calls the gateway itself produced earlier (dedupe handles it).
    tool_chain = _tool_chain(messages, last_user_index)
    for msg in tool_chain:
        if msg["role"] == "tool":
            store.archive_message(
                ctx.db,
                conversation_id=conversation_id,
                role="tool",
                content=msg.get("content") or "",
                source="api",
                request_id=request_id,
                injected=True,
                embed_dim=ctx.settings.embed_dim,
            )

    exclude = {user_msg_id}
    chain_sigs = {(m["role"], content_hash(m.get("content") or "")) for m in tool_chain}
    history = [
        dict(r)
        for r in history_rows
        if r["id"] != user_msg_id
        and (r["role"], content_hash(r["content"])) not in chain_sigs
    ]

    # Opt-in knowledge base retrieval (separate from conversation memory)
    kb_items: list[dict] = []
    kb_plan_extra: dict = {}
    headers = {k.lower(): v for k, v in request.headers.items()}
    kb_header = headers.get("x-aio-knowledge")
    if kb_header and ctx.kb is not None and ctx.settings.knowledge_enabled:
        slugs = [s.strip() for s in str(kb_header).split(",") if s.strip()]
        # Only proceed if collection slugs are valid; never auto-include everything.
        if slugs:
            try:
                kb_items_raw = ctx.kb.search(
                    user_text, collections=slugs, limit=ctx.settings.knowledge_top_k,
                    embed_dim=ctx.settings.embed_dim
                )
                # Never archive KB items into memory; inject only for this turn.
                kb_items = kb_items_raw
                kb_plan_extra = {"knowledge_items": len(kb_items), "collections": slugs}
            except Exception:
                pass

    # Opt-in web search (per-message, never followed automatically)
    web_results: list[dict] = []
    web_plan_extra: dict = {}
    web_header = headers.get("x-aio-web")
    if web_header == "1" and ctx.web_search is not None and ctx.settings.web_search_enabled:
        try:
            web_results = ctx.web_search.search(user_text, limit=ctx.settings.web_search_max_results)
            web_plan_extra = {"web_search": len(web_results)}
        except Exception as exc:
            web_plan_extra = {"web_search": 0, "web_search_error": str(exc)[:400]}

    result = retrieve(
        ctx.db,
        user_text,
        settings=ctx.settings,
        conversation_id=conversation_id,
        exclude_message_ids=exclude,
        request_id=request_id,
        force_lookup=(headers.get("x-aio-memory-lookup") == "1"),
    )

    mode = str(request.headers.get("x-aio-mode") or "chat").lower()
    if mode not in ("chat", "code", "document", "re", "medical"):
        mode = "chat"

    try:
        prompt_messages, plan = assemble(
            ctx.settings,
            user_text=user_text,
            tool_chain=tool_chain,
            history=history,
            items=result.items if ctx.settings.memory_enabled else [],
            mode=mode,
            request_id=request_id,
            extra_context=_build_extra_context(kb_items, web_results),
        )
    except InputTooLarge as exc:
        return _error(
            413,
            f"Your message is too large for this model's context window ({exc.needed} tokens "
            f"needed, {exc.available} available). Increase the context setting or shorten it.",
            "context_overflow",
        )

    plan["retrieval"] = result.plan
    plan["conversation_id"] = conversation_id
    plan["user_message_id"] = user_msg_id
    if image_parts:
        # Images are forwarded to the model but never written to the memory DB.
        plan["image_parts"] = len(image_parts)
        prompt_messages = [dict(m) for m in prompt_messages]
        for message in prompt_messages:
            if message.get("role") == "user":
                message["content"] = [{"type": "text", "text": user_text}, *image_parts]
                break
    plan.update(kb_plan_extra)
    plan.update(web_plan_extra)

    # Record that KB/web results were injected but do NOT archive them into exact memory.
    if kb_items or web_results or kb_plan_extra or web_plan_extra:
        try:
            store.attach_retrieval_extra(ctx.db, request_id, "extra_injected", {
                "knowledge": bool(kb_items),
                "web_search": bool(web_results),
            })
        except Exception:
            pass

    try:
        store.attach_retrieval_extra(ctx.db, request_id, "assembly", plan)
    except Exception:  # pragma: no cover - inspection must never block a reply
        pass

    payload = dict(body)
    payload["messages"] = prompt_messages
    payload.pop("chat_id", None)
    payload.pop("client_msg_id", None)
    if not payload.get("model"):
        payload["model"] = ctx.settings.chat_model
    _apply_generation_defaults(ctx, payload)

    return payload, conversation_id, request_id, user_msg_id, user_text, tool_chain, plan


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        body = await request.json()
    except Exception:
        return _error(400, "request body is not valid JSON", "invalid_json")

    headers = {k.lower(): v for k, v in request.headers.items()}
    messages = body.get("messages") if isinstance(body, dict) else None
    classification = classify(body or {}, headers, messages if isinstance(messages, list) else [])

    if classification in ("side", "probe"):
        return await _passthrough(ctx, body)

    prepared = await _prepare(ctx, body, request)
    if isinstance(prepared, JSONResponse):
        return prepared
    payload, conversation_id, request_id, user_msg_id, user_text, _chain, plan = prepared

    stream = bool(body.get("stream", False))
    if stream:
        try:
            upstream_resp = await ctx.upstream.start_stream(payload)
        except UpstreamError as exc:
            return _error(
                502,
                f"The local model server rejected the request: {exc.message}. "
                "Check that OVMS is running and the model is loaded.",
                "upstream_error",
                "upstream_error",
            )
        except httpx.HTTPError as exc:
            return _error(
                503,
                f"Cannot reach the local model server at {ctx.settings.upstream_base_url} "
                f"({type(exc).__name__}). Start OVMS or change the upstream URL in Settings.",
                "upstream_unavailable",
                "upstream_error",
            )
        return StreamingResponse(
            _stream_and_archive(ctx, upstream_resp, conversation_id, request_id, user_msg_id, user_text, plan),
            media_type="text/event-stream",
            headers={
                **plan_headers(conversation_id, request_id),
                "X-AIO-Memory-Items": str(len(plan.get("items") or [])),
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        data = await ctx.upstream.complete(payload)
    except UpstreamError as exc:
        return _error(502, exc.message, "upstream_error", "upstream_error")
    except httpx.HTTPError as exc:
        return _error(
            503,
            f"Cannot reach the local model server at {ctx.settings.upstream_base_url} "
            f"({type(exc).__name__}). Start OVMS or change the upstream URL in Settings.",
            "upstream_unavailable",
            "upstream_error",
        )

    content = ""
    tool_calls: list[dict] = []
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []

    assistant_id, _ = _archive_assistant(
        ctx,
        conversation_id=conversation_id,
        content=content,
        tool_calls=tool_calls,
        request_id=request_id,
        model=str(payload.get("model") or ""),
    )
    _enqueue_filing(
        ctx,
        conversation_id=conversation_id,
        user_message_id=user_msg_id,
        assistant_message_id=assistant_id,
        user_text=user_text,
        assistant_text=content,
    )
    return JSONResponse(
        content=data,
        headers={
            **plan_headers(conversation_id, request_id),
            "X-AIO-Memory-Items": str(len(plan.get("items") or [])),
        },
    )


def plan_headers(conversation_id: str, request_id: str) -> dict[str, str]:
    return {
        "X-AIO-Conversation": conversation_id,
        "X-AIO-Request-Id": request_id,
    }


async def _stream_and_archive(
    ctx: AppContext,
    upstream_resp: httpx.Response,
    conversation_id: str,
    request_id: str,
    user_message_id: str,
    user_text: str,
    plan: dict,
) -> AsyncIterator[bytes]:
    state: dict = {"content": [], "reasoning": [], "tool_calls": {}, "finish_reason": None}
    buffer = ""
    try:
        async for chunk in upstream_resp.aiter_bytes():
            yield chunk
            buffer += chunk.decode("utf-8", "ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                _accumulate_sse(line.strip("\r"), state)
    except Exception as exc:  # upstream died mid-stream; keep the partial answer
        frame = {
            "error": {
                "message": f"The model stream stopped unexpectedly ({type(exc).__name__}).",
                "code": "upstream_stream_error",
            }
        }
        try:
            yield f"data: {json.dumps(frame)}\n\n".encode("utf-8")
        except Exception:
            pass
    finally:
        await upstream_resp.aclose()
        content = "".join(state["content"])
        tool_calls = [state["tool_calls"][k] for k in sorted(state["tool_calls"])]
        assistant_id, inserted = _archive_assistant(
            ctx,
            conversation_id=conversation_id,
            content=content,
            tool_calls=tool_calls,
            request_id=request_id,
            model=str(plan.get("model") or ""),
        )
        if inserted and (content.strip() or tool_calls):
            _enqueue_filing(
                ctx,
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                assistant_message_id=assistant_id,
                user_text=user_text,
                assistant_text=content,
            )
        ctx.bus.publish(
            "chat_done",
            {
                "conversation_id": conversation_id,
                "request_id": request_id,
                "chars": len(content),
                "memory_items": len(plan.get("items") or []),
            },
        )


async def _passthrough(ctx: AppContext, body: dict):
    """Forward a side/probe request without touching memory."""
    payload = _apply_generation_defaults(ctx, dict(body))
    if payload.get("stream"):
        try:
            upstream_resp = await ctx.upstream.start_stream(payload)
        except UpstreamError as exc:
            return _error(502, exc.message, "upstream_error", "upstream_error")
        except httpx.HTTPError as exc:
            return _error(503, f"Cannot reach the model server ({type(exc).__name__}).", "upstream_unavailable", "upstream_error")

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(relay(), media_type="text/event-stream")
    try:
        data = await ctx.upstream.complete(payload)
        return JSONResponse(content=data)
    except UpstreamError as exc:
        return _error(502, exc.message, "upstream_error", "upstream_error")
    except httpx.HTTPError as exc:
        return _error(503, f"Cannot reach the model server ({type(exc).__name__}).", "upstream_unavailable", "upstream_error")


@router.get("/v1/models")
async def list_models(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        data = await ctx.upstream.list_models()
        return JSONResponse(content=data)
    except Exception:
        from .models import CATALOG

        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": m["id"],
                        "object": "model",
                        "created": 0,
                        "owned_by": "aiolocal",
                        "active": False,
                    }
                    for m in CATALOG
                ],
            },
            headers={"X-AIO-Upstream": "unavailable"},
        )
