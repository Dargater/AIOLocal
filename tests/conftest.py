from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from aiolocal.app import create_app
from aiolocal.config import Paths, Settings, SettingsStore

TOKEN = "test-token-0123456789abcdef"
PORT = 7317
BASE = f"http://127.0.0.1:{PORT}"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_fake_upstream(state: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        state["last"] = body
        state["calls"] = state.get("calls", 0) + 1
        if body.get("stream"):
            async def gen():
                for piece in ["Hello", " from", " fake"]:
                    frame = {"choices": [{"delta": {"content": piece}, "finish_reason": None, "index": 0}]}
                    yield f"data: {json.dumps(frame)}\n\n"
                final = {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return JSONResponse(
            {
                "id": "fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "non-stream reply"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    @app.post("/v1/images/generations")
    async def images(request: Request):
        body = await request.json()
        state["last_image"] = body
        # 1x1 transparent PNG
        b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        return {"created": 0, "data": [{"b64_json": b64}]}

    return app


@pytest.fixture()
def upstream():
    state: dict = {}
    port = free_port()
    config = uvicorn.Config(
        make_fake_upstream(state),
        host="127.0.0.1",
        port=port,
        log_config=None,
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("fake upstream did not start")

    state["base_url"] = f"http://127.0.0.1:{port}"
    yield state

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def env(tmp_path: Path, upstream):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    settings = Settings(
        upstream_base_url=upstream["base_url"],
        chat_model="fake-model",
        context_tokens=4096,
        max_output_tokens=256,
        short_history_turns=4,
        embedding_backend="hash",
    )
    store.save(settings)
    app = create_app(paths, store, TOKEN)
    with TestClient(app, base_url=BASE) as client:
        yield {
            "client": client,
            "paths": paths,
            "store": store,
            "settings": settings,
            "app": app,
            "upstream": upstream,
            "token": TOKEN,
            "base": BASE,
        }


@pytest.fixture()
def client(env):
    return env["client"]
