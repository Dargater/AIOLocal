"""Image generation and editing proxy.

OVMS exposes an OpenAI-compatible image API. The gateway forwards a sanitised subset
of the request, then stores any returned base64 image locally with a random id. Images
are never written to a user-supplied path.
"""

from __future__ import annotations

import base64
import binascii
import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from .context import AppContext
from .db import json_dumps, now_ms
from .memory.text import new_id
from .security import safe_join

router = APIRouter()

MAX_IMAGE_BYTES = 32 * 1024 * 1024
ALLOWED_FIELDS = {
    "model", "prompt", "negative_prompt", "width", "height", "num_inference_steps",
    "guidance_scale", "seed", "n", "response_format", "strength",
}


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )


def _sniff(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif", "image/gif"
    return "", "application/octet-stream"


def _decode_b64(value: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError("image data must be a base64 string")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    if len(value) > MAX_IMAGE_BYTES * 4 // 3 + 16:
        raise ValueError("image is too large")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 image data") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image is too large")
    if not data:
        raise ValueError("empty image")
    return data


def _save_image(
    ctx: AppContext,
    data: bytes,
    *,
    prompt: str,
    negative: str,
    model: str,
    width: int | None,
    height: int | None,
    seed: int | None,
) -> dict:
    ext, mime = _sniff(data)
    if not ext:
        raise ValueError("upstream returned an unsupported image format")
    image_id = new_id()
    filename = f"{image_id}.{ext}"
    path = safe_join(ctx.paths.image_dir, filename)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    with ctx.db.write() as c:
        c.execute(
            """INSERT INTO images(id, prompt, negative, model, path, width, height, seed, created_at, meta)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                image_id, prompt[:4000], negative[:2000], model, filename,
                width, height, seed, now_ms(), json_dumps({"mime": mime, "bytes": len(data)}),
            ),
        )
    return {
        "id": image_id,
        "url": f"/api/v1/images/{image_id}/file",
        "width": width,
        "height": height,
        "seed": seed,
        "mime": mime,
    }


def _active_image_model(ctx: AppContext, default: str) -> str:
    try:
        from .models import catalog_entry

        active = (ctx.models.status() if ctx.models else {}).get("active") or {}
        for model_id in active:
            entry = catalog_entry(model_id)
            if entry and entry["task"] == "image_generation":
                return model_id
    except Exception:
        pass
    return default


def _sanitise(payload: dict) -> dict:
    clean: dict = {}
    for key in ALLOWED_FIELDS:
        if key in payload and payload[key] is not None:
            clean[key] = payload[key]
    prompt = str(clean.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("a prompt is required")
    if len(prompt) > 4000:
        raise ValueError("prompt is too long (4000 characters max)")
    clean["prompt"] = prompt
    clean["negative_prompt"] = str(clean.get("negative_prompt") or "")[:2000]
    clean["model"] = str(clean.get("model") or "sd15")
    for key, low, high, default in (("width", 64, 1024, 512), ("height", 64, 1024, 512)):
        try:
            clean[key] = max(low, min(high, int(clean.get(key, default))))
        except (TypeError, ValueError):
            clean[key] = default
    try:
        clean["n"] = max(1, min(4, int(clean.get("n", 1))))
    except (TypeError, ValueError):
        clean["n"] = 1
    if "seed" in clean:
        try:
            clean["seed"] = int(clean["seed"])
        except (TypeError, ValueError):
            clean.pop("seed", None)
    return clean


@router.post("/api/v1/images/generate")
async def generate(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        body = await request.json()
    except Exception:
        return _error(400, "body must be JSON", "invalid_json")
    if not isinstance(body, dict):
        return _error(400, "body must be an object", "invalid_body")
    try:
        payload = _sanitise(body)
    except ValueError as exc:
        return _error(400, str(exc), "invalid_prompt")
    payload["model"] = _active_image_model(ctx, payload["model"])

    try:
        result = await ctx.upstream.images_generate(payload)
    except Exception as exc:
        return _error(
            502,
            f"The image model is not available: {exc}. Load an image_generation model in OVMS.",
            "image_upstream_error",
        )

    return _store_result(ctx, result, payload)


def _store_result(ctx: AppContext, result: dict, payload: dict) -> JSONResponse:
    items = result.get("data") or []
    saved: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        b64 = item.get("b64_json")
        if not b64:
            continue
        try:
            data = _decode_b64(b64)
            saved.append(
                _save_image(
                    ctx,
                    data,
                    prompt=str(payload.get("prompt") or ""),
                    negative=str(payload.get("negative_prompt") or ""),
                    model=str(payload.get("model") or "sd15"),
                    width=payload.get("width"),
                    height=payload.get("height"),
                    seed=payload.get("seed"),
                )
            )
        except (ValueError, OSError) as exc:
            return _error(502, f"Could not store the generated image: {exc}", "image_store_error")
    if not saved:
        return _error(502, "the image model returned no image data", "empty_image")
    ctx.bus.publish("image_done", {"count": len(saved), "ids": [s["id"] for s in saved]})
    return JSONResponse(content={"created": int(time.time()), "data": saved})


@router.post("/api/v1/images/edit")
async def edit(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        body = await request.json()
    except Exception:
        return _error(400, "body must be JSON", "invalid_json")
    if not isinstance(body, dict):
        return _error(400, "body must be an object", "invalid_body")

    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return _error(400, "a prompt is required", "invalid_prompt")
    if len(prompt) > 4000:
        return _error(400, "prompt is too long", "invalid_prompt")

    try:
        image_bytes, _ = _load_source(ctx, body.get("image_id"), body.get("image_b64"))
    except ValueError as exc:
        return _error(400, str(exc), "invalid_image")

    files = {"image": ("image.png", image_bytes, "image/png")}
    mask_bytes = None
    try:
        mask_bytes, _ = _load_source(ctx, body.get("mask_id"), body.get("mask_b64"))
    except ValueError:
        mask_bytes = None
    if mask_bytes:
        files["mask"] = ("mask.png", mask_bytes, "image/png")

    data = {
        "model": _active_image_model(ctx, "sd15-inpaint"),
        "prompt": prompt,
        "n": str(max(1, min(2, int(body.get("n", 1) or 1)))),
    }
    if body.get("negative_prompt"):
        data["negative_prompt"] = str(body["negative_prompt"])[:2000]

    try:
        result = await ctx.upstream.images_edit(data, files)
    except Exception as exc:
        return _error(
            502,
            f"The inpainting model is not available: {exc}. Load sd15-inpaint in OVMS.",
            "image_upstream_error",
        )
    return _store_result(
        ctx,
        result,
        {"prompt": prompt, "negative_prompt": data.get("negative_prompt", ""), "model": "sd15-inpaint"},
    )


def _load_source(ctx: AppContext, image_id: object, image_b64: object) -> tuple[bytes, str]:
    if image_id:
        with ctx.db.read() as c:
            row = c.execute("SELECT path FROM images WHERE id=?", (str(image_id),)).fetchone()
        if not row:
            raise ValueError("image not found")
        path = safe_join(ctx.paths.image_dir, str(row["path"]))
        if not path.exists():
            raise ValueError("image file is missing")
        return path.read_bytes(), str(row["path"])
    if image_b64:
        return _decode_b64(str(image_b64)), ""
    raise ValueError("provide image_id or image_b64")


@router.get("/api/v1/images")
async def list_images(request: Request, limit: int = 60, offset: int = 0):
    ctx: AppContext = request.app.state.ctx
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    with ctx.db.read() as c:
        rows = c.execute(
            """SELECT id, prompt, negative, model, width, height, seed, created_at
               FROM images ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    return JSONResponse(
        content={
            "data": [
                {
                    **dict(r),
                    "url": f"/api/v1/images/{r['id']}/file",
                }
                for r in rows
            ]
        }
    )


@router.delete("/api/v1/images/{image_id}")
async def delete_image(image_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not image_id or len(image_id) > 64:
        return _error(404, "image not found", "not_found")
    with ctx.db.write() as c:
        row = c.execute("SELECT path FROM images WHERE id=?", (image_id,)).fetchone()
        if not row:
            return _error(404, "image not found", "not_found")
        c.execute("DELETE FROM images WHERE id=?", (image_id,))
    try:
        path = safe_join(ctx.paths.image_dir, str(row["path"]))
        path.unlink(missing_ok=True)
    except (ValueError, OSError):
        pass
    return {"ok": True, "deleted": image_id}


@router.get("/api/v1/images/{image_id}/file")
async def image_file(image_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not image_id or len(image_id) > 64:
        return _error(404, "image not found", "not_found")
    with ctx.db.read() as c:
        row = c.execute("SELECT path, meta FROM images WHERE id=?", (image_id,)).fetchone()
    if not row:
        return _error(404, "image not found", "not_found")
    try:
        path = safe_join(ctx.paths.image_dir, str(row["path"]))
    except ValueError:
        return _error(404, "image not found", "not_found")
    if not path.exists():
        return _error(404, "image file is missing", "not_found")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=31536000, immutable"})
