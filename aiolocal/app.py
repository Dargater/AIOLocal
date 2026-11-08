"""FastAPI application: wiring, security middleware and static UI."""

from __future__ import annotations

import html
import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Paths, SettingsStore
from .context import AppContext
from .db import Database
from .events import EventBus
from .gateway import router as gateway_router
from .control import router as control_router
from .images import router as images_router
from .llm import Upstream
from .memory.jobs import MemoryWorker
from .models import ModelManager
from .knowledge import KnowledgeBase, KnowledgeWorker
from .websearch import WebSearch
from .security import host_allowed, origin_allowed, tokens_equal, SecretStore
from .code import CodeWorkspace
from .reverse import ReverseEngineering
from .gimp import GimpDock
from .requirements import RequirementsManager

log = logging.getLogger("aiolocal.app")

MAX_BODY_BYTES = 48 * 1024 * 1024

CSP = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "font-src 'self'; "
    # The embedded OpenCode server is a loopback origin on a different port. It is
    # the only frame source; everything else stays blocked.
    "frame-src http://127.0.0.1:* http://localhost:*; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)


def web_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        candidate = Path(bundled) / "aiolocal" / "web"
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parent / "web"


def _token_from_request(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    header = request.headers.get("x-aio-token")
    if header:
        return header.strip()
    # EventSource and <img> cannot set headers. These two GET-only endpoints accept
    # the token as a query parameter; they are still protected by Host/Origin checks.
    if request.method == "GET" and (
        request.url.path == "/api/v1/events"
        or (request.url.path.startswith("/api/v1/images/") and request.url.path.endswith("/file"))
    ):
        return request.query_params.get("token")
    return None


def create_app(
    paths: Paths, settings_store: SettingsStore, token: str, port: int = 7317
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = settings_store.load()
        db = Database(paths.db_path)
        try:
            upstream = Upstream(settings.upstream_base_url)
        except ValueError:
            log.warning("invalid upstream URL in settings; falling back to 127.0.0.1:8000")
            upstream = Upstream("http://127.0.0.1:8000")
        bus = EventBus()
        worker = MemoryWorker(db, settings, on_event=bus.publish)
        models = ModelManager(paths, settings_store, publish=bus.publish)
        requirements = RequirementsManager(paths, settings_store, publish=bus.publish)
        kb = None
        kb_worker = None
        try:
            kb = KnowledgeBase(paths.knowledge_db_path, files_dir=paths.knowledge_files)
            kb_worker = KnowledgeWorker(kb, paths, settings_store, on_event=lambda ev, d: bus.publish(ev, d))
            kb_worker.start()
        except Exception:
            log.exception("knowledge base failed to start")
        from .security import SecretStore
        secrets = SecretStore(paths.secret_path)
        web_search = WebSearch(settings, secrets)
        # OpenCode talks to this app's gateway, not directly to OVMS: it must go
        # through memory, knowledge opt-in and web-safety handling.
        code_ws = CodeWorkspace(paths, settings_store, base_url=f"http://127.0.0.1:{port}")
        reverse = ReverseEngineering(settings_store, paths)

        def lookup_image(image_id: str):
            from .security import safe_join

            with db.read() as conn:
                row = conn.execute("SELECT path FROM images WHERE id=?", (str(image_id),)).fetchone()
            if not row:
                raise ValueError("image not found")
            path = safe_join(paths.image_dir, str(row["path"]))
            if not path.is_file():
                raise ValueError("image file is missing")
            return path

        gimp = GimpDock(paths, settings_store, image_lookup=lookup_image)
        ctx = AppContext(
            paths=paths,
            settings_store=settings_store,
            settings=settings,
            db=db,
            upstream=upstream,
            bus=bus,
            token=token,
            worker=worker,
            models=models,
            kb=kb,
            kb_worker=kb_worker,
            web_search=web_search,
            secrets=secrets,
            code_ws=code_ws,
            reverse=reverse,
            gimp=gimp,
            requirements=requirements,
        )
        app.state.ctx = ctx
        worker.start()
        log.info("AIO Local %s started; data=%s", __version__, paths.data_dir)

        # Start OVMS in the background when models are already selected, unless an
        # external server is already answering on the configured upstream URL.
        def _auto_start_ovms() -> None:
            try:
                if not settings.ovms_auto_start:
                    return
                if not (models._state.get("active") or {}):  # noqa: SLF001 - same package
                    return
                if not models._ovms_executable():  # noqa: SLF001
                    return
                import httpx

                from .llm import api_url

                try:
                    response = httpx.get(
                        api_url(settings.upstream_base_url, "models"),
                        timeout=2.5,
                        trust_env=False,
                    )
                    if response.status_code == 200:
                        log.info("external model server already running; not starting OVMS")
                        return
                except Exception:
                    pass
                models.start_ovms()
                log.info("OVMS auto-start requested")
            except Exception:
                log.exception("OVMS auto-start failed")

        threading.Thread(target=_auto_start_ovms, name="ovms-autostart", daemon=True).start()
        try:
            yield
        finally:
            log.info("shutting down")
            worker.stop()
            try:
                code_ws.stop()
            except Exception:
                log.exception("failed to stop OpenCode")
            try:
                reverse.stop()
            except Exception:
                log.exception("failed to stop REA")
            try:
                gimp.release()
            except Exception:
                log.exception("failed to release the GIMP dock")
            try:
                models.stop_ovms()
            except Exception:
                log.exception("failed to stop OVMS")
            await upstream.aclose()
            db.close()

    app = FastAPI(
        title="AIO Local",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        port = request.url.port or 7317

        if not host_allowed(request.headers.get("host"), port):
            return JSONResponse(
                status_code=403,
                content={"error": {"message": "host not allowed", "code": "bad_host"}},
            )
        origin = request.headers.get("origin") or request.headers.get("referer")
        if not origin_allowed(origin, port):
            return JSONResponse(
                status_code=403,
                content={"error": {"message": "cross-origin request rejected", "code": "bad_origin"}},
            )
        if request.method == "OPTIONS":
            return JSONResponse(status_code=403, content={"error": {"message": "CORS is not enabled"}})

        path = request.url.path
        protected = path.startswith("/api/") or path.startswith("/v1/")
        if protected and not tokens_equal(token, _token_from_request(request)):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "missing or invalid local token", "code": "unauthorized"}},
                headers={"WWW-Authenticate": "Bearer"},
            )

        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": {"message": "request body is too large", "code": "too_large"}},
            )

        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if not protected or path == "/":
            response.headers.setdefault("Content-Security-Policy", CSP)
        return response

    app.include_router(control_router)
    app.include_router(images_router)
    app.include_router(gateway_router)

    root = web_root()
    static_dir = root
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index():
        page = root / "index.html"
        if not page.exists():
            return Response("UI files are missing from this build.", status_code=500, media_type="text/plain")
        markup = page.read_text(encoding="utf-8").replace("__AIO_TOKEN__", html.escape(token))
        try:
            stamp = max(
                (root / "app.js").stat().st_mtime,
                (root / "styles.css").stat().st_mtime,
            )
            markup = markup.replace("__AIO_ASSET_VERSION__", str(int(stamp)))
        except OSError:
            markup = markup.replace("__AIO_ASSET_VERSION__", "1")
        return Response(
            markup,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/favicon.ico")
    async def favicon():
        icon = root / "favicon.svg"
        if icon.exists():
            return FileResponse(icon, media_type="image/svg+xml")
        return Response(status_code=204)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "version": __version__}

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "internal error; see the local log", "code": "internal_error"}},
        )

    return app
