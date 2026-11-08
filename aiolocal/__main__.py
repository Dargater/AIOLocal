"""Launcher: starts the local server and opens the UI.

The server is loopback-only by construction. There is no flag to bind a public
interface, because the design does not support multi-user or remote access.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import APP_NAME, __version__
from .app import create_app
from .config import Paths, SettingsStore
from .security import load_or_create_token

HOST = "127.0.0.1"
DEFAULT_PORT = 7317


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((HOST, port))
            return True
        except OSError:
            return False


def pick_port(preferred: int, attempts: int = 20) -> int:
    for offset in range(attempts):
        candidate = preferred + offset
        if _port_free(candidate):
            return candidate
    raise RuntimeError(f"no free port in range {preferred}-{preferred + attempts - 1}")


def configure_logging(log_dir: Path, level: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "aiolocal.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    root.addHandler(console)

    # These libraries are noisy and may log request paths.
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def open_when_ready(port: int, timeout: float = 25.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.5):
                webbrowser.open(f"http://{HOST}:{port}/")
                return
        except OSError:
            time.sleep(0.25)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiolocal", description=f"{APP_NAME} {__version__}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--data-dir", type=str, default="", help="override the data directory (testing)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    paths = Paths.create(args.data_dir or None)
    configure_logging(paths.log_dir, args.log_level)
    log = logging.getLogger("aiolocal")

    settings_store = SettingsStore(paths)
    settings = settings_store.load()
    token = load_or_create_token(paths.token_path)
    port = pick_port(args.port)

    app = create_app(paths, settings_store, token, port=port)

    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(port,), daemon=True).start()

    print(f"{APP_NAME} {__version__}")
    print(f"  UI:    http://{HOST}:{port}/")
    print(f"  Data:  {paths.data_dir}")
    print("  Stop with Ctrl+C.")
    if settings.upstream_base_url:
        print(f"  Model server: {settings.upstream_base_url}")
    log.info("listening on %s:%s", HOST, port)

    import uvicorn

    try:
        uvicorn.run(
            app,
            host=HOST,
            port=port,
            log_config=None,
            access_log=False,
            timeout_keep_alive=30,
            server_header=False,
            date_header=False,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log.exception("server failed")
        print(f"Server failed to start: {exc}", file=sys.stderr)
        print(f"See {paths.log_dir / 'aiolocal.log'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
