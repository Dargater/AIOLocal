"""Shared application context.

Everything that lives for the lifetime of the server (database, upstream client,
memory worker, event bus) is created once and passed explicitly. No module-level
globals, which keeps the app testable and makes shutdown deterministic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Paths, Settings, SettingsStore
from .db import Database
from .events import EventBus
from .llm import Upstream
from .memory.jobs import MemoryWorker
from .models import ModelManager
from .knowledge import KnowledgeBase, KnowledgeWorker
from .websearch import WebSearch
from .security import SecretStore
from .code import CodeWorkspace
from .reverse import ReverseEngineering
from .gimp import GimpDock
from .requirements import RequirementsManager


@dataclass
class AppContext:
    paths: Paths
    settings_store: SettingsStore
    settings: Settings
    db: Database
    upstream: Upstream
    bus: EventBus
    token: str
    started_at: int = field(default_factory=lambda: int(time.time() * 1000))
    worker: MemoryWorker | None = None
    models: ModelManager | None = None
    kb: KnowledgeBase | None = None
    kb_worker: KnowledgeWorker | None = None
    web_search: WebSearch | None = None
    secrets: SecretStore | None = None
    code_ws: CodeWorkspace | None = None
    reverse: ReverseEngineering | None = None
    gimp: GimpDock | None = None
    requirements: RequirementsManager | None = None

    def update_settings(self, settings: Settings) -> None:
        self.settings_store.save(settings)
        self.settings = settings
        if self.worker is not None:
            self.worker.settings = settings
        self.upstream.base_url = settings.upstream_base_url.rstrip("/")
        # code_ws intentionally keeps pointing at this app's gateway, not at OVMS.

    def close(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self.kb_worker is not None:
            try:
                self.kb_worker.stop(timeout=3.0)
            except Exception:
                pass
        if self.kb is not None:
            try:
                self.kb.close()
            except Exception:
                pass
        if self.code_ws is not None and self.code_ws.running():
            try:
                self.code_ws.stop()
            except Exception:
                pass
        if self.reverse is not None:
            try:
                if self.reverse.client is not None:
                    self.reverse.stop()
            except Exception:
                pass
        if self.gimp is not None:
            try:
                self.gimp.shutdown()
            except Exception:
                pass
        self.db.close()
