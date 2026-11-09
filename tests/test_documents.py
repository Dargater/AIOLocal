from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from pypdf import PdfWriter

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal.db import Database
from aiolocal.documents import DocumentAnalyzer, DocumentError
from aiolocal.memory import store


def make_pdf(path: Path, pages: int = 5) -> Path:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def make_analyzer(tmp_path: Path, **settings_kwargs) -> tuple[DocumentAnalyzer, Database, Paths]:
    paths = Paths.create(tmp_path / "data")
    settings_store = SettingsStore(paths)
    settings_store.save(Settings(chat_model="qwen3.5-9b", **settings_kwargs))
    db = Database(paths.db_path)
    return DocumentAnalyzer(paths, settings_store, db), db, paths


def wait_for(analyzer: DocumentAnalyzer, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = analyzer.get(job_id)
        if current and current["state"] in ("done", "error", "cancelled"):
            return current
        time.sleep(0.02)
    raise AssertionError("document job did not finish")


def test_pdf_is_analysed_in_three_page_batches(tmp_path: Path):
    analyzer, db, _ = make_analyzer(tmp_path)
    pdf = make_pdf(tmp_path / "lesson.pdf", pages=5)
    calls: list[tuple[int, int, int]] = []

    analyzer._render_pages = lambda path, first, last, dpi: [b"page"] * (last - first + 1)

    def vision(settings, model, title, first, last, images):
        calls.append((first, last, len(images)))
        return f"Notes for pages {first}-{last}."

    analyzer._vision_batch = vision
    analyzer._consolidate = lambda settings, job, notes: "CONSOLIDATED MEMORY"

    job = analyzer.start(pdf, "lesson.pdf")
    current = wait_for(analyzer, job["id"])

    assert current["state"] == "done", current
    assert current["pages_total"] == 5
    assert current["pages_done"] == 5
    assert current["batches_total"] == 2
    assert current["batches_done"] == 2
    assert calls == [(1, 3, 3), (4, 5, 2)]
    assert current["final"] == "CONSOLIDATED MEMORY"
    deadline = time.time() + 5
    while pdf.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert not pdf.exists()

    summary = store.get_summary(db, current["summary_id"])
    assert summary is not None
    assert summary["kind"] == "document"
    assert "CONSOLIDATED MEMORY" in summary["body"]
    assert "### Pages 1-3" in summary["body"]
    assert "### Pages 4-5" in summary["body"]


def test_cancel_stops_before_the_final_consolidation(tmp_path: Path):
    analyzer, db, _ = make_analyzer(tmp_path)
    pdf = make_pdf(tmp_path / "cancel.pdf", pages=4)
    seen: list[str] = []
    gate = threading.Event()
    holder: list[str] = []

    analyzer._render_pages = lambda path, first, last, dpi: [b"page"] * (last - first + 1)

    def vision(settings, model, title, first, last, images):
        gate.wait(5)
        seen.append(f"{first}-{last}")
        if holder:
            analyzer.cancel(holder[0])
        return "partial"

    analyzer._vision_batch = vision
    analyzer._consolidate = lambda settings, job, notes: "SHOULD NOT RUN"

    job = analyzer.start(pdf, "cancel.pdf")
    holder.append(job["id"])
    gate.set()
    current = wait_for(analyzer, job["id"])

    assert current["state"] == "cancelled"
    assert current["final"] == ""
    summary = store.get_summary(db, current["summary_id"])
    assert summary is not None
    assert "SHOULD NOT RUN" not in summary["body"]


def test_non_vision_chat_model_falls_back_to_a_vision_model(tmp_path: Path):
    analyzer, _, _ = make_analyzer(tmp_path)
    analyzer.settings_store.save(Settings(chat_model="qwen3-4b"))
    pdf = make_pdf(tmp_path / "plain.pdf", pages=1)

    job = analyzer.start(pdf, "plain.pdf")

    assert job["model"] == "qwen3.5-9b"
    analyzer.close()


def test_start_requires_a_vision_model(tmp_path: Path, monkeypatch):
    analyzer, _, _ = make_analyzer(tmp_path)
    import aiolocal.documents as documents

    monkeypatch.setattr(documents, "CATALOG", [])
    pdf = make_pdf(tmp_path / "plain.pdf", pages=1)

    with pytest.raises(DocumentError, match="vision model"):
        analyzer.start(pdf, "plain.pdf")
    assert not pdf.exists()


def test_only_pdf_files_are_accepted(tmp_path: Path):
    analyzer, _, _ = make_analyzer(tmp_path)
    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")

    with pytest.raises(DocumentError, match="only PDF"):
        analyzer.start(text, "notes.txt")
    assert not text.exists()


def test_document_endpoint_streams_upload_and_finishes(env):
    import io

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)

    response = env["client"].post(
        "/api/v1/documents/analyze?filename=lesson.pdf",
        content=buffer.getvalue(),
        headers={"X-AIO-Token": env["token"], "Content-Type": "application/pdf"},
    )
    assert response.status_code == 200, response.text
    job = response.json()["job"]

    deadline = time.time() + 15
    current = job
    while time.time() < deadline:
        current = env["client"].get(
            f"/api/v1/documents/jobs/{job['id']}", headers={"X-AIO-Token": env["token"]}
        ).json()
        if current["state"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)

    assert current["state"] == "done", current
    assert current["pages_total"] == 2
    assert current["summary_id"]
    summary = store.get_summary(env["app"].state.ctx.db, current["summary_id"])
    assert summary is not None
    assert summary["kind"] == "document"
