"""PDF page vision analysis into conversation memory.

A document attached in chat is rendered page by page, sent to the main vision
model three pages at a time, and the analysis is appended to one durable summary
entry for that document. The final consolidated memory is written into the same
entry so Memory search can find it later.

Rendering needs ``pypdfium2`` and Pillow. Both are small and ship with the
Windows build. Nothing is written to the user's PDF or to the knowledge files
folder unless ``document_keep_pages`` is enabled; the temporary upload is deleted
when the job ends.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from .config import Paths, Settings, SettingsStore
from .db import Database, now_ms
from .llm import api_url
from .memory import store
from .memory import semantic
from .memory.text import new_id
from .models import CATALOG

VISION_SYSTEM = (
    "You are the document indexing engine of a local AI workspace. You read PDF "
    "page images carefully and write durable memory notes. Never invent content: "
    "when a page is unclear, say so. Preserve exact names, numbers, dates, units, "
    "definitions, equations and code."
)

BATCH_PROMPT = """Index pages {first}-{last} of "{title}" for long-term memory.

For EACH page, write a section exactly like "Page N" and cover:
- the subject and purpose of the page;
- key facts, names, dates, numbers, definitions and conclusions;
- diagrams, charts and tables: describe axes, labels, values and relationships;
- equations or code: transcribe them exactly;
- anything a reader would need to find this page again (terms, titles, headings).

Finish the whole answer with a "Memory-worthy points" bullet list of the five
most important takeaways across these pages. Be concise but complete. If a page
is blank or illegible, say so instead of guessing."""

FINAL_PROMPT = """Consolidate the page notes below into ONE durable memory entry for
"{title}" ({pages} pages).

Write 250-700 words in these sections:
1. Overview - what the document is and what it covers.
2. Key facts - the most important details, numbers, names and definitions.
3. Structure - the main sections/pages and what each contains.
4. Practical details - equations, code, tables, instructions or warnings.
5. Retrieval keywords - 8-15 terms a person would use to find this document.

Do not invent anything that is not in the notes. If the notes are partial, say
which parts are missing. Return plain Markdown only.

PAGE NOTES:
{notes}"""


@dataclass
class DocumentJob:
    id: str
    filename: str
    path: str
    title: str
    state: str = "queued"
    model: str = ""
    summary_id: str = ""
    pages_total: int = 0
    pages_done: int = 0
    batches_total: int = 0
    batches_done: int = 0
    final: str = ""
    error: str = ""
    created_at: int = field(default_factory=now_ms)
    updated_at: int = field(default_factory=now_ms)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "title": self.title,
            "state": self.state,
            "model": self.model,
            "summary_id": self.summary_id,
            "pages_total": self.pages_total,
            "pages_done": self.pages_done,
            "batches_total": self.batches_total,
            "batches_done": self.batches_done,
            "final": self.final,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class DocumentError(Exception):
    """The document job could not be started or completed."""


class DocumentAnalyzer:
    """Runs PDF jobs on private worker threads."""

    def __init__(
        self,
        paths: Paths,
        settings_store: SettingsStore,
        db: Database,
        *,
        publish: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.paths = paths
        self.settings_store = settings_store
        self.db = db
        self.publish = publish
        self.dir = paths.data_dir / "documents"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, DocumentJob] = {}
        self._threads: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                job.cancel.set()
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=2.0)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.public() for j in jobs[: max(1, min(int(limit), 100))]]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.public() if job else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job.cancel.set()
            return True

    # ------------------------------------------------------------------ start
    def start(self, upload: Path, filename: str) -> dict:
        """Take ownership of an uploaded PDF and return the queued job."""
        settings = self.settings_store.load()
        model = self._vision_model(settings)
        if not model:
            try:
                upload.unlink(missing_ok=True)
            except OSError:
                pass
            raise DocumentError(
                "No vision model is available. Download and load Qwen3.5 9B or "
                "Qwen3-VL 4B in the Models view first."
            )
        suffix = Path(filename or upload.name).suffix.lower()
        if suffix != ".pdf":
            try:
                upload.unlink(missing_ok=True)
            except OSError:
                pass
            raise DocumentError("only PDF files can be analysed page by page")
        try:
            import pypdfium2 as pdfium  # noqa: F401
        except ImportError as exc:
            try:
                upload.unlink(missing_ok=True)
            except OSError:
                pass
            raise DocumentError(
                "PDF rendering is not installed in this build (pypdfium2/Pillow missing)."
            ) from exc

        job_id = new_id()
        clean = Path(filename or upload.name).name[:180] or "document.pdf"
        job = DocumentJob(
            id=job_id,
            filename=clean,
            path=str(upload),
            title=clean.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip() or clean,
            model=model,
        )
        with self._lock:
            self._jobs[job_id] = job
            thread = threading.Thread(target=self._run, args=(job,), name=f"doc-{job_id}", daemon=True)
            self._threads[job_id] = thread
        thread.start()
        self._emit(job)
        return job.public()

    # ------------------------------------------------------------------ worker
    def _run(self, job: DocumentJob) -> None:
        settings = self.settings_store.load()
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(job.path)
            try:
                total = len(pdf)
            finally:
                pdf.close()
            if total <= 0:
                raise DocumentError("this PDF has no pages")
            job.pages_total = min(total, settings.document_max_pages)
        except DocumentError as exc:
            self._fail(job, str(exc))
            return
        except Exception as exc:
            self._fail(job, f"could not open the PDF ({type(exc).__name__}: {exc})")
            return

        job.batches_total = (job.pages_total + settings.document_pages_per_batch - 1) // max(
            1, settings.document_pages_per_batch
        )
        job.state = "running"
        job.updated_at = now_ms()
        self._emit(job)

        header = (
            f"# Document: {job.title}\n"
            f"Source: {job.filename} ({job.pages_total} pages). "
            "Analysed page by page with the local vision model.\n\n"
            "## Consolidated memory\n_Analysis in progress._\n\n## Page notes\n"
        )
        try:
            job.summary_id = store.insert_summary(
                self.db,
                title=f"Document: {job.title}"[:200],
                body=header,
                topic_slug="documents",
                kind="document",
                salience=0.72,
                confidence=0.8,
                source="document",
                entities=["document", job.title],
                embed_dim=settings.embed_dim,
            )
        except Exception as exc:
            self._fail(job, f"could not create the memory entry ({type(exc).__name__}: {exc})")
            return
        try:
            semantic.index_summary(
                self.db, settings, job.summary_id, f"Document: {job.title}", header
            )
        except Exception:
            pass
        self._emit(job)

        notes: list[tuple[int, int, str]] = []
        try:
            for batch_no, start in enumerate(
                range(1, job.pages_total + 1, settings.document_pages_per_batch), start=1
            ):
                if job.cancel.is_set():
                    job.state = "cancelled"
                    break
                end = min(job.pages_total, start + settings.document_pages_per_batch - 1)
                images = self._render_pages(job.path, start, end, settings.document_dpi)
                if not images:
                    raise DocumentError(f"could not render pages {start}-{end}")
                text = self._vision_batch(
                    settings, job.model, job.title, start, end, images
                )
                notes.append((start, end, text))
                job.pages_done = end
                job.batches_done = batch_no
                job.updated_at = now_ms()
                self._append_notes(job, start, end, text, settings.embed_dim)
                self._emit(job)

            if job.state != "cancelled":
                final = self._consolidate(settings, job, notes)
                job.final = final
                job.state = "done"
                body = self._compose_body(job, notes, final, in_progress=False)
                store.update_summary_body(
                    self.db, job.summary_id, body, title=f"Document: {job.title}", embed_dim=settings.embed_dim
                )
                try:
                    semantic.index_summary(
                        self.db, settings, job.summary_id, f"Document: {job.title}", body
                    )
                except Exception:
                    pass
                job.updated_at = now_ms()
                self._emit(job)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"[:1000]
            job.state = "error"
            job.updated_at = now_ms()
            if job.summary_id:
                try:
                    body = self._compose_body(
                        job,
                        notes,
                        "",
                        in_progress=False,
                        error=job.error,
                    )
                    store.update_summary_body(self.db, job.summary_id, body, embed_dim=settings.embed_dim)
                except Exception:
                    pass
            self._emit(job)
        finally:
            with self._lock:
                self._threads.pop(job.id, None)
            if not settings.document_keep_pages:
                try:
                    Path(job.path).unlink(missing_ok=True)
                except OSError:
                    pass

    # ------------------------------------------------------------------ helpers
    def _vision_model(self, settings: Settings) -> str:
        preferred = str(settings.chat_model or "").strip()
        for entry in CATALOG:
            if entry["id"] == preferred and entry.get("vision") and entry.get("task") == "text_generation":
                return preferred
        for entry in CATALOG:
            if entry.get("vision") and entry.get("task") == "text_generation":
                return entry["id"]
        return ""

    def _client(self, settings: Settings) -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(600.0, connect=5.0, write=60.0),
            trust_env=False,
        )

    def _render_pages(self, path: str, first: int, last: int, dpi: int) -> list[bytes]:
        import pypdfium2 as pdfium

        images: list[bytes] = []
        pdf = pdfium.PdfDocument(path)
        try:
            for index in range(first - 1, last):
                page = pdf[index]
                bitmap = None
                pil = None
                try:
                    bitmap = page.render(scale=max(1.0, dpi / 72.0))
                    pil = bitmap.to_pil().convert("RGB")
                    buffer = io.BytesIO()
                    pil.save(buffer, format="JPEG", quality=84, optimize=True)
                    images.append(buffer.getvalue())
                finally:
                    if pil is not None:
                        pil.close()
                    if bitmap is not None:
                        bitmap.close()
                    page.close()
        finally:
            pdf.close()
        return images

    def _vision_batch(
        self,
        settings: Settings,
        model: str,
        title: str,
        first: int,
        last: int,
        images: list[bytes],
    ) -> str:
        parts: list[dict] = [
            {"type": "text", "text": BATCH_PROMPT.format(title=title, first=first, last=last)}
        ]
        for image in images:
            encoded = base64.b64encode(image).decode("ascii")
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": parts},
            ],
            "temperature": 0.2,
            "max_tokens": 1600,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = api_url(settings.upstream_base_url, "chat/completions")
        with self._client(settings) as client:
            response = client.post(url, json=payload)
        if response.status_code >= 400:
            raise DocumentError(
                f"the model server rejected pages {first}-{last} "
                f"(HTTP {response.status_code}): {response.text[:400]}"
            )
        try:
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            raise DocumentError(f"the model returned an unreadable answer ({type(exc).__name__})") from exc
        text = str(content).strip()
        if not text:
            raise DocumentError(f"the model returned an empty analysis for pages {first}-{last}")
        return text[:20_000]

    def _consolidate(
        self, settings: Settings, job: DocumentJob, notes: list[tuple[int, int, str]]
    ) -> str:
        notes_text = "\n\n".join(f"### Pages {a}-{b}\n{text}" for a, b, text in notes)
        if len(notes_text) > 14_000:
            notes_text = notes_text[:14_000] + "\n… notes truncated for consolidation"
        prompt = FINAL_PROMPT.format(title=job.title, pages=job.pages_total, notes=notes_text)
        payload = {
            "model": job.model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1400,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = api_url(settings.upstream_base_url, "chat/completions")
        try:
            with self._client(settings) as client:
                response = client.post(url, json=payload)
            if response.status_code < 400:
                content = response.json()["choices"][0]["message"].get("content") or ""
                content = str(content).strip()
                if content:
                    return content[:16_000]
        except Exception:
            pass
        # The page notes are already in memory; a failed consolidation must not lose
        # them. Build a deterministic summary instead.
        overview = notes[0][2][:1200] if notes else ""
        return (
            f"## Overview\n{overview}\n\n"
            f"## Key facts\nSee the page notes below. The consolidating model was unavailable, "
            f"so this document's memory contains {len(notes)} analysed batches covering "
            f"{job.pages_total} pages."
        )

    def _compose_body(
        self,
        job: DocumentJob,
        notes: list[tuple[int, int, str]],
        final: str,
        *,
        in_progress: bool,
        error: str = "",
    ) -> str:
        parts = [
            f"# Document: {job.title}",
            f"Source: {job.filename} ({job.pages_total} pages). "
            "Analysed page by page with the local vision model.",
            "## Consolidated memory",
            final or ("_Analysis in progress._" if in_progress else "_No consolidated memory was produced._"),
        ]
        if error:
            parts.extend(["", "## Analysis error", error])
        parts.append("")
        parts.append("## Page notes")
        for first, last, text in notes:
            parts.extend([f"### Pages {first}-{last}", text, ""])
        body = "\n".join(parts)
        # Keep the consolidated memory and the most recent page notes. The start
        # of the body is the most useful part, so trim the tail for very large PDFs.
        if len(body) > 40_000:
            body = body[:39_000] + "\n… page notes truncated at the memory size limit"
        return body

    def _append_notes(self, job: DocumentJob, first: int, last: int, text: str, embed_dim: int) -> None:
        row = store.get_summary(self.db, job.summary_id)
        if not row:
            return
        body = row.get("body") or ""
        marker = "## Page notes\n"
        addition = f"\n### Pages {first}-{last}\n{text}\n"
        if marker in body:
            body = body.replace(marker, marker + addition.lstrip("\n"), 1)
        else:
            body += addition
        if len(body) > 40_000:
            body = body[:39_000] + "\n… page notes truncated at the memory size limit"
        # Keep the progress marker accurate without discarding earlier batches.
        store.update_summary_body(
            self.db,
            job.summary_id,
            body,
            title=f"Document: {job.title}",
            embed_dim=embed_dim,
        )

    def _fail(self, job: DocumentJob, message: str) -> None:
        job.state = "error"
        job.error = message[:1000]
        job.updated_at = now_ms()
        self._emit(job)
        try:
            Path(job.path).unlink(missing_ok=True)
        except OSError:
            pass

    def _emit(self, job: DocumentJob) -> None:
        if not self.publish:
            return
        try:
            self.publish("document_job", job.public())
        except Exception:
            pass
