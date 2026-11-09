"""Optional PhotoGIMP installer.

PhotoGIMP is a GPL-3.0 configuration pack by Diolinux that makes GIMP feel like
Photoshop. It is never bundled with AIO Local: the user asks for it, the app
downloads the official release from GitHub, checks the zip structure, backs up
files it is about to replace, and installs only into this app's private GIMP
profile. Removing it restores those backups.

Linux is not managed here. The UI shows the official manual instructions instead,
so users keep control of their own GIMP profile.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .config import Paths

API_URL = "https://api.github.com/repos/Diolinux/PhotoGIMP/releases/latest"
FALLBACK_VERSION = "3.1"
FALLBACK_URL = "https://github.com/Diolinux/PhotoGIMP/releases/download/3.1/PhotoGIMP.zip"
MANUAL_URL = "https://github.com/Diolinux/PhotoGIMP"
MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 120 * 1024 * 1024


class PhotoGimpError(Exception):
    pass


class PhotoGIMP:
    def __init__(self, paths: Paths, gimp) -> None:
        self.paths = paths
        self.gimp = gimp
        self.root = paths.data_dir / "gimp"
        self.manifest_path = self.root / "photogimp.json"
        self.backup_root = self.root / "backups"

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        manifest = self._manifest()
        profile = self.gimp.personal_dir()
        files = manifest.get("files") or []
        present = 0
        for rel in files:
            try:
                if (profile / rel).is_file():
                    present += 1
            except OSError:
                pass
        return {
            "available": os.name == "nt",
            "installed": bool(manifest),
            "version": manifest.get("version") or "",
            "installed_at": manifest.get("installed_at") or 0,
            "source": manifest.get("source") or "",
            "profile": str(profile),
            "backup": manifest.get("backup") or "",
            "files": len(files),
            "files_present": present,
            "license": "GPL-3.0",
            "manual_url": MANUAL_URL,
            "note": (
                "PhotoGIMP is downloaded from the official Diolinux GitHub release on "
                "demand and installed only into this app's private GIMP profile. "
                "Linux users should follow the official manual instructions."
                if os.name != "nt"
                else "Downloads the official PhotoGIMP release from GitHub on demand."
            ),
        }

    def _manifest(self) -> dict:
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    # ------------------------------------------------------------------ install
    def install(self) -> dict:
        if os.name != "nt":
            raise PhotoGimpError(
                "Managed PhotoGIMP installation is Windows-only. On Linux, follow the "
                f"official instructions: {MANUAL_URL}"
            )
        version, url = self._latest_release()
        data = self._download(url)
        profile = self.gimp.personal_dir()
        previous = self._manifest()
        backup = Path(previous["backup"]) if previous.get("backup") else (
            self.backup_root / f"photogimp-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        backup.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        backed_up: list[str] = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                total = sum(item.file_size for item in archive.infolist())
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise PhotoGimpError("the PhotoGIMP archive is unexpectedly large")
                for item in archive.infolist():
                    if item.is_dir():
                        continue
                    rel = self._safe_relative(item.filename)
                    if rel is None:
                        continue
                    target = profile / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_file():
                        saved = backup / rel
                        if not saved.is_file():
                            saved.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(target, saved)
                            backed_up.append(rel)
                    with archive.open(item) as source, open(target, "wb") as sink:
                        shutil.copyfileobj(source, sink)
                    written.append(str(rel))
        except PhotoGimpError:
            raise
        except (OSError, zipfile.BadZipFile) as exc:
            raise PhotoGimpError(f"could not install PhotoGIMP: {exc}") from exc

        manifest = {
            "version": version,
            "installed_at": int(time.time() * 1000),
            "source": url,
            "files": sorted(written),
            "backup": str(backup),
            "license": "GPL-3.0",
        }
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temporary, self.manifest_path)
        except OSError as exc:
            raise PhotoGimpError(f"could not write the PhotoGIMP manifest: {exc}") from exc
        return {
            "ok": True,
            "version": version,
            "installed_files": len(written),
            "backed_up": len(backed_up),
            "backup": str(backup),
            "status": self.status(),
        }

    def _safe_relative(self, name: str) -> PurePosixPath | None:
        path = PurePosixPath(str(name or "").replace("\\", "/"))
        parts = [p for p in path.parts if p not in ("", ".")]
        if not parts or path.is_absolute() or ".." in parts or "__MACOSX" in parts:
            return None
        # The Windows PhotoGIMP zip has a top-level "3.0" folder that maps onto the
        # private GIMP profile. Flat archives are also accepted.
        if parts[0] == "3.0":
            parts = parts[1:]
        if not parts or parts[0].startswith("."):
            return None
        return PurePosixPath(*parts)

    def _latest_release(self) -> tuple[str, str]:
        try:
            request = urllib.request.Request(
                API_URL,
                headers={"User-Agent": "AIOLocal", "Accept": "application/vnd.github+json"},
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read(2_000_000).decode("utf-8", "replace"))
            tag = str(payload.get("tag_name") or FALLBACK_VERSION)
            for asset in payload.get("assets") or []:
                if isinstance(asset, dict) and asset.get("name") == "PhotoGIMP.zip":
                    url = str(asset.get("browser_download_url") or "")
                    if url.startswith("https://github.com/"):
                        return tag, url
        except Exception:
            pass
        return FALLBACK_VERSION, FALLBACK_URL

    def _download(self, url: str) -> bytes:
        if not url.startswith("https://github.com/Diolinux/PhotoGIMP/"):
            raise PhotoGimpError("the PhotoGIMP download URL was not the official repository")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AIOLocal"})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read(MAX_DOWNLOAD_BYTES + 1)
        except Exception as exc:
            raise PhotoGimpError(f"could not download PhotoGIMP: {exc}") from exc
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise PhotoGimpError("the PhotoGIMP download is too large")
        if not data:
            raise PhotoGimpError("the PhotoGIMP download was empty")
        return data

    # ------------------------------------------------------------------ remove
    def remove(self, *, restore: bool = True) -> dict:
        manifest = self._manifest()
        if not manifest:
            return {"ok": False, "reason": "PhotoGIMP is not installed by this app"}
        profile = self.gimp.personal_dir()
        backup = Path(manifest.get("backup") or "") if manifest.get("backup") else None
        removed = 0
        restored = 0
        for rel in manifest.get("files") or []:
            try:
                target = profile / str(rel)
                if target.is_file():
                    target.unlink()
                    removed += 1
                if restore and backup and backup.is_dir():
                    saved = backup / str(rel)
                    if saved.is_file():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(saved, target)
                        restored += 1
            except OSError:
                continue
        try:
            self.manifest_path.unlink(missing_ok=True)
        except OSError:
            pass
        if restore and backup and backup.is_dir():
            try:
                shutil.rmtree(backup, ignore_errors=True)
            except OSError:
                pass
        return {
            "ok": True,
            "removed": removed,
            "restored": restored,
            "status": self.status(),
        }
