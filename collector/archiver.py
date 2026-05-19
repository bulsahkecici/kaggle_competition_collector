"""
Run-level archiving, SHA-256 caching, and structured log management.

Creates:
  <base_dir>/<slug>/latest/          ← live results for this competition
  <base_dir>/<slug>/archives/<slug>_YYYYMMDD_HHMMSS/  ← immutable snapshot

All collectors write to <slug>/latest/.  At the end, finalize() copies latest/ →
<slug>/archives/ and writes run_metadata.json, RUN_LOG.md, ERRORS.md, WARNINGS.md.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


# Directories that hold per-run output and must be cleaned before a fresh run
_RUN_DIRS = ["pages", "screenshots", "discussions", "data_profile", "leaderboard"]

# Directories that cache expensive downloads — only deleted with --clean-latest
_CACHE_DIRS = ["data", "code_notebooks"]


class Archiver:
    """Central run-state manager."""

    def __init__(self, base_dir: Path, slug: str) -> None:
        self.output_root = base_dir
        self.slug = slug
        self.base_dir = base_dir / slug
        self.started_at = datetime.now()
        self.timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        # run_id is the stable identifier for this run (used in headers of every file)
        self.run_id = f"{slug}_{self.timestamp}"

        self.latest_dir = self.base_dir / "latest"
        self.latest_dir.mkdir(parents=True, exist_ok=True)

        self.archive_dir = self.base_dir / "archives" / f"{slug}_{self.timestamp}"

        self._log_lines: list[str] = []
        self._error_lines: list[str] = []
        self._warning_lines: list[str] = []

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, msg: str) -> None:
        entry = f"[{_ts()}] {msg}"
        self._log_lines.append(entry)

    def error(self, msg: str) -> None:
        entry = f"[{_ts()}] {msg}"
        self._error_lines.append(entry)

    def warning(self, msg: str) -> None:
        entry = f"[{_ts()}] {msg}"
        self._warning_lines.append(entry)

    # ── Run directory management ──────────────────────────────────────────────

    def clean_run_dirs(
        self,
        clean_latest_completely: bool = False,
        keep_data: bool = True,
        keep_notebooks: bool = True,
    ) -> None:
        """
        Remove stale output from a previous run inside latest/.

        Parameters
        ----------
        clean_latest_completely : If True, wipe the entire latest/ directory
                                   (used with --clean-latest).
        keep_data               : Keep latest/data/ (expensive re-download).
        keep_notebooks          : Keep latest/code_notebooks/ (notebooks are cached per-slug).
        """
        import logging as _logging
        log = _logging.getLogger("kaggle_collector")

        if clean_latest_completely:
            import tempfile
            log.info("--clean-latest: wiping entire latest/ directory …")
            # Move precious subdirs to a temp location BEFORE wiping latest/
            preserved_tmp: dict[str, Path] = {}
            preserve_names = (["data"] if keep_data else []) + (["code_notebooks"] if keep_notebooks else [])
            for subdir_name in preserve_names:
                subdir = self.latest_dir / subdir_name
                if subdir.exists():
                    tmp = Path(tempfile.mkdtemp(prefix=f"kcc_{subdir_name}_"))
                    shutil.copytree(subdir, tmp / subdir_name)
                    preserved_tmp[subdir_name] = tmp
            # Wipe and recreate
            shutil.rmtree(self.latest_dir)
            self.latest_dir.mkdir(parents=True, exist_ok=True)
            # Move preserved directories back
            for name, tmp_root in preserved_tmp.items():
                dst = self.latest_dir / name
                shutil.copytree(tmp_root / name, dst)
                shutil.rmtree(tmp_root)
            log.info(f"  latest/ recreated (preserved: {list(preserved_tmp)})")
        else:
            # Default: clean run-generated directories. Cached directories are
            # preserved unless the caller explicitly disables keeping them
            # (e.g. --overwrite-cache for code_notebooks).
            for dir_name in _RUN_DIRS:
                target = self.latest_dir / dir_name
                if target.exists():
                    shutil.rmtree(target)
                    log.debug(f"  Cleaned stale: latest/{dir_name}/")
            if not keep_data:
                target = self.latest_dir / "data"
                if target.exists():
                    shutil.rmtree(target)
                    log.debug("  Cleaned cache: latest/data/")
            if not keep_notebooks:
                target = self.latest_dir / "code_notebooks"
                if target.exists():
                    shutil.rmtree(target)
                    log.debug("  Cleaned cache: latest/code_notebooks/")

        # Write a .run_id marker so quality validator can detect stale files
        self.write_run_id_marker()

    def write_run_id_marker(self) -> None:
        """Write .run_id into latest/ so any leftover file can be compared."""
        marker = self.latest_dir / ".run_id"
        marker.write_text(self.run_id, encoding="utf-8")

    def run_id_from_dir(self) -> str:
        """Read the run_id written by the previous invocation (or '' if absent)."""
        marker = self.latest_dir / ".run_id"
        if marker.exists():
            return marker.read_text(encoding="utf-8").strip()
        return ""

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def sha256_of(self, path: Path) -> str:
        """Return hex SHA-256 of a file."""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                h.update(chunk)
        return h.hexdigest()

    def file_unchanged(self, path: Path) -> bool:
        """
        True when `path` already exists in latest/ and its content is identical
        to what was there from the previous run (stored alongside as .sha256).
        """
        cache_file = path.with_suffix(path.suffix + ".sha256")
        if not path.exists() or not cache_file.exists():
            return False
        stored = cache_file.read_text(encoding="utf-8").strip()
        return stored == self.sha256_of(path)

    def record_hash(self, path: Path) -> None:
        """Persist the SHA-256 of `path` next to it for future cache checks."""
        cache_file = path.with_suffix(path.suffix + ".sha256")
        cache_file.write_text(self.sha256_of(path), encoding="utf-8")

    # ── Finalization ──────────────────────────────────────────────────────────

    def finalize(
        self,
        counters: dict[str, Any],
        pkg_versions: dict[str, str],
        run_config: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write log / metadata files into latest/, copy everything to
        archives/<slug>_<timestamp>/, then create a zip of ONLY that folder.
        Returns the archive directory path.
        """
        self._flush_log_files()
        self._write_run_metadata(counters, pkg_versions, run_config or {})

        if self.archive_dir.exists():
            shutil.rmtree(self.archive_dir)
        shutil.copytree(self.latest_dir, self.archive_dir, ignore=_ignore_sha256)

        # Create a zip containing ONLY this run's archive folder
        self._create_run_zip()

        return self.archive_dir

    def _create_run_zip(self) -> None:
        """
        Zip archives/<slug>_<timestamp>/ into archives/<slug>_<timestamp>.zip.
        The zip contains a single top-level folder named <slug>_<timestamp>/.
        Any pre-existing .zip files with a conflicting name are removed first.
        """
        zip_path = self.archive_dir.parent / f"{self.archive_dir.name}.zip"

        # Remove any existing zip with this name (could be stale from a previous run)
        if zip_path.exists():
            zip_path.unlink()

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                for file in sorted(self.archive_dir.rglob("*")):
                    if file.is_file():
                        arcname = self.archive_dir.name + "/" + file.relative_to(self.archive_dir).as_posix()
                        zf.write(file, arcname)
        except Exception as exc:
            # Zip creation is best-effort — don't crash the whole run
            import logging
            logging.getLogger("kaggle_collector").warning(f"Could not create archive zip: {exc}")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _flush_log_files(self) -> None:
        _write(
            self.latest_dir / "RUN_LOG.md",
            f"# Run Log — {self.slug} — {self.timestamp}\n\n"
            + "\n".join(self._log_lines or ["(empty)"]),
        )
        _write(
            self.latest_dir / "ERRORS.md",
            f"# Errors — {self.slug} — {self.timestamp}\n\n"
            + "\n".join(self._error_lines or ["No errors encountered."]),
        )
        _write(
            self.latest_dir / "WARNINGS.md",
            f"# Warnings — {self.slug} — {self.timestamp}\n\n"
            + "\n".join(self._warning_lines or ["No warnings."]),
        )

    def _write_run_metadata(
        self,
        counters: dict[str, Any],
        pkg_versions: dict[str, str],
        run_config: dict[str, Any] | None = None,
    ) -> None:
        meta: dict[str, Any] = {
            "run_id": self.run_id,
            "run_timestamp": self.timestamp,
            "started_at": self.started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "competition_slug": self.slug,
            "competition_dir": str(self.base_dir),
            "output_latest": str(self.latest_dir),
            "archive": str(self.archive_dir),
            "machine": {
                "os": platform.platform(),
                "python_version": sys.version,
                "python_executable": sys.executable,
            },
            "package_versions": pkg_versions,
            "run_config": run_config or {},
            "counters": counters,
        }
        _write(
            self.latest_dir / "run_metadata.json",
            json.dumps(meta, indent=2, default=str),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _ignore_sha256(directory: str, contents: list[str]) -> list[str]:
    """Exclude .sha256 sidecar files from the archive copy."""
    return [f for f in contents if f.endswith(".sha256")]


def collect_pkg_versions() -> dict[str, str]:
    """Return installed versions of key packages."""
    packages = [
        "kaggle", "playwright", "python-dotenv",
        "markdownify", "beautifulsoup4", "pandas",
        "pyarrow", "pyyaml", "openpyxl", "tabulate",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            import importlib.metadata as im
            versions[pkg] = im.version(pkg)
        except Exception:
            versions[pkg] = "not installed"
    return versions
