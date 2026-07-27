"""
Run-level archiving, SHA-256 caching, and structured log management.

Creates:
  <base_dir>/<slug>/latest/                 live results
  <base_dir>/<slug>/archives/<timestamp>/   immutable snapshot
  <base_dir>/<slug>/archives/<run_id>.zip   portable archive

The physical archive directory intentionally uses only the timestamp. This keeps
Windows paths short while the ZIP and metadata retain the descriptive run_id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


_RUN_DIRS = ["pages", "screenshots", "discussions", "data_profile", "leaderboard"]
_CACHE_DIRS = ["data", "code_notebooks"]


class Archiver:
    """Central run-state manager."""

    def __init__(self, base_dir: Path, slug: str) -> None:
        self.output_root = base_dir
        self.slug = slug
        self.base_dir = base_dir / slug
        self.started_at = datetime.now()
        self.timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{slug}_{self.timestamp}"

        self.latest_dir = self.base_dir / "latest"
        self.latest_dir.mkdir(parents=True, exist_ok=True)

        self.archive_dir = self.base_dir / "archives" / self.timestamp

        self._log_lines: list[str] = []
        self._error_lines: list[str] = []
        self._warning_lines: list[str] = []

    def log(self, msg: str) -> None:
        self._log_lines.append(f"[{_ts()}] {msg}")

    def error(self, msg: str) -> None:
        self._error_lines.append(f"[{_ts()}] {msg}")

    def warning(self, msg: str) -> None:
        self._warning_lines.append(f"[{_ts()}] {msg}")

    def clean_run_dirs(
        self,
        clean_latest_completely: bool = False,
        keep_data: bool = True,
        keep_notebooks: bool = True,
    ) -> None:
        """Remove stale run output while optionally preserving expensive caches."""
        log = logging.getLogger("kaggle_collector")

        if clean_latest_completely:
            import tempfile

            log.info("--clean-latest: wiping entire latest/ directory …")
            preserved_tmp: dict[str, Path] = {}
            preserve_names = (
                (["data"] if keep_data else [])
                + (["code_notebooks"] if keep_notebooks else [])
            )

            for subdir_name in preserve_names:
                subdir = self.latest_dir / subdir_name
                if not subdir.exists():
                    continue
                tmp = Path(tempfile.mkdtemp(prefix=f"kcc_{subdir_name}_"))
                shutil.copytree(
                    _filesystem_path(subdir),
                    _filesystem_path(tmp / subdir_name),
                )
                preserved_tmp[subdir_name] = tmp

            if self.latest_dir.exists():
                shutil.rmtree(_filesystem_path(self.latest_dir))
            self.latest_dir.mkdir(parents=True, exist_ok=True)

            for name, tmp_root in preserved_tmp.items():
                shutil.copytree(
                    _filesystem_path(tmp_root / name),
                    _filesystem_path(self.latest_dir / name),
                )
                shutil.rmtree(_filesystem_path(tmp_root))

            log.info(f"  latest/ recreated (preserved: {list(preserved_tmp)})")
        else:
            for dir_name in _RUN_DIRS:
                target = self.latest_dir / dir_name
                if target.exists():
                    shutil.rmtree(_filesystem_path(target))
                    log.debug(f"  Cleaned stale: latest/{dir_name}/")

            if not keep_data:
                target = self.latest_dir / "data"
                if target.exists():
                    shutil.rmtree(_filesystem_path(target))
                    log.debug("  Cleaned cache: latest/data/")

            if not keep_notebooks:
                target = self.latest_dir / "code_notebooks"
                if target.exists():
                    shutil.rmtree(_filesystem_path(target))
                    log.debug("  Cleaned cache: latest/code_notebooks/")

        self.write_run_id_marker()

    def write_run_id_marker(self) -> None:
        (self.latest_dir / ".run_id").write_text(self.run_id, encoding="utf-8")

    def run_id_from_dir(self) -> str:
        marker = self.latest_dir / ".run_id"
        if marker.exists():
            return marker.read_text(encoding="utf-8").strip()
        return ""

    def sha256_of(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(_filesystem_path(path), "rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                h.update(chunk)
        return h.hexdigest()

    def file_unchanged(self, path: Path) -> bool:
        cache_file = path.with_suffix(path.suffix + ".sha256")
        if not path.exists() or not cache_file.exists():
            return False
        stored = cache_file.read_text(encoding="utf-8").strip()
        return stored == self.sha256_of(path)

    def record_hash(self, path: Path) -> None:
        cache_file = path.with_suffix(path.suffix + ".sha256")
        cache_file.write_text(self.sha256_of(path), encoding="utf-8")

    def finalize(
        self,
        counters: dict[str, Any],
        pkg_versions: dict[str, str],
        run_config: dict[str, Any] | None = None,
    ) -> Path:
        """Write metadata, AI prompt, immutable snapshot, and ZIP archive."""
        self._flush_log_files()
        self._write_run_metadata(counters, pkg_versions, run_config or {})
        self._write_ai_prompt()

        log = logging.getLogger("kaggle_collector")
        archive_copy_ok = False

        try:
            if self.archive_dir.exists():
                shutil.rmtree(_filesystem_path(self.archive_dir))
            self.archive_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                _filesystem_path(self.latest_dir),
                _filesystem_path(self.archive_dir),
                ignore=_ignore_sha256,
            )
            archive_copy_ok = True
        except Exception as exc:
            warning = (
                "Archive directory copy could not be completed; the live latest/ "
                f"output is intact and ZIP creation will still be attempted. Reason: {exc}"
            )
            log.warning(warning)
            self.warning(warning)

        self._create_run_zip(source_dir=self.latest_dir)
        return self.archive_dir if archive_copy_ok else self.latest_dir

    def _write_ai_prompt(self) -> None:
        try:
            from collector.ai_prompt import write_ai_analysis_prompt

            path = write_ai_analysis_prompt(self.latest_dir, self.slug)
            logging.getLogger("kaggle_collector").info(
                f"Ready-to-use AI prompt written: {path.name}"
            )
        except Exception as exc:
            warning = f"Could not create ready-to-use AI prompt: {exc}"
            logging.getLogger("kaggle_collector").warning(warning)
            self.warning(warning)

    def _create_run_zip(self, source_dir: Path) -> Path | None:
        archive_root = self.base_dir / "archives"
        archive_root.mkdir(parents=True, exist_ok=True)
        zip_path = archive_root / f"{self.run_id}.zip"

        if zip_path.exists():
            zip_path.unlink()

        try:
            source_fs = Path(_filesystem_path(source_dir))
            with zipfile.ZipFile(
                _filesystem_path(zip_path),
                "w",
                zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as zf:
                for file in sorted(source_fs.rglob("*")):
                    if not file.is_file() or file.name.endswith(".sha256"):
                        continue
                    relative = file.relative_to(source_fs).as_posix()
                    zf.write(_filesystem_path(file), f"{self.run_id}/{relative}")
            return zip_path
        except Exception as exc:
            warning = f"Could not create archive zip: {exc}"
            logging.getLogger("kaggle_collector").warning(warning)
            self.warning(warning)
            return None

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
            "archive_zip": str(self.base_dir / "archives" / f"{self.run_id}.zip"),
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


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _ignore_sha256(directory: str, contents: list[str]) -> list[str]:
    return [name for name in contents if name.endswith(".sha256")]


def _filesystem_path(path: Path | str) -> str:
    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def collect_pkg_versions() -> dict[str, str]:
    packages = [
        "kaggle",
        "playwright",
        "python-dotenv",
        "markdownify",
        "beautifulsoup4",
        "pandas",
        "pyarrow",
        "pyyaml",
        "openpyxl",
        "tabulate",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            import importlib.metadata as im

            versions[pkg] = im.version(pkg)
        except Exception:
            versions[pkg] = "not installed"
    return versions
