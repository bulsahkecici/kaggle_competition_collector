"""
Kaggle API / CLI-based collector.

Responsibilities:
- Authenticate via ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY env vars
- Fetch competition metadata through the kaggle Python package
- List and download competition data files
- Fall back to the `kaggle` CLI when the Python API raises an error
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("kaggle_collector")


def _normalize_list(response: Any) -> list:
    """
    Normalize a Kaggle SDK response into a plain Python list.

    Handles:
      - plain list
      - object with .competitions / .files / .submissions / .results / .data attribute
      - object with .to_dict() containing any of the above keys
      - object with __dict__
    """
    if response is None:
        return []
    if isinstance(response, list):
        return response

    # Try common attribute names used by kagglesdk response objects
    for attr in ("competitions", "files", "submissions", "results", "data", "items"):
        val = getattr(response, attr, None)
        if isinstance(val, list):
            return val

    # Try to_dict
    if hasattr(response, "to_dict"):
        try:
            d = response.to_dict()
            if isinstance(d, dict):
                for key in ("competitions", "files", "submissions", "results", "data", "items"):
                    val = d.get(key)
                    if isinstance(val, list):
                        return val
        except Exception:
            pass

    # Try __dict__
    if hasattr(response, "__dict__"):
        for val in vars(response).values():
            if isinstance(val, list) and val:
                return val

    return []


def _get_snake(obj: Any, *names: str) -> Any:
    """Try both camelCase and snake_case attribute variants."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
        # try snake_case conversion
        snake = "".join(
            "_" + c.lower() if c.isupper() else c for c in name
        ).lstrip("_")
        val = getattr(obj, snake, None)
        if val is not None:
            return val
    return None


class KaggleAPICollector:
    """Collects competition metadata and data files using the Kaggle API/CLI."""

    def __init__(self, slug: str, output_dir: Path) -> None:
        self.slug = slug
        self.output_dir = output_dir
        self.data_dir = output_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def collect(self, download_data: bool = True) -> dict[str, Any]:
        """Run all API-based collection steps and return a results dict."""
        result: dict[str, Any] = {}

        api = self._try_authenticate()
        if api is not None:
            logger.info("Kaggle API authenticated successfully")

            metadata = self._get_competition_metadata(api)
            if metadata:
                result["metadata"] = metadata
                self._save_json(metadata, self.output_dir / "metadata.json")
                logger.info("metadata.json saved")

            files_manifest = self._get_files_manifest(api)
            result["files_manifest"] = files_manifest
            if files_manifest:
                self._save_json(files_manifest, self.output_dir / "files_manifest.json")
                logger.info("files_manifest.json saved")

            if download_data:
                result["data_downloaded"] = self._download_data_api(api)
        else:
            logger.warning("Kaggle Python API unavailable; trying CLI fallback")
            if download_data:
                result["data_downloaded"] = self._download_data_cli()

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Authentication
    # ──────────────────────────────────────────────────────────────────────────

    def _try_authenticate(self):
        """Return an authenticated KaggleApi instance, or None on failure."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
            api = KaggleApi()
            api.authenticate()
            return api
        except ImportError:
            logger.warning("kaggle package not installed; skipping Python API")
        except Exception as exc:
            logger.warning(f"Kaggle API authentication failed: {exc}")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Metadata
    # ──────────────────────────────────────────────────────────────────────────

    def _get_competition_metadata(self, api) -> dict[str, Any]:
        """Fetch competition-level metadata from the Kaggle REST API."""
        try:
            response = api.competitions_list(search=self.slug)
            competitions = _normalize_list(response)

            for comp in competitions:
                ref = getattr(comp, "ref", "") or ""
                # ref may be a full URL like https://www.kaggle.com/competitions/slug
                if ref == self.slug or ref.endswith("/" + self.slug):
                    return self._serialize_competition(comp)

            # Slug not found exactly — use first hit as fallback
            if competitions:
                logger.warning(
                    f"Exact slug '{self.slug}' not found in list results; using closest match"
                )
                return self._serialize_competition(competitions[0])
        except Exception as exc:
            logger.error(f"Failed to fetch competition metadata: {exc}")
        return {}

    def _serialize_competition(self, comp) -> dict[str, Any]:
        """Serialize a competition object to a plain dict, trying both naming styles."""
        field_map = [
            ("ref",                    "ref"),
            ("title",                  "title"),
            ("url",                    "url"),
            ("description",            "description"),
            ("category",               "category"),
            ("reward",                 "reward"),
            # New SDK uses snake_case; old used camelCase — try both
            ("team_count",             "teamCount"),
            ("teamCount",              "teamCount"),
            ("user_has_entered",       "userHasEntered"),
            ("userHasEntered",         "userHasEntered"),
            ("user_rank",              "userRank"),
            ("userRank",               "userRank"),
            ("evaluation_metric",      "evaluationMetric"),
            ("evaluationMetric",       "evaluationMetric"),
            ("is_kernels_submissions_only", "isKernelsSubmissionsOnly"),
            ("isKernelsSubmissionsOnly",    "isKernelsSubmissionsOnly"),
            ("enabled_date",           "enabledDate"),
            ("enabledDate",            "enabledDate"),
            ("deadline",               "deadline"),
            ("merger_deadline",        "mergerDeadline"),
            ("mergerDeadline",         "mergerDeadline"),
            ("new_entrant_deadline",   "newEntrantDeadline"),
            ("newEntrantDeadline",     "newEntrantDeadline"),
            ("max_daily_submissions",  "maxDailySubmissions"),
            ("maxDailySubmissions",    "maxDailySubmissions"),
            ("max_team_size",          "maxTeamSize"),
            ("maxTeamSize",            "maxTeamSize"),
            ("organization_name",      "organizationName"),
            ("organizationName",       "organizationName"),
            ("organization_ref",       "organizationRef"),
            ("organizationRef",        "organizationRef"),
            ("tags",                   "tags"),
        ]
        result: dict[str, Any] = {}
        seen_dest: set[str] = set()
        for attr, dest_key in field_map:
            if dest_key in seen_dest:
                continue
            val = getattr(comp, attr, None)
            if val is not None:
                result[dest_key] = val
                seen_dest.add(dest_key)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Files manifest
    # ──────────────────────────────────────────────────────────────────────────

    def _get_files_manifest(self, api) -> list[dict[str, Any]]:
        """List all data files available for the competition."""
        try:
            response = api.competition_list_files(self.slug)
            files = _normalize_list(response)
            manifest = []
            for f in files:
                # New SDK: total_bytes; old SDK: size
                size = getattr(f, "total_bytes", None) or getattr(f, "size", None)
                # New SDK: creation_date; old SDK: creationDate
                cdate = getattr(f, "creation_date", None) or getattr(f, "creationDate", None)
                manifest.append({
                    "name": getattr(f, "name", str(f)),
                    "size": size,
                    "creationDate": str(cdate) if cdate else "",
                })
            return manifest
        except Exception as exc:
            error_lower = str(exc).lower()
            if "403" in str(exc) or "rules" in error_lower or "forbidden" in error_lower:
                logger.warning(
                    "⚠️  Cannot list files – you may need to accept competition rules.\n"
                    f"   Visit: https://www.kaggle.com/competitions/{self.slug}/rules"
                )
            else:
                logger.error(f"Failed to list competition files: {exc}")
        return []

    # ──────────────────────────────────────────────────────────────────────────
    # Data download – Python API
    # ──────────────────────────────────────────────────────────────────────────

    def _download_data_api(self, api) -> bool:
        """Download and unzip competition data using the kaggle Python API."""
        logger.info(f"Downloading competition data → {self.data_dir}")
        try:
            api.competition_download_files(
                self.slug,
                path=str(self.data_dir),
                quiet=False,
                force=False,
            )
            self._unzip_in_dir(self.data_dir)
            logger.info("Data download complete")
            return True
        except Exception as exc:
            self._handle_download_error(exc)
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Data download – CLI fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _download_data_cli(self) -> bool:
        """Download competition data using the `kaggle` CLI as a subprocess."""
        logger.info("Attempting data download via kaggle CLI …")
        cmd = [
            sys.executable, "-m", "kaggle",
            "competitions", "download",
            "-c", self.slug,
            "-p", str(self.data_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            self._unzip_in_dir(self.data_dir)
            logger.info("CLI download complete")
            return True
        self._handle_download_error_text(output)
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _unzip_in_dir(self, directory: Path) -> None:
        """Unzip all .zip files found in directory (in-place)."""
        for zip_path in directory.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(directory)
                zip_path.unlink()
                logger.debug(f"Unzipped and removed: {zip_path.name}")
            except Exception as exc:
                logger.warning(f"Could not unzip {zip_path.name}: {exc}")

    def _handle_download_error(self, exc: Exception) -> None:
        error_text = str(exc)
        self._handle_download_error_text(error_text)

    def _handle_download_error_text(self, text: str) -> None:
        lower = text.lower()
        if "403" in text or "forbidden" in lower or "rules" in lower or "accept" in lower:
            logger.warning(
                "⚠️  Data download blocked – competition rules acceptance required.\n"
                f"   Visit: https://www.kaggle.com/competitions/{self.slug}/rules\n"
                "   Accept the rules there, then re-run this tool."
            )
        else:
            logger.error(f"Data download failed:\n{text.strip()}")

    @staticmethod
    def _save_json(data: Any, path: Path) -> None:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
