"""Inspect Kaggle competition data size without downloading the files."""

from __future__ import annotations

import re
from typing import Any

from collector.kaggle_api import _normalize_list


class DatasetInspectionError(RuntimeError):
    """Raised when competition file metadata cannot be inspected."""


def parse_size_bytes(value: Any) -> int:
    """Convert Kaggle SDK size values or human-readable strings to bytes."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))

    text = str(value).strip().replace(",", "")
    if not text:
        return 0

    try:
        return max(0, int(float(text)))
    except ValueError:
        pass

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgtpe]?i?b?)", text, re.IGNORECASE)
    if not match:
        return 0

    amount = float(match.group(1))
    unit = match.group(2).lower().rstrip("b")
    unit = unit.rstrip("i")
    powers = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6}
    return max(0, int(amount * (1024 ** powers.get(unit, 0))))


def format_bytes(size_bytes: int) -> str:
    """Format a byte count using binary units."""
    value = float(max(0, size_bytes))
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            decimals = 0 if unit == "B" else 2
            return f"{value:.{decimals}f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def _next_page_token(response: Any) -> str | None:
    for attr in ("next_page_token", "nextPageToken"):
        value = getattr(response, attr, None)
        if value:
            return str(value)

    if hasattr(response, "to_dict"):
        try:
            payload = response.to_dict()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            value = payload.get("next_page_token") or payload.get("nextPageToken")
            if value:
                return str(value)
    return None


def _serialize_file(file_obj: Any) -> dict[str, Any]:
    size_raw = getattr(file_obj, "total_bytes", None)
    if size_raw is None:
        size_raw = getattr(file_obj, "size", None)

    creation_date = getattr(file_obj, "creation_date", None)
    if creation_date is None:
        creation_date = getattr(file_obj, "creationDate", None)

    size_bytes = parse_size_bytes(size_raw)
    return {
        "name": getattr(file_obj, "name", str(file_obj)),
        "size_bytes": size_bytes,
        "size_display": format_bytes(size_bytes),
        "creation_date": str(creation_date) if creation_date else "",
    }


def inspect_competition_data(slug: str, page_size: int = 1000) -> dict[str, Any]:
    """Return file count and approximate extracted size without downloading data."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        raise DatasetInspectionError(f"Kaggle API oturumu açılamadı: {exc}") from exc

    collected: dict[str, dict[str, Any]] = {}
    page_token: str | None = None

    for _ in range(100):
        try:
            kwargs: dict[str, Any] = {"page_size": page_size}
            if page_token:
                kwargs["page_token"] = page_token
            response = api.competition_list_files(slug, **kwargs)
        except TypeError:
            # Compatibility fallback for older Kaggle API package versions.
            try:
                response = api.competition_list_files(slug)
            except Exception as exc:
                raise DatasetInspectionError(_friendly_api_error(slug, exc)) from exc
        except Exception as exc:
            raise DatasetInspectionError(_friendly_api_error(slug, exc)) from exc

        files = _normalize_list(response)
        for file_obj in files:
            item = _serialize_file(file_obj)
            collected[item["name"]] = item

        next_token = _next_page_token(response)
        if not next_token or next_token == page_token:
            break
        page_token = next_token

    files = sorted(collected.values(), key=lambda item: item["size_bytes"], reverse=True)
    if not files:
        raise DatasetInspectionError(
            "Dosya listesi alınamadı. Yarışma kurallarını kabul ettiğinizi ve Kaggle erişiminizi kontrol edin."
        )

    total_bytes = sum(item["size_bytes"] for item in files)
    return {
        "slug": slug,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_display": format_bytes(total_bytes),
        "largest_file": files[0] if files else None,
        "files": files,
    }


def _friendly_api_error(slug: str, exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "403" in text or "forbidden" in lowered or "rules" in lowered or "accept" in lowered:
        return (
            "Yarışma dosyalarına erişilemiyor. Önce Kaggle yarışma kurallarını kabul edin: "
            f"https://www.kaggle.com/competitions/{slug}/rules"
        )
    return f"Yarışma veri boyutu alınamadı: {text}"
