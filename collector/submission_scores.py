"""
Reliable submission score fetcher.

Strategy (in order):
  1. Kaggle Python API  — competition_submissions() (most complete)
  2. kaggle CLI         — `kaggle competitions submissions -c <slug> --csv`
  3. Browser scraping   — /submissions page → __NEXT_DATA__ or table parse

Outputs:
  leaderboard/my_submissions.csv    (always written, even if empty)
  leaderboard/my_submissions.json
  leaderboard/_debug/submissions_cli_raw.txt  (if CLI used)
  leaderboard/_debug/submissions_page_raw.txt (if browser used)
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("kaggle_collector")

# ──────────────────────────────────────────────────────────────────────────────
# Canonical field schema for one submission row
# ──────────────────────────────────────────────────────────────────────────────
_FIELDS = [
    "submissionId",
    "fileName",
    "date",
    "publicScore",
    "privateScore",
    "status",
    "description",
    "errorDescription",
    "source",
]

# JS to extract submission data from the /submissions page __NEXT_DATA__
_SUB_NEXTDATA_JS = """
() => {
    const el = document.getElementById('__NEXT_DATA__');
    return el ? el.textContent : null;
}
"""

# JS to extract submission table rows if __NEXT_DATA__ misses them
_SUB_TABLE_JS = r"""
() => {
    const rows = [];
    document.querySelectorAll('table tr').forEach(tr => {
        const cells = Array.from(tr.querySelectorAll('td,th')).map(c => c.innerText.trim());
        if (cells.length >= 2) rows.push(cells);
    });
    return rows;
}
"""

_SUB_INNERTEXT_JS = """
() => {
    const el = document.getElementById('__NEXT_DATA__');
    return el ? el.textContent : (document.body.innerText || '');
}
"""


class SubmissionScoreFetcher:
    """Fetches the user's own submission history with public scores."""

    def __init__(self, slug: str, lb_dir: Path) -> None:
        self.slug = slug
        self.lb_dir = lb_dir
        self.lb_dir.mkdir(parents=True, exist_ok=True)
        self._debug_dir = lb_dir / "_debug"

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def fetch(self, page=None) -> dict[str, Any]:
        """
        Try every strategy in order. Returns a result dict with keys:
          rows        : list of submission dicts
          source      : which strategy worked
          warning     : any warning string (may be None)
        """
        # Strategy 1: Python API
        rows, source = self._via_python_api()
        if rows:
            self._log_score_status(rows, source)
            if not _all_scores_missing(rows):
                self._write_csv(rows)
                self._write_json(rows)
                return {"rows": rows, "source": source, "warning": None}

        # Strategy 2: kaggle CLI
        if not rows:
            rows, source = self._via_cli()

        if rows and not _all_scores_missing(rows):
            self._log_score_status(rows, source)
            self._write_csv(rows)
            self._write_json(rows)
            return {"rows": rows, "source": source, "warning": None}

        # Strategy 3: Browser
        if page is not None:
            browser_rows, browser_source = await self._via_browser(page)
            if browser_rows:
                rows, source = browser_rows, browser_source

        if rows:
            self._log_score_status(rows, source)
            self._write_csv(rows)
            self._write_json(rows)
            warning = None
            if _all_scores_missing(rows):
                warning = (
                    "publicScore is empty for all submissions. "
                    "Possible reasons: competition has not revealed public scores yet, "
                    "or submissions are still being evaluated."
                )
                logger.warning(f"  ⚠️  {warning}")
            return {"rows": rows, "source": source, "warning": warning}

        # All strategies failed — still write an empty CSV so downstream checks don't crash
        # Build a specific reason for the failure
        reasons: list[str] = []
        try:
            import importlib
            importlib.import_module("kaggle")
        except ImportError:
            reasons.append("kaggle Python package not installed (run: pip install kaggle)")
        if not reasons:
            reasons.append("API auth failed or no submissions found")
        reasons.append("browser fallback unavailable (playwright not loaded)")
        warning = (
            "Could not retrieve submission history. Reasons: "
            + "; ".join(reasons)
        )
        logger.warning(f"  ⚠️  {warning}")
        self._write_csv([])
        return {"rows": [], "source": "none", "warning": warning}

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy 1: Python API
    # ──────────────────────────────────────────────────────────────────────────

    def _via_python_api(self) -> tuple[list[dict], str]:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
            api = KaggleApi()
            api.authenticate()

            raw_subs = None
            try:
                raw_subs = api.competition_submissions(self.slug)
            except Exception:
                pass

            if raw_subs is None:
                try:
                    raw_subs = api.competitions_submissions_list(id=self.slug)
                except Exception:
                    pass

            if not raw_subs:
                return [], "api_empty"

            # Normalize response (may be a list or a response object)
            from .kaggle_api import _normalize_list
            items = _normalize_list(raw_subs) if not isinstance(raw_subs, list) else raw_subs

            rows = [_parse_api_submission(s) for s in items]
            logger.info(f"  API: {len(rows)} submissions found")
            return rows, "kaggle_python_api"

        except Exception as exc:
            logger.debug(f"  Python API submissions failed: {exc}")
            return [], "api_failed"

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy 2: kaggle CLI
    # ──────────────────────────────────────────────────────────────────────────

    def _via_cli(self) -> tuple[list[dict], str]:
        """Try CLI via module (-m kaggle) then fall back to kaggle binary."""
        # Build a list of command variants to try in order
        cmd_variants = [
            [sys.executable, "-m", "kaggle",
             "competitions", "submissions", "-c", self.slug, "--csv"],
            ["kaggle",
             "competitions", "submissions", "-c", self.slug, "--csv"],
        ]
        last_output = ""
        for cmd in cmd_variants:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                raw_output = (result.stdout or "") + (result.stderr or "")
                last_output = raw_output
                self._save_debug("submissions_cli_raw.txt", raw_output)

                if result.returncode != 0:
                    logger.debug(
                        f"  CLI submissions (cmd={cmd[0]}) rc={result.returncode}: "
                        f"{raw_output[:200]}"
                    )
                    continue  # try next variant

                # Strip leading log/warning lines (non-CSV) before parsing
                csv_text = _strip_non_csv_prefix(result.stdout)
                if not csv_text:
                    logger.debug(f"  CLI stdout empty after stripping non-CSV prefix")
                    continue

                rows = _parse_csv_output(csv_text)
                if rows:
                    logger.info(f"  CLI: {len(rows)} submissions found (cmd={cmd[0]})")
                    return rows, "kaggle_cli"
                # stdout was CSV-like but no data rows
                return [], "cli_empty"

            except FileNotFoundError:
                logger.debug(f"  CLI binary not found: {cmd[0]}")
            except Exception as exc:
                logger.debug(f"  CLI submissions failed: {exc}")

        self._save_debug("submissions_cli_raw.txt", last_output)
        return [], "cli_failed"

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy 3: Browser
    # ──────────────────────────────────────────────────────────────────────────

    async def _via_browser(self, page) -> tuple[list[dict], str]:
        url = f"https://www.kaggle.com/competitions/{self.slug}/submissions"
        try:
            await page.goto(url, wait_until="load", timeout=40_000)
            await page.wait_for_timeout(4_000)

            # Try __NEXT_DATA__ first
            raw_json: Optional[str] = await page.evaluate(_SUB_NEXTDATA_JS)
            if raw_json:
                # Save for debugging
                self._save_debug("submissions_page_raw.txt", raw_json[:5000])
                rows = _parse_nextdata_submissions(raw_json)
                if rows:
                    logger.info(f"  Browser/__NEXT_DATA__: {len(rows)} submissions found")
                    return rows, "browser_nextdata"

            # Fall back to table scrape
            table_rows: list[list[str]] = await page.evaluate(_SUB_TABLE_JS)
            if table_rows:
                rows = _parse_table_rows(table_rows)
                if rows:
                    logger.info(f"  Browser/table: {len(rows)} submissions found")
                    return rows, "browser_table"

            # Save full page text for debugging
            try:
                page_text = await page.evaluate("() => document.body.innerText")
                self._save_debug("submissions_page_raw.txt", (page_text or "")[:5000])
            except Exception:
                pass

            return [], "browser_empty"
        except Exception as exc:
            logger.debug(f"  Browser submissions failed: {exc}")
            return [], "browser_error"

    # ──────────────────────────────────────────────────────────────────────────
    # Writers
    # ──────────────────────────────────────────────────────────────────────────

    def _write_csv(self, rows: list[dict]) -> None:
        path = self.lb_dir / "my_submissions.csv"
        try:
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=_FIELDS, extrasaction="ignore")
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k, "") for k in _FIELDS})
            if rows:
                logger.info(f"  my_submissions.csv → {path}")
            else:
                logger.info(f"  my_submissions.csv → {path} (empty — no submissions found)")
        except Exception as exc:
            logger.error(f"  Could not write my_submissions.csv: {exc}")

    def _write_json(self, rows: list[dict]) -> None:
        path = self.lb_dir / "my_submissions.json"
        try:
            path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
            logger.info(f"  my_submissions.json → {path}")
        except Exception as exc:
            logger.error(f"  Could not write my_submissions.json: {exc}")

    def _log_score_status(self, rows: list[dict], source: str) -> None:
        scored = [r for r in rows if _has_score(r.get("publicScore"))]
        logger.info(
            f"  Submissions: {len(rows)} total, {len(scored)} with publicScore "
            f"(source={source})"
        )

    def _save_debug(self, filename: str, content: str) -> None:
        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            (self._debug_dir / filename).write_text(content, encoding="utf-8", errors="replace")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_api_submission(sub: Any) -> dict[str, Any]:
    """
    Extract fields from a Kaggle API submission object.
    The new kagglesdk uses snake_case: public_score, private_score, file_name, etc.
    Also handles older camelCase variants.
    """
    d: dict[str, Any] = {}

    # ID — new SDK uses 'ref', old used 'id' / 'submissionId'
    d["submissionId"] = (
        _get_attr(sub, "ref", "id", "submissionId", "submission_id") or ""
    )

    # File name — new SDK: file_name; old: fileName
    d["fileName"] = _get_attr(sub, "file_name", "fileName", "description", "ref") or ""

    # Date — new SDK: date; old: submissionDate / submitted_at
    raw_date = _get_attr(sub, "date", "submissionDate", "submitted_at")
    d["date"] = str(raw_date) if raw_date else ""

    # Public score — new SDK: public_score; old: publicScore / score
    d["publicScore"] = _coerce_score(
        _get_attr(sub, "public_score", "publicScore", "PublicScore", "score")
    )

    # Private score — new SDK: private_score; old: privateScore
    d["privateScore"] = _coerce_score(
        _get_attr(sub, "private_score", "privateScore", "PrivateScore")
    )

    # Status — may be an enum (SubmissionStatus.COMPLETE → strip prefix)
    raw_status = _get_attr(sub, "status", "evaluationStatus") or ""
    status_str = str(raw_status)
    if "." in status_str:
        status_str = status_str.split(".")[-1]
    d["status"] = status_str

    # Description — new SDK: description; old: description / fileName
    d["description"] = _get_attr(sub, "description", "fileName", "file_name") or ""

    # Error description — new SDK: error_description; old: errorDescription
    d["errorDescription"] = _get_attr(sub, "error_description", "errorDescription", "error") or ""

    d["source"] = "kaggle_python_api"
    return d


def _parse_csv_output(csv_text: str) -> list[dict]:
    """Parse the CSV output of `kaggle competitions submissions --csv`."""
    rows: list[dict] = []
    reader = csv.DictReader(csv_text.splitlines())
    for raw in reader:
        row: dict[str, Any] = {}
        for dest_key, src_keys in {
            "submissionId": ["ref", "id", "submissionId"],
            "fileName": ["fileName", "file_name", "description"],
            "date": ["date", "submissionDate", "submitted_at"],
            "publicScore": ["publicScore", "public_score", "score"],
            "privateScore": ["privateScore", "private_score"],
            "status": ["status"],
            "description": ["description", "fileName", "file_name"],
            "errorDescription": ["errorDescription", "error_description"],
        }.items():
            for sk in src_keys:
                if sk in raw and raw[sk] not in (None, "", "None"):
                    row[dest_key] = raw[sk]
                    break
            if dest_key not in row:
                row[dest_key] = ""
        row["publicScore"] = _coerce_score(row.get("publicScore"))
        row["privateScore"] = _coerce_score(row.get("privateScore"))
        row["source"] = "kaggle_cli"
        rows.append(row)
    return rows


def _parse_nextdata_submissions(raw_json: str) -> list[dict]:
    """Search __NEXT_DATA__ for a submissions list."""
    try:
        data = json.loads(raw_json)
    except Exception:
        return []

    found: list[Any] = []

    def _walk(obj: Any, depth: int = 0) -> None:
        if depth > 10 or found:
            return
        if isinstance(obj, list) and len(obj) > 0:
            first = obj[0]
            if isinstance(first, dict) and any(
                k.lower() in ("publicscore", "public_score", "score", "status")
                for k in first.keys()
            ):
                found.extend(obj)
                return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:50]:
                _walk(item, depth + 1)

    _walk(data)
    if not found:
        return []
    return [_normalise_nextdata_sub(s) for s in found if isinstance(s, dict)]


def _normalise_nextdata_sub(raw: dict) -> dict[str, Any]:
    row: dict[str, Any] = {}
    row["submissionId"] = _dict_lookup(raw, ["ref", "id", "submissionId", "submission_id"]) or ""
    row["fileName"] = _dict_lookup(raw, ["fileName", "file_name", "description"]) or ""
    row["date"] = str(_dict_lookup(raw, ["date", "submissionDate", "submitted_at"]) or "")
    row["publicScore"] = _coerce_score(
        _dict_lookup(raw, ["public_score", "publicScore", "PublicScore", "score"])
    )
    row["privateScore"] = _coerce_score(
        _dict_lookup(raw, ["private_score", "privateScore", "PrivateScore"])
    )
    row["status"] = _dict_lookup(raw, ["status", "evaluationStatus"]) or ""
    row["description"] = _dict_lookup(raw, ["description", "fileName"]) or ""
    row["errorDescription"] = _dict_lookup(raw, ["errorDescription", "error_description", "error"]) or ""
    row["source"] = "browser_nextdata"
    return row


def _parse_table_rows(table_rows: list[list[str]]) -> list[dict]:
    """Convert raw table rows (from browser scrape) to submission dicts."""
    if len(table_rows) < 2:
        return []
    header = [h.lower().strip() for h in table_rows[0]]
    rows: list[dict] = []
    for cells in table_rows[1:]:
        row: dict[str, Any] = {k: "" for k in _FIELDS}
        row["source"] = "browser_table"
        for i, cell in enumerate(cells):
            if i >= len(header):
                break
            h = header[i]
            if any(k in h for k in ("score", "public")):
                row["publicScore"] = _coerce_score(cell)
            elif "private" in h:
                row["privateScore"] = _coerce_score(cell)
            elif any(k in h for k in ("date", "time", "submit")):
                row["date"] = cell
            elif "status" in h:
                row["status"] = cell
            elif any(k in h for k in ("file", "name", "desc")):
                row["fileName"] = cell
        rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_attr(obj: Any, *names: str) -> Any:
    """Return the first non-None, non-empty attribute from the object."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None and str(val).strip() not in ("None", ""):
            return val
    return None


def _dict_lookup(d: dict, keys: list[str]) -> Any:
    """Case-insensitive dict lookup across multiple key variants."""
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip() not in ("None", ""):
            return v
    lower_map = {k2.lower(): v2 for k2, v2 in d.items()}
    for k in keys:
        v = lower_map.get(k.lower())
        if v is not None and str(v).strip() not in ("None", ""):
            return v
    return None


def _coerce_score(val: Any) -> str:
    """Convert a raw score value to a clean string, or empty string if absent."""
    if val is None:
        return ""
    s = str(val).strip()
    if s in ("None", "null", "nan", "NaN", ""):
        return ""
    return s


def _has_score(val: Any) -> bool:
    return bool(_coerce_score(val))


def _all_scores_missing(rows: list[dict]) -> bool:
    return all(not _has_score(r.get("publicScore")) for r in rows)


def _strip_non_csv_prefix(text: str) -> str:
    """
    Remove leading non-CSV lines (log messages, warnings, blank lines) from CLI output.
    The first CSV line is identified as the one starting with a letter and containing commas
    that looks like a header (e.g. 'fileName,date,description,...').
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # A CSV header or data row has commas and doesn't look like a log message
        if "," in stripped and not stripped.startswith("[") and not stripped.startswith("Warning"):
            return "\n".join(lines[i:])
    return ""
