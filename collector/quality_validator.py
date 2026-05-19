"""
Collection quality validator.

After all phases complete, inspects every key output file and generates:
  collection_quality_report.md

Status values per item:
  OK       — file exists, non-empty, passes content checks
  WARNING  — file exists but content may be incomplete / suspicious
  FAILED   — file missing or known-bad content
  SKIPPED  — collection was intentionally skipped for this item
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .page_extractor import is_crash_or_garbage

logger = logging.getLogger("kaggle_collector")

# Minimum char counts to consider a page file "real" (key pages get full validation)
_PAGE_MIN_CHARS: dict[str, int] = {
    "overview":   200,
    "evaluation": 100,
    "rules":      100,
    "data":       100,
}

# All pages get crash detection, with lower bar
_ALL_PAGE_NAMES = ["overview", "data", "evaluation", "rules",
                   "leaderboard", "discussion", "code"]

# Phrases that indicate a page failed to load properly
_CRASH_STRINGS = [
    "something went wrong and this page crashed",
    "unexpected token '<'",
    "content extraction failed",
    "cloudflare",
    "navigation failed",
    "page appears to be a crash",
    "err_internet_disconnected",
    "err_connection_timed_out",
    "an unexpected error has occurred",
    "just a moment",
    "checking your browser",
]


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass (plain class to keep it lightweight)
# ──────────────────────────────────────────────────────────────────────────────

class CheckResult:
    def __init__(
        self,
        item: str,
        status: str,          # OK | WARNING | FAILED | SKIPPED
        details: str,
        path: Optional[Path] = None,
    ) -> None:
        self.item = item
        self.status = status
        self.details = details
        self.path = path

    @property
    def emoji(self) -> str:
        return {
            "OK": "✅",
            "WARNING": "⚠️ ",
            "FAILED": "❌",
            "SKIPPED": "⬜",
        }.get(self.status, "❔")


class QualityValidator:
    """Runs all validation checks and writes collection_quality_report.md."""

    def __init__(
        self,
        output_dir: Path,
        browser_data: dict[str, Any],
        metric_result: dict[str, Any],
        lb_result: dict[str, Any],
        data_profiles: dict[str, Any],
        archiver: Any,          # Archiver instance for warning/error logging
    ) -> None:
        self.out = output_dir
        self.browser = browser_data
        self.metric = metric_result
        self.lb = lb_result
        self.profiles = data_profiles
        self.archiver = archiver

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def validate(self) -> list[CheckResult]:
        """Run all checks. Returns list of CheckResult objects."""
        results: list[CheckResult] = []

        results.extend(self._check_pages())
        results.extend(self._check_extra_pages())
        results.append(self._check_screenshots())
        results.append(self._check_metric())
        results.extend(self._check_data_files())
        results.append(self._check_submission_format())
        results.extend(self._check_submissions())
        results.append(self._check_leaderboard())
        results.append(self._check_notebooks())
        results.append(self._check_discussions())
        results.append(self._check_summary_report())

        self._write_report(results)
        self._propagate_to_archiver(results)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Checks
    # ──────────────────────────────────────────────────────────────────────────

    def _check_pages(self) -> list[CheckResult]:
        """Validate pages/overview.md, evaluation.md, rules.md, data.md."""
        checks: list[CheckResult] = []
        for tab_name, min_chars in _PAGE_MIN_CHARS.items():
            path = self.out / "pages" / f"{tab_name}.md"
            label = f"pages/{tab_name}.md"

            if not path.exists():
                tab_data = self.browser.get(tab_name, {})
                if isinstance(tab_data, dict) and tab_data.get("error") == "404 — tab not available":
                    checks.append(CheckResult(label, "SKIPPED", "Tab returned 404 (not available for this competition)", path))
                else:
                    checks.append(CheckResult(label, "FAILED", "File does not exist", path))
                continue

            text = path.read_text(encoding="utf-8", errors="replace")

            # ── Parse the quality metadata written by page_extractor ──────────
            header_meta = _parse_md_header(text)
            file_quality = header_meta.get("quality", "")  # 'good' | 'acceptable' | 'failed'

            # If the file itself reports failed extraction, never mark as OK
            if file_quality == "failed" or _has_crash_string(text):
                checks.append(self._page_failed_result(label, tab_name, path))
                continue

            content = _strip_header(text)

            if len(content.strip()) < min_chars:
                checks.append(CheckResult(
                    label, "FAILED",
                    f"Content too short ({len(content.strip())} chars, min={min_chars})",
                    path,
                ))
                continue

            if is_crash_or_garbage(content):
                checks.append(self._page_failed_result(label, tab_name, path))
                continue

            # File looks good — check for soft warnings
            tab_meta = self.browser.get(tab_name, {})
            if isinstance(tab_meta, dict):
                if tab_meta.get("requires_acceptance"):
                    checks.append(CheckResult(
                        label, "WARNING",
                        f"OK ({len(content.strip()):,} chars) but rule acceptance required — content may be incomplete",
                        path,
                    ))
                    continue
                if file_quality == "acceptable":
                    checks.append(CheckResult(
                        label, "WARNING",
                        f"Extracted via innerText fallback ({len(content.strip()):,} chars) — may contain noise",
                        path,
                    ))
                    continue

            strategy = header_meta.get("method", tab_meta.get("strategy", "?") if isinstance(tab_meta, dict) else "?")
            checks.append(CheckResult(
                label, "OK",
                f"{len(content.strip()):,} chars — strategy: {strategy}",
                path,
            ))
        return checks

    def _page_failed_result(self, label: str, tab_name: str, path) -> CheckResult:
        """
        Return the appropriate WARN/FAIL result for a page that failed browser extraction.
        WARN when a meaningful API fallback exists; FAIL when the info is truly missing.
        """
        metric_source = self.metric.get("source", "")
        metric_ok = bool(self.metric.get("raw_metric"))

        if tab_name == "evaluation":
            if metric_ok:
                return CheckResult(
                    label, "WARNING",
                    f"Browser extraction failed (Kaggle crash/Cloudflare page) — "
                    f"evaluation metric recovered from {metric_source} (ROC AUC). "
                    "See pages/_debug/evaluation.raw.txt for the raw page snippet.",
                    path,
                )
            return CheckResult(
                label, "FAILED",
                "Browser extraction failed and metric not available from API. "
                "Re-run with --headed to attempt manual extraction.",
                path,
            )

        if tab_name == "overview":
            api_has_meta = bool(self.metric.get("raw_metric"))
            return CheckResult(
                label, "WARNING",
                "Browser extraction failed (Kaggle crash/Cloudflare page) — "
                "competition title and metric are available from API. "
                "Full overview text unavailable." if api_has_meta else
                "Browser extraction failed — overview content unavailable.",
                path,
            )

        if tab_name == "rules":
            return CheckResult(
                label, "WARNING",
                "Browser extraction failed (Kaggle crash/Cloudflare page) — "
                "rules page unavailable; some competition rules may be incomplete. "
                "See pages/_debug/rules.raw.txt.",
                path,
            )

        if tab_name == "data":
            files_ok = bool((self.out / "data").exists() and any((self.out / "data").iterdir()))
            if files_ok:
                return CheckResult(
                    label, "WARNING",
                    "Browser extraction failed (Kaggle crash/Cloudflare page) — "
                    "data description page unavailable, but data files were downloaded successfully.",
                    path,
                )
            return CheckResult(
                label, "FAILED",
                "Browser extraction failed and no data files were downloaded.",
                path,
            )

        return CheckResult(
            label, "WARNING",
            "Browser extraction failed (Kaggle crash/Cloudflare page) — "
            "content unavailable. See pages/_debug/ for raw page snapshot.",
            path,
        )

    def _check_metric(self) -> CheckResult:
        """Check whether the evaluation metric was successfully parsed."""
        entry = self.metric.get("matched_entry")
        raw = self.metric.get("raw_metric")
        source = self.metric.get("source", "unknown")
        confidence = self.metric.get("confidence", "low")

        if entry:
            full = entry.get("full_name", "?")
            if source == "slug_known_fallback":
                return CheckResult(
                    "Evaluation metric parsed",
                    "WARNING",
                    f"Matched via slug_known_fallback (confidence={confidence}): {full} "
                    "— evaluation page was unavailable (Kaggle blocked page)",
                )
            return CheckResult(
                "Evaluation metric parsed",
                "OK",
                f"Matched: {full} (source={source}, confidence={confidence})",
            )
        if raw:
            return CheckResult(
                "Evaluation metric parsed",
                "WARNING",
                f"Raw metric string found ('{raw}', source={source}) "
                "but not matched in knowledge base",
            )
        return CheckResult(
            "Evaluation metric parsed",
            "FAILED",
            "Could not extract metric from API or evaluation page",
        )

    def _check_data_files(self) -> list[CheckResult]:
        """Check for train.csv, test.csv existence."""
        checks: list[CheckResult] = []
        data_dir = self.out / "data"
        if not data_dir.exists() or not any(data_dir.iterdir()):
            checks.append(CheckResult("data/ files", "FAILED", "data/ directory is empty or missing"))
            return checks

        for label in ("train", "test"):
            match = _find_file_fuzzy(data_dir, label)
            if match:
                size_mb = match.stat().st_size / 1_048_576
                checks.append(CheckResult(
                    f"data/{match.name}",
                    "OK",
                    f"{size_mb:.2f} MB",
                    match,
                ))
            else:
                checks.append(CheckResult(
                    f"data/{label}.*",
                    "WARNING",
                    f"No file matching '{label}' found in data/",
                ))
        return checks

    def _check_submission_format(self) -> CheckResult:
        fmt_path = self.out / "SUBMISSION_FORMAT.md"
        sample_path = _find_file_fuzzy(self.out / "data", "sample_submission")
        if fmt_path.exists() and sample_path:
            size = fmt_path.stat().st_size
            return CheckResult(
                "SUBMISSION_FORMAT.md",
                "OK",
                f"Exists ({size} bytes), sample_submission found at {sample_path.name}",
                fmt_path,
            )
        if not sample_path:
            return CheckResult(
                "SUBMISSION_FORMAT.md",
                "WARNING",
                "sample_submission file not found in data/ — submission format may be inferred only",
                fmt_path if fmt_path.exists() else None,
            )
        return CheckResult(
            "SUBMISSION_FORMAT.md",
            "FAILED" if not fmt_path.exists() else "WARNING",
            "SUBMISSION_FORMAT.md missing" if not fmt_path.exists() else "sample_submission not found",
            fmt_path,
        )

    def _check_submissions(self) -> list[CheckResult]:
        checks: list[CheckResult] = []
        csv_path = self.out / "leaderboard" / "my_submissions.csv"

        if not csv_path.exists():
            checks.append(CheckResult(
                "leaderboard/my_submissions.csv",
                "WARNING",
                "File does not exist — you may not have joined the competition yet",
                csv_path,
            ))
            return checks

        # Read and check scores
        import csv as csv_mod
        rows: list[dict] = []
        try:
            with csv_path.open(encoding="utf-8") as fh:
                rows = list(csv_mod.DictReader(fh))
        except Exception as exc:
            checks.append(CheckResult(
                "leaderboard/my_submissions.csv",
                "FAILED",
                f"Could not read file: {exc}",
                csv_path,
            ))
            return checks

        if not rows:
            checks.append(CheckResult(
                "leaderboard/my_submissions.csv",
                "WARNING",
                "File exists but is empty (no submissions found)",
                csv_path,
            ))
            return checks

        total = len(rows)
        scored = sum(
            1 for r in rows
            if r.get("publicScore", "").strip() not in ("", "None", "nan", "NaN")
        )

        if scored == 0:
            source = self.lb.get("source", "?")
            checks.append(CheckResult(
                "leaderboard/my_submissions.csv",
                "WARNING",
                f"{total} submissions found but publicScore is empty for all of them "
                f"(source={source}). "
                "Possible causes: competition hasn't revealed scores yet; "
                "submissions still evaluating; or scores not exposed via this API.",
                csv_path,
            ))
        else:
            checks.append(CheckResult(
                "leaderboard/my_submissions.csv",
                "OK",
                f"{total} submissions, {scored} with publicScore",
                csv_path,
            ))

        return checks

    def _check_leaderboard(self) -> CheckResult:
        lb_path = self.out / "leaderboard" / "leaderboard_snapshot.csv"
        if not lb_path.exists():
            return CheckResult(
                "leaderboard/leaderboard_snapshot.csv",
                "WARNING",
                "File not found — leaderboard may not be public yet or you haven't entered",
                lb_path,
            )
        import csv as csv_mod
        try:
            with lb_path.open(encoding="utf-8") as fh:
                n = sum(1 for _ in csv_mod.DictReader(fh))
        except Exception:
            n = 0
        return CheckResult(
            "leaderboard/leaderboard_snapshot.csv",
            "OK" if n > 0 else "WARNING",
            f"{n} teams on leaderboard" if n > 0 else "File exists but appears empty",
            lb_path,
        )

    def _check_notebooks(self) -> CheckResult:
        """Check notebook collection status using failed_notebooks.csv and folder count."""
        nb_dir = self.out / "code_notebooks"
        failed_csv = nb_dir / "failed_notebooks.csv"

        if not nb_dir.exists():
            return CheckResult(
                "code_notebooks/",
                "SKIPPED",
                "Notebook collection was skipped",
            )

        # Count active notebook folders (stable nb_ prefix only; exclude _duplicates, _debug)
        nb_folders = [
            d for d in nb_dir.iterdir()
            if d.is_dir() and (d.name.startswith("nb_") or d.name.startswith("notebook_"))
            and d.name not in ("_duplicates", "_debug")
        ]
        total = len(nb_folders)
        dupe_dir = nb_dir / "_duplicates"
        n_dupes = sum(1 for d in dupe_dir.iterdir() if d.is_dir()) if dupe_dir.exists() else 0

        if total == 0:
            return CheckResult(
                "code_notebooks/",
                "FAILED",
                "No notebooks were collected",
                nb_dir,
            )

        # Count failed
        n_failed = 0
        if failed_csv.exists():
            try:
                import csv as _csv
                with failed_csv.open(encoding="utf-8") as fh:
                    n_failed = sum(1 for _ in _csv.DictReader(fh))
            except Exception:
                pass

        n_ok = total - n_failed

        dupe_note = f"; {n_dupes} duplicate(s) moved to _duplicates/" if n_dupes else ""

        # Check if CODE_NOTEBOOKS_SUMMARY.md still has unparsed placeholders
        summary_file = self.out / "CODE_NOTEBOOKS_SUMMARY.md"
        summary_unparsed = False
        if summary_file.exists():
            txt = summary_file.read_text(encoding="utf-8", errors="replace")
            summary_unparsed = "_Content not yet parsed._" in txt or "not yet parsed" in txt.lower()

        if n_failed == 0:
            status = "WARNING" if summary_unparsed else "OK"
            detail = f"{total} unique notebooks collected, all downloaded successfully{dupe_note}"
            if summary_unparsed:
                detail += "; CODE_NOTEBOOKS_SUMMARY.md still contains unparsed placeholders"
            return CheckResult("code_notebooks/", status, detail, nb_dir)
        elif n_ok > 0:
            return CheckResult(
                "code_notebooks/",
                "WARNING",
                f"{n_ok}/{total} notebooks downloaded successfully; "
                f"{n_failed} failed after retries{dupe_note} — see code_notebooks/failed_notebooks.csv",
                failed_csv,
            )
        else:
            return CheckResult(
                "code_notebooks/",
                "FAILED",
                f"All {total} notebook downloads failed{dupe_note} — see code_notebooks/failed_notebooks.csv",
                failed_csv,
            )

    def _check_extra_pages(self) -> list[CheckResult]:
        """Crash-detect all page files not already validated by _check_pages."""
        checks: list[CheckResult] = []
        already = set(_PAGE_MIN_CHARS.keys())
        pages_dir = self.out / "pages"
        if not pages_dir.exists():
            return checks

        for tab_name in _ALL_PAGE_NAMES:
            if tab_name in already:
                continue
            path = pages_dir / f"{tab_name}.md"
            label = f"pages/{tab_name}.md"
            if not path.exists():
                continue  # absence of optional pages is not an error

            text = path.read_text(encoding="utf-8", errors="replace")
            header_meta = _parse_md_header(text)
            file_quality = header_meta.get("quality", "")

            if file_quality == "failed" or _has_crash_string(text):
                checks.append(CheckResult(
                    label, "WARNING",
                    "Page contains crash/error text — browser extraction failed. "
                    f"See pages/_debug/{tab_name}.raw.txt.",
                    path,
                ))
            else:
                content = _strip_header(text)
                checks.append(CheckResult(
                    label, "OK",
                    f"{len(content.strip()):,} chars",
                    path,
                ))
        return checks

    def _check_screenshots(self) -> CheckResult:
        """Validate screenshots: count unique hashes, flag duplicates and crash pages."""
        ss_dir = self.out / "screenshots"
        if not ss_dir.exists():
            return CheckResult("screenshots/", "SKIPPED", "Screenshots directory not found")

        pngs = sorted(ss_dir.glob("*.png"))
        if not pngs:
            return CheckResult("screenshots/", "WARNING", "No screenshot files found", ss_dir)

        # Compute SHA256 for every screenshot
        hash_to_files: dict[str, list[str]] = {}
        for png in pngs:
            h = _sha256_file(png)
            hash_to_files.setdefault(h, []).append(png.name)

        total = len(pngs)
        unique_hashes = len(hash_to_files)
        dupes = {h: names for h, names in hash_to_files.items() if len(names) > 1}
        n_dupes = sum(len(v) - 1 for v in dupes.values())  # extra copies

        # Check which pages had extraction failures
        pages_dir = self.out / "pages"
        crash_tabs: list[str] = []
        for png in pngs:
            tab_name = png.stem  # e.g. "overview"
            page_md = pages_dir / f"{tab_name}.md"
            if page_md.exists():
                txt = page_md.read_text(encoding="utf-8", errors="replace")
                header_meta = _parse_md_header(txt)
                if header_meta.get("quality") == "failed" or _has_crash_string(txt):
                    crash_tabs.append(tab_name)

        n_valid = total - len(crash_tabs)

        detail_parts = [
            f"{total} saved, {n_valid} valid, {unique_hashes} unique hashes"
        ]
        if n_dupes:
            dup_desc = "; ".join(
                f"{'+'.join(names)} identical" for names in dupes.values()
            )
            detail_parts.append(f"{n_dupes} duplicate(s): {dup_desc}")
        if crash_tabs:
            detail_parts.append(
                f"{len(crash_tabs)} screenshot(s) correspond to crash pages: {', '.join(crash_tabs)}"
            )

        status = "WARNING" if (n_dupes or crash_tabs) else "OK"
        return CheckResult("screenshots/", status, "; ".join(detail_parts), ss_dir)

    def _check_discussions(self) -> CheckResult:
        """
        Validate discussion output consistency:
        - Count actual thread_*.md files on disk
        - Read the reported count from discussion_quality_report.md
        - Flag a WARNING if they do not match (stale files from previous run)
        """
        disc_dir = self.out / "discussions"
        threads_dir = disc_dir / "threads"
        quality_report = disc_dir / "discussion_quality_report.md"

        actual_count = 0
        if threads_dir.exists():
            actual_count = len(list(threads_dir.glob("thread_*.md")))

        if not quality_report.exists():
            if actual_count == 0:
                return CheckResult(
                    "discussions/",
                    "SKIPPED",
                    "No discussion files collected (browser unavailable or no threads found)",
                )
            return CheckResult(
                "discussions/",
                "WARNING",
                f"{actual_count} thread file(s) exist on disk but discussion_quality_report.md is missing",
                threads_dir,
            )

        # Parse "Valid threads saved: N" from the quality report
        report_text = quality_report.read_text(encoding="utf-8", errors="replace")
        reported_count: Optional[int] = None
        m = re.search(r"\*\*Valid threads saved:\*\*\s*(\d+)", report_text)
        if m:
            reported_count = int(m.group(1))

        if reported_count is None:
            return CheckResult(
                "discussions/",
                "WARNING",
                "Could not parse 'Valid threads saved' from discussion_quality_report.md",
                quality_report,
            )

        if actual_count != reported_count:
            return CheckResult(
                "discussions/",
                "FAILED",
                f"STALE FILES DETECTED: discussion_quality_report.md says {reported_count} valid threads "
                f"but {actual_count} thread_*.md file(s) exist on disk. "
                "Old files from a previous run were not cleaned. "
                "Re-run without --keep-latest to fix this automatically.",
                threads_dir,
            )

        if actual_count == 0:
            # Read the quality report to get the root cause note
            cause = "browser may have returned crash/Cloudflare page or competition has no discussions"
            if quality_report.exists():
                rpt = quality_report.read_text(encoding="utf-8", errors="replace")
                if "crash" in rpt.lower() or "cloudflare" in rpt.lower():
                    cause = "discussion page crashed (Kaggle/Cloudflare) — re-run with --headed --refresh-browser-state"
                elif "may have no discussions" in rpt.lower():
                    cause = "competition appears to have no discussion threads yet"
            return CheckResult(
                "discussions/",
                "WARNING",
                f"0 discussion threads collected: {cause}",
                disc_dir,
            )

        return CheckResult(
            "discussions/",
            "OK",
            f"{actual_count} thread file(s) on disk match discussion_quality_report.md (reported={reported_count})",
            threads_dir,
        )

    def _check_summary_report(self) -> CheckResult:
        path = self.out / "SUMMARY_REPORT.md"
        if path.exists() and path.stat().st_size > 200:
            return CheckResult("SUMMARY_REPORT.md", "OK", f"{path.stat().st_size:,} bytes", path)
        return CheckResult("SUMMARY_REPORT.md", "FAILED", "Missing or empty", path)

    # ──────────────────────────────────────────────────────────────────────────
    # Report writer
    # ──────────────────────────────────────────────────────────────────────────

    def _write_report(self, results: list[CheckResult]) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ok = sum(1 for r in results if r.status == "OK")
        warn = sum(1 for r in results if r.status == "WARNING")
        fail = sum(1 for r in results if r.status == "FAILED")
        skip = sum(1 for r in results if r.status == "SKIPPED")

        lines: list[str] = [
            "# Collection Quality Report",
            "",
            f"**Generated:** {now}",
            f"**Summary:** ✅ {ok} OK  |  ⚠️  {warn} WARNING  |  ❌ {fail} FAILED  |  ⬜ {skip} SKIPPED",
            "",
            "---",
            "",
            "## Check Results",
            "",
            "| Item | Status | Details |",
            "|------|--------|---------|",
        ]

        for r in results:
            details = r.details.replace("|", "\\|")  # escape markdown pipe
            lines.append(f"| `{r.item}` | {r.emoji} {r.status} | {details} |")

        lines += [
            "",
            "---",
            "",
        ]

        if fail > 0:
            lines += [
                "## Failed Items — What to Do",
                "",
            ]
            for r in results:
                if r.status == "FAILED":
                    lines += [
                        f"### `{r.item}`",
                        f"**Reason:** {r.details}",
                        "",
                        _remediation_hint(r.item),
                        "",
                    ]

        if warn > 0:
            lines += [
                "## Warnings — Review Recommended",
                "",
            ]
            for r in results:
                if r.status == "WARNING":
                    lines.append(f"- **`{r.item}`:** {r.details}")
            lines.append("")

        path = self.out / "collection_quality_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"collection_quality_report.md written ({ok} OK, {warn} WARN, {fail} FAIL)")

    def _propagate_to_archiver(self, results: list[CheckResult]) -> None:
        for r in results:
            if r.status == "FAILED":
                self.archiver.error(f"Quality check FAILED: {r.item} — {r.details}")
            elif r.status == "WARNING":
                self.archiver.warning(f"Quality check WARNING: {r.item} — {r.details}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_crash_string(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _CRASH_STRINGS)


def _parse_md_header(text: str) -> dict:
    """
    Parse the metadata header written by page_extractor into a plain dict.
    Header looks like:
      **Quality:** `failed`
      **Method:** `crash_check`
    Stops at the first '---' separator line.
    """
    meta: dict = {}
    for line in text.splitlines():
        if line.strip() == "---":
            break
        m = re.match(r"\*\*(\w+):\*\*\s*`([^`]*)`", line.strip())
        if m:
            meta[m.group(1).lower()] = m.group(2)
    return meta


def _strip_header(text: str) -> str:
    """Remove the metadata header lines added by page_extractor."""
    lines = text.splitlines()
    body_start = 0
    for i, ln in enumerate(lines):
        if ln.strip() == "---":
            body_start = i + 1
            break
    return "\n".join(lines[body_start:])


def _find_file_fuzzy(directory: Path, keyword: str) -> Optional[Path]:
    """Find first file in directory whose name contains `keyword` (case-insensitive)."""
    if not directory.exists():
        return None
    kw = keyword.lower().replace("_", "")
    for f in sorted(directory.rglob("*")):
        if f.is_file() and kw in f.name.lower().replace("_", ""):
            return f
    return None


def _remediation_hint(item: str) -> str:
    item_lower = item.lower()
    if "overview" in item_lower:
        return (
            "- Ensure you are logged in to Kaggle (run with `--headed` for manual login)\n"
            "- Check that the competition URL is correct\n"
            "- Try re-running; Kaggle may have returned a transient error page"
        )
    if "evaluation" in item_lower:
        return (
            "- The evaluation tab may not exist for all competitions\n"
            "- Check `pages/overview.md` — evaluation is sometimes described there\n"
            "- Manually check the competition page"
        )
    if "rules" in item_lower:
        return (
            "- You may need to accept the competition rules first\n"
            "- Run with `--headed` and navigate to the rules page manually"
        )
    if "data" in item_lower:
        return (
            "- Run without `--no-download` to download data files\n"
            "- Accept competition rules if required"
        )
    if "submission" in item_lower and "format" in item_lower:
        return "- Download competition data first — sample_submission.csv is needed"
    if "submission" in item_lower:
        return (
            "- Ensure you have joined the competition\n"
            "- Set KAGGLE_USERNAME and KAGGLE_KEY in ~/.kaggle/kaggle.json\n"
            "- Try running with `--headed` for browser-based score extraction"
        )
    if "metric" in item_lower:
        return (
            "- Check `pages/evaluation.md` — the metric may be described there\n"
            "- Add the metric name to the knowledge base in metric_analyzer.py if it's uncommon"
        )
    return "- Check ERRORS.md for details and re-run the relevant phase."
