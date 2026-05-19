"""
Public notebook / code collector.

Strategy:
  1. Kaggle kernels API  → list + download .ipynb files (primary)
  2. Browser             → collect per-notebook comments (no API endpoint)
  3. Parse .ipynb        → extract code cells, markdown cells, output summaries
  4. Write per-notebook folder under code_notebooks/
  5. Generate CODE_NOTEBOOKS_SUMMARY.md

Retry / resume behaviour:
  - Up to 3 download attempts with 5s / 10s / 20s waits between retries
  - If .ipynb or extracted .md already exists and is valid, skip download (resume)
  - Failed notebooks logged to code_notebooks/failed_notebooks.csv
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from .utils import polite_delay, sanitize_filename

logger = logging.getLogger("kaggle_collector")

KAGGLE_BASE = "https://www.kaggle.com"

_SORT_MODES: list[tuple[str, str]] = [
    ("voteCount", "most_voted"),
    ("scoreDescending", "best_score"),
    ("hotness", "hot"),
    ("dateCreated", "most_recent"),
]

_COMMENTS_JS = """
() => {
    const NAV_PHRASES = [
        'competitions', 'datasets', 'models', 'code', 'discussions',
        'sign in', 'register', 'kaggle', 'notifications', 'settings',
        'predicting f1', 'pit stop', 'home', 'profile', 'search',
        'f1 strategy', 'dataset'
    ];
    function isNavNoise(t) {
        const lower = t.toLowerCase().trim();
        return NAV_PHRASES.some(p => lower === p || lower.startsWith(p + '\\n'))
            || t.length < 20;
    }
    const texts = [];
    const selectors = [
        '[data-testid*="comment-content"]',
        '[data-testid*="CommentContent"]',
        '.comment-content',
        '[class*="CommentBody"]',
        '[class*="comment-body"]',
    ];
    let found = false;
    for (const sel of selectors) {
        const els = document.querySelectorAll(sel);
        if (els.length >= 1) {
            els.forEach(el => {
                const t = el.innerText.trim();
                if (!isNavNoise(t)) texts.push(t);
            });
            found = true;
            break;
        }
    }
    if (!found) return [];
    return texts.slice(0, 50);
}
"""

# Retry settings
_MAX_ATTEMPTS = 3
_RETRY_WAITS = [5, 10, 20]   # seconds between attempts

# Minimum file size to consider a cached .ipynb valid
_MIN_IPYNB_BYTES = 100


class NotebookCollector:
    """Downloads and parses public competition notebooks."""

    def __init__(
        self,
        slug: str,
        output_dir: Path,
        max_notebooks: int = 15,
        delay: float = 2.5,
    ) -> None:
        self.slug = slug
        self.output_dir = output_dir
        self.nb_dir = output_dir / "code_notebooks"
        self.nb_dir.mkdir(parents=True, exist_ok=True)
        self.max_notebooks = max_notebooks
        self.delay = delay

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def collect(self, page=None) -> list[dict[str, Any]]:
        """Collect notebooks. Returns list of notebook metadata dicts."""
        api = self._try_auth()
        if api is None:
            logger.warning("Kaggle API unavailable; skipping notebook collection")
            self._regenerate_summary_from_cache()
            return []

        nb_list = self._list_notebooks(api)
        if not nb_list:
            logger.warning("No public notebooks found for this competition")
            self._regenerate_summary_from_cache()
            return []

        total = min(len(nb_list), self.max_notebooks)
        logger.info(f"Collecting {total} notebooks …")

        # ── Deduplicate existing folders before starting ──────────────────────
        n_dupes_cleaned = self._cleanup_duplicate_folders()
        if n_dupes_cleaned:
            logger.info(f"  Cleaned up {n_dupes_cleaned} duplicate notebook folder(s)")

        collected: list[dict[str, Any]] = []
        failed_entries: list[dict[str, Any]] = []
        n_downloaded = 0
        n_cached = 0
        n_failed = 0

        for i, meta in enumerate(nb_list[:total], 1):
            ref = meta.get("ref", "")
            title = meta.get("title", f"notebook_{i:03d}")

            # Stable folder name based on owner+slug, not run-time rank
            nb_folder = self._stable_folder(ref, title)
            nb_folder.mkdir(parents=True, exist_ok=True)

            logger.info(f"  [{i}/{total}] {title}")

            # Always write/refresh metadata
            (nb_folder / "metadata.json").write_text(
                json.dumps(meta, indent=2, default=str), encoding="utf-8"
            )

            # ── Resume check ─────────────────────────────────────────────────
            cached_ipynb = self._find_cached_ipynb(nb_folder)
            extracted_ok = self._extraction_valid(nb_folder)

            if extracted_ok:
                logger.info("    ✓ cached — skipping download")
                n_cached += 1
                collected.append(meta)
                if page is not None:
                    await self._collect_comments(page, meta, nb_folder)
                polite_delay(self.delay)
                continue

            if cached_ipynb and not extracted_ok:
                logger.info("    ✓ .ipynb cached — re-extracting content")
                self._extract(cached_ipynb, nb_folder, title)
                n_cached += 1
                collected.append(meta)
                if page is not None:
                    await self._collect_comments(page, meta, nb_folder)
                polite_delay(self.delay)
                continue

            # ── Download with retry ──────────────────────────────────────────
            ipynb, last_error, attempts = self._download_with_retry(api, ref, nb_folder)

            if ipynb and ipynb.exists():
                self._extract(ipynb, nb_folder, title)
                n_downloaded += 1
            else:
                n_failed += 1
                owner = ref.split("/")[0] if "/" in ref else ""
                slug  = ref.split("/")[1] if "/" in ref else ref
                logger.warning(f"    ✗ failed after {attempts} attempt(s): {_short_err(last_error)}")
                _write(nb_folder / "notebook.md",
                       f"# {title}\n\n_Download failed after {attempts} attempts._\n\n**Last error:** `{last_error}`")
                _write(nb_folder / "notebook_code.py",
                       f"# {title}\n# Download failed after {attempts} attempts\n")
                _write(nb_folder / "outputs_summary.md", "_No outputs (download failed)._")
                failed_entries.append({
                    "rank": i,
                    "title": title,
                    "owner": owner,
                    "slug": slug,
                    "url": meta.get("url", ""),
                    "reason": "kernels_pull failed",
                    "last_error": str(last_error)[:300],
                    "attempts": attempts,
                })

            collected.append(meta)
            if page is not None:
                await self._collect_comments(page, meta, nb_folder)
            polite_delay(self.delay)

        if failed_entries:
            self._write_failed_csv(failed_entries)

        self._write_summary(collected, n_downloaded, n_cached, n_failed,
                            n_dupes_cleaned, failed_entries)
        self._write_comments_quality(collected)

        # If this run collected nothing but cached folders exist, still regenerate summary
        if not collected:
            self._regenerate_summary_from_cache()

        failed_csv_path = self.nb_dir / "failed_notebooks.csv"
        logger.info(
            f"\n  Notebook collection stats:\n"
            f"    Total selected  : {total}\n"
            f"    Downloaded      : {n_downloaded}\n"
            f"    From cache      : {n_cached}\n"
            f"    Failed          : {n_failed}"
            + (f"\n    Duplicates removed: {n_dupes_cleaned}" if n_dupes_cleaned else "")
            + (f"\n    Failed list     : {failed_csv_path}" if failed_entries else "")
        )

        return collected

    # ──────────────────────────────────────────────────────────────────────────
    # Stable folder naming
    # ──────────────────────────────────────────────────────────────────────────

    def _stable_folder(self, ref: str, title: str) -> Path:
        """
        Return a stable folder path for a notebook, based on owner+slug from
        the ref field (e.g. 'lucabasa/f1-strategy-eda-and-base-models').
        Falls back to a sanitized title if ref is empty.
        """
        if ref and "/" in ref:
            owner, slug = ref.split("/", 1)
            safe = sanitize_filename(f"{owner}__{slug}")[:72]
        else:
            safe = sanitize_filename(ref or title)[:72]
        return self.nb_dir / f"nb_{safe}"

    # ──────────────────────────────────────────────────────────────────────────
    # Duplicate folder cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def _cleanup_duplicate_folders(self) -> int:
        """
        Detect folders with the OLD rank-prefixed naming (notebook_NNN_title),
        check if a stable nb_owner__slug folder already exists for the same ref,
        and move old duplicates to _duplicates/.

        Returns the number of folders moved.
        """
        old_pattern = re.compile(r"^notebook_\d{3}_")
        old_folders = [
            d for d in self.nb_dir.iterdir()
            if d.is_dir() and old_pattern.match(d.name)
        ]
        if not old_folders:
            return 0

        # Build map of ref → stable folder
        ref_to_stable: dict[str, Path] = {}
        for d in self.nb_dir.iterdir():
            if not d.is_dir() or old_pattern.match(d.name):
                continue
            meta_file = d / "metadata.json"
            if meta_file.exists():
                try:
                    ref = json.loads(meta_file.read_text(encoding="utf-8")).get("ref", "")
                    if ref:
                        ref_to_stable[ref] = d
                except Exception:
                    pass

        dupes_dir = self.nb_dir / "_duplicates"
        moved = 0
        for old_folder in old_folders:
            meta_file = old_folder / "metadata.json"
            ref = ""
            if meta_file.exists():
                try:
                    ref = json.loads(meta_file.read_text(encoding="utf-8")).get("ref", "")
                except Exception:
                    pass

            # Build what the stable folder would be for this ref
            title = old_folder.name
            stable = self._stable_folder(ref, title) if ref else None

            if stable and stable.exists():
                # Stable folder already exists — this old folder is a duplicate
                dupes_dir.mkdir(exist_ok=True)
                dest = dupes_dir / old_folder.name
                if dest.exists():
                    import shutil as _shutil
                    _shutil.rmtree(dest)
                old_folder.rename(dest)
                moved += 1
            elif stable and not stable.exists() and ref:
                # Stable folder doesn't exist yet — rename old folder to stable name
                old_folder.rename(stable)
                # Don't count as moved; it's a rename to the correct name
            else:
                # Can't determine ref — move to _duplicates for safety
                dupes_dir.mkdir(exist_ok=True)
                dest = dupes_dir / old_folder.name
                if dest.exists():
                    import shutil as _shutil
                    _shutil.rmtree(dest)
                old_folder.rename(dest)
                moved += 1

        return moved

    # ──────────────────────────────────────────────────────────────────────────
    # Resume helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _find_cached_ipynb(self, folder: Path) -> Optional[Path]:
        """Return a valid cached .ipynb in the folder, or None."""
        for f in folder.glob("*.ipynb"):
            if f.stat().st_size >= _MIN_IPYNB_BYTES:
                return f
        return None

    def _extraction_valid(self, folder: Path) -> bool:
        """Return True if notebook.md and notebook_code.py exist and are non-trivial."""
        md = folder / "notebook.md"
        code = folder / "notebook_code.py"
        if not (md.exists() and code.exists()):
            return False
        # Must have meaningful content, not just the failure stub
        md_text = md.read_text(encoding="utf-8", errors="replace")
        return (
            len(md_text) > 50
            and "_Download failed" not in md_text
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Download with retry + exponential backoff
    # ──────────────────────────────────────────────────────────────────────────

    def _download_with_retry(
        self, api, ref: str, folder: Path
    ) -> tuple[Optional[Path], str, int]:
        """
        Try to download the notebook up to _MAX_ATTEMPTS times.
        Returns (ipynb_path_or_None, last_error_str, attempts_made).
        """
        if not ref:
            return None, "empty ref", 0

        last_error = ""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                api.kernels_pull(ref, path=str(folder), metadata=True, quiet=True)
                for f in folder.glob("*.ipynb"):
                    if f.stat().st_size >= _MIN_IPYNB_BYTES:
                        return f, "", attempt
                # kernels_pull succeeded but no .ipynb found
                last_error = "kernels_pull completed but no .ipynb file produced"
            except Exception as exc:
                last_error = str(exc)
                if attempt < _MAX_ATTEMPTS:
                    wait = _RETRY_WAITS[attempt - 1]
                    logger.warning(
                        f"    Attempt {attempt}/{_MAX_ATTEMPTS} failed for {ref} — "
                        f"retrying in {wait}s … ({_short_err(last_error)})"
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"    kernels_pull failed for {ref}: {last_error}"
                    )

        return None, last_error, _MAX_ATTEMPTS

    # ──────────────────────────────────────────────────────────────────────────
    # API helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _try_auth(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
            api = KaggleApi()
            api.authenticate()
            return api
        except Exception as exc:
            logger.warning(f"Kaggle API auth failed for notebooks: {exc}")
            return None

    def _list_notebooks(self, api) -> list[dict[str, Any]]:
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        for sort_by, label in _SORT_MODES:
            try:
                kernels = api.kernels_list(
                    competition=self.slug,
                    sort_by=sort_by,
                    page_size=min(self.max_notebooks, 20),
                    language="python",
                ) or []
                for k in kernels:
                    ref = getattr(k, "ref", "") or str(k)
                    if ref in seen:
                        continue
                    seen.add(ref)
                    results.append({**self._ser(k), "_found_via": label})
                polite_delay(1.0)
            except Exception as exc:
                logger.warning(f"  Notebook list failed (sort={sort_by}): {exc}")

        return results

    def _ser(self, k) -> dict[str, Any]:
        attrs = [
            "ref", "title", "author", "slug", "lastRunTime",
            "totalVotes", "totalComments", "language", "kernelType",
            "isPrivate", "enableGpu", "enableInternet", "categoryIds",
        ]
        d: dict[str, Any] = {}
        for a in attrs:
            d[a] = getattr(k, a, None)
        ref = d.get("ref", "")
        if ref:
            d["url"] = f"{KAGGLE_BASE}/{ref}"
        return d

    # ──────────────────────────────────────────────────────────────────────────
    # .ipynb parsing
    # ──────────────────────────────────────────────────────────────────────────

    def _extract(self, ipynb: Path, folder: Path, title: str) -> None:
        """Parse the .ipynb and write notebook.md, notebook_code.py, outputs_summary.md."""
        try:
            data = json.loads(ipynb.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            logger.warning(f"    Could not parse {ipynb.name}: {exc}")
            return

        cells = data.get("cells", [])
        md_parts: list[str] = [f"# {title}\n"]
        code_parts: list[str] = [f"# {title} — extracted code cells\n"]
        out_parts: list[str] = [f"# {title} — output summary\n"]

        for idx, cell in enumerate(cells):
            ctype = cell.get("cell_type", "")
            src = "".join(cell.get("source", []))

            if ctype == "markdown":
                md_parts.append(src)
                md_parts.append("")

            elif ctype == "code":
                code_parts.append(f"\n# ── Cell {idx + 1} {'─' * 40}")
                code_parts.append(src)

                for out in cell.get("outputs", []):
                    otype = out.get("output_type", "")
                    if otype in ("stream", "display_data", "execute_result"):
                        raw = out.get("text") or out.get("data", {}).get("text/plain", [])
                        text = "".join(raw) if isinstance(raw, list) else str(raw)
                        if text.strip():
                            out_parts.append(f"\n--- Cell {idx + 1} ---")
                            out_parts.append(text[:3_000])

        _write(folder / "notebook.md", "\n".join(md_parts))
        _write(folder / "notebook_code.py", "\n".join(code_parts))
        _write(
            folder / "outputs_summary.md",
            "\n".join(out_parts) if len(out_parts) > 1 else "_No captured outputs._",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Browser: comment collection
    # ──────────────────────────────────────────────────────────────────────────

    async def _collect_comments(
        self, page, meta: dict[str, Any], folder: Path
    ) -> None:
        url = meta.get("url", "")
        if not url:
            return
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=40_000)
            await page.wait_for_timeout(3_000)
            comments: list[str] = await page.evaluate(_COMMENTS_JS)
            # Reject obvious false positives: very short or navigation-like entries
            real = [c for c in comments if len(c) >= 20 and not _is_nav_noise(c)]
            if real:
                lines = [f"# Comments\n", f"Source: {url}\n", "---\n"]
                for i, c in enumerate(real, 1):
                    lines.append(f"**Comment {i}:**\n\n{c}\n\n---")
                _write(folder / "comments.md", "\n".join(lines))
                logger.debug(f"    {len(real)} real comments saved")
            else:
                _write(folder / "comments.md",
                       f"# Comments\n\nSource: {url}\n\n"
                       "_No real notebook comments collected. "
                       "(Comment elements not found or only navigation text was detected.)_")
        except Exception as exc:
            logger.warning(f"    Comment collection failed for {url}: {exc}")
            _write(folder / "comments.md",
                   f"# Comments\n\nSource: {url}\n\n"
                   f"_Navigation timeout or error: {exc}_")

    # ──────────────────────────────────────────────────────────────────────────
    # Outputs
    # ──────────────────────────────────────────────────────────────────────────

    def _write_comments_quality(self, notebooks: list[dict[str, Any]]) -> None:
        """Write comments_quality_report.md summarising real vs. noise comments."""
        lines = [
            "# Notebook Comments Quality Report\n",
            "| # | Title | Comments File | Status |",
            "|---|-------|---------------|--------|",
        ]
        for i, nb in enumerate(notebooks, 1):
            title = nb.get("title", "?")[:60]
            ref = nb.get("ref", "")
            folder = self._stable_folder(ref, nb.get("title", ""))
            comments_file = folder / "comments.md"
            if not comments_file.exists():
                status = "❔ Not collected"
            else:
                txt = comments_file.read_text(encoding="utf-8", errors="replace")
                if "No real notebook comments" in txt:
                    status = "⚠️ No real comments"
                elif "Navigation timeout" in txt or "error:" in txt.lower():
                    status = "❌ Extraction error"
                elif "Comment 1:" in txt:
                    # Count real comments
                    n = txt.count("**Comment ")
                    status = f"✅ {n} comment(s)"
                else:
                    status = "⚠️ Unknown"
            lines.append(f"| {i} | {title} | `comments.md` | {status} |")

        lines += [
            "",
            "---",
            "",
            "**Note:** Kaggle's comment DOM is not reliably targetable in headless mode. "
            "Many comment sections show only navigation/chrome text (e.g. tab names, page titles). "
            "The collector now rejects navigation noise and writes 'No real notebook comments collected.' instead.",
        ]
        _write(self.nb_dir / "comments_quality_report.md", "\n".join(lines))
        logger.info("comments_quality_report.md written")

    def _write_failed_csv(self, entries: list[dict[str, Any]]) -> None:
        path = self.nb_dir / "failed_notebooks.csv"
        fields = ["rank", "title", "owner", "slug", "url", "reason", "last_error", "attempts"]
        try:
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(entries)
            logger.info(f"  failed_notebooks.csv → {path}")
        except Exception as exc:
            logger.warning(f"  Could not write failed_notebooks.csv: {exc}")

    def _write_summary(
        self,
        notebooks: list[dict[str, Any]],
        n_downloaded: int,
        n_cached: int,
        n_failed: int,
        n_dupes_cleaned: int,
        failed_entries: list[dict[str, Any]],
    ) -> None:
        lines: list[str] = [
            "# Code Notebooks Summary\n",
            f"Total notebooks collected: **{len(notebooks)}**  ",
            f"Downloaded: **{n_downloaded}**  |  From cache: **{n_cached}**  "
            f"|  Failed: **{n_failed}**  |  Duplicates cleaned: **{n_dupes_cleaned}**\n",
            "---\n",
        ]

        if failed_entries:
            lines += [
                "## ⚠️ Failed Downloads\n",
                "| # | Title | Slug | Attempts | Reason |",
                "|---|-------|------|----------|--------|",
            ]
            for fe in failed_entries:
                lines.append(
                    f"| {fe['rank']} | {fe['title']} | {fe['slug']} "
                    f"| {fe['attempts']} | {fe['reason']} |"
                )
            lines += ["", "---", ""]

        for i, nb in enumerate(notebooks, 1):
            title = nb.get("title", "?")
            author = nb.get("author", "?")
            url = nb.get("url", "")
            ref = nb.get("ref", "")
            votes = nb.get("totalVotes", "?")
            comments = nb.get("totalComments", "?")
            sort = nb.get("_found_via", "")
            gpu = "Yes" if nb.get("enableGpu") else "No"
            internet = "Yes" if nb.get("enableInternet") else "No"

            # Use the stable folder path derived from ref
            nb_folder = self._stable_folder(ref, title)

            lines += [
                f"## {i}. {title}",
                "",
                f"- **Author:** {author}",
                f"- **URL:** {url}",
                f"- **Votes:** {votes}  |  **Comments:** {comments}",
                f"- **Discovered via:** {sort}",
                "",
                _deep_analysis(nb_folder, slug=self.slug),
                "",
                "---",
                "",
            ]

        lines += [
            "## Quick Takeaways for Your Own Solution",
            "",
            "*(Review each notebook above for models, features, and validation strategies.)*",
            "",
            "### Common patterns to look for:",
            "- Feature engineering ideas in markdown cells",
            "- Cross-validation setup in code cells",
            "- Ensemble / stacking / blending approaches",
            "- Public LB vs. CV correlation",
            "- Warnings about data leakage or overfitting",
        ]

        _write(self.output_dir / "CODE_NOTEBOOKS_SUMMARY.md", "\n".join(lines))
        logger.info("CODE_NOTEBOOKS_SUMMARY.md written")

    def _regenerate_summary_from_cache(self) -> None:
        """
        Scan existing nb_* folders, read their metadata.json and code files,
        and regenerate CODE_NOTEBOOKS_SUMMARY.md without requiring a live API call.
        Called when this run collected 0 new notebooks but cached folders exist.
        """
        nb_folders = sorted(
            f for f in self.nb_dir.iterdir()
            if f.is_dir() and f.name.startswith("nb_")
        )
        if not nb_folders:
            return

        existing_summary = self.output_dir / "CODE_NOTEBOOKS_SUMMARY.md"
        if existing_summary.exists():
            current = existing_summary.read_text(encoding="utf-8")
            # Only skip regeneration if the file already has real content
            if "_Content not yet parsed._" not in current and "not yet parsed" not in current.lower():
                return  # already has real parsed content

        logger.info(
            f"  Regenerating CODE_NOTEBOOKS_SUMMARY.md from {len(nb_folders)} cached folder(s) …"
        )

        lines: list[str] = [
            "# Code Notebooks Summary\n",
            f"Total notebooks collected: **{len(nb_folders)}**  ",
            f"_(Regenerated from cached notebook files — no live API call in this run.)_\n",
            "---\n",
        ]

        for i, folder in enumerate(nb_folders, 1):
            # Load metadata if available
            meta_file = folder / "metadata.json"
            meta: dict[str, Any] = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            title = meta.get("title") or folder.name
            author = meta.get("author") or meta.get("userName") or "?"
            url = meta.get("url") or meta.get("scriptUrl") or ""
            votes = meta.get("totalVotes") or meta.get("voteCount") or "?"
            gpu = "Yes" if meta.get("enableGpu") else "No"
            internet = "Yes" if meta.get("enableInternet") else "No"

            lines += [
                f"## {i}. {title}",
                "",
                f"- **Author:** {author}",
                f"- **URL:** {url}",
                f"- **Votes:** {votes}",
                "",
                _deep_analysis(folder, slug=self.slug),
                "",
                "---",
                "",
            ]

        lines += [
            "## Quick Takeaways for Your Own Solution",
            "",
            "*(Review each notebook above for models, features, and validation strategies.)*",
        ]

        _write(self.output_dir / "CODE_NOTEBOOKS_SUMMARY.md", "\n".join(lines))
        logger.info(f"  CODE_NOTEBOOKS_SUMMARY.md regenerated from {len(nb_folders)} cached folders")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _short_err(msg: str, max_len: int = 120) -> str:
    msg = msg.replace("\n", " ").strip()
    return msg[:max_len] + "…" if len(msg) > max_len else msg


def _deep_analysis(folder: Path, slug: str = "") -> str:
    """
    Parse notebook_code.py and notebook.md to produce a rich summary.
    Falls back to '_Download failed._' if content is absent.

    `slug` is used to distinguish competition-provided data from external datasets.
    """
    import json as _json

    code_file = folder / "notebook_code.py"
    md_file = folder / "notebook.md"
    meta_file = folder / "metadata.json"

    if not code_file.exists() and not md_file.exists():
        return "_Notebook files not found (download may have failed)._"

    code_text = code_file.read_text(encoding="utf-8", errors="replace") if code_file.exists() else ""
    md_text = md_file.read_text(encoding="utf-8", errors="replace") if md_file.exists() else ""

    if "_Download failed" in md_text:
        return "_Download failed — content not available._"

    # Load metadata.json for authoritative flags (internet / GPU / dataset sources)
    meta: dict = {}
    if meta_file.exists():
        try:
            meta = _json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    combined = code_text + "\n" + md_text
    snippets: list[str] = []

    # ── Models / libraries ────────────────────────────────────────────────────
    models = _detect_patterns(code_file, {
        "LightGBM":              r"lgb\.|LGBMClassifier|LGBMRegressor|import lightgbm",
        "XGBoost":               r"xgb\.|XGBClassifier|XGBRegressor|import xgboost",
        "CatBoost":              r"CatBoost|CatBoostClassifier|import catboost",
        "HistGradientBoosting":  r"HistGradientBoostingClassifier|HistGradientBoostingRegressor",
        "Random Forest":         r"RandomForestClassifier|RandomForestRegressor",
        "LogisticRegression":    r"LogisticRegression",
        "Neural Net / PyTorch":  r"import torch|nn\.Module|nn\.Linear",
        "Neural Net / Keras":    r"import keras|from keras|from tensorflow",
        "RealMLP":               r"RealMLP|realmlp|TabularPredictor.*realmlp",
        "TabPFN":                r"TabPFN|tabpfn",
        "sklearn Pipeline":      r"Pipeline\(|make_pipeline",
        "SVM":                   r"\bSVC\b|\bSVR\b",
    })
    if models:
        snippets.append(f"**Models / libraries:** {', '.join(models)}")

    # ── CV strategy ───────────────────────────────────────────────────────────
    cv = _detect_patterns(code_file, {
        "StratifiedKFold":      r"StratifiedKFold",
        "StratifiedGroupKFold": r"StratifiedGroupKFold",
        "GroupKFold":           r"GroupKFold",
        "KFold":                r"\bKFold\b",
        "TimeSeriesSplit":      r"TimeSeriesSplit",
    })
    if cv:
        snippets.append(f"**CV strategy:** {', '.join(cv)}")

    # ── Feature engineering ───────────────────────────────────────────────────
    fe = _detect_patterns(code_file, {
        "Target encoding":       r"target.*encod|TargetEncoder",
        "Frequency encoding":    r"freq.*encod|value_counts\(\)",
        "Label encoding":        r"LabelEncoder|\.factorize\(",
        "GroupBy aggregations":  r"\.groupby\(.*\)\.agg|\.groupby\(.*\)\.transform",
        "Date/time features":    r"\.dt\.\w+",
        "Interaction features":  r"interact|cross.*feat",
        "Embeddings":            r"Embedding\(",
    })
    if fe:
        snippets.append(f"**Feature engineering:** {', '.join(fe)}")

    # ── Ensemble / blending ───────────────────────────────────────────────────
    blend = _detect_patterns(code_file, {
        "Rank blending":   r"rank.*blend|rankdata|scipy\.stats\.rankdata",
        "Weighted blend":  r"weight.*blend|blend.*weight|\* ?0\.\d+ ?\+",
        "Stacking":        r"StackingClassifier|StackingRegressor|meta.*model|level.?2",
        "Voting":          r"VotingClassifier|VotingRegressor",
        "OOF predictions": r"oof|out.of.fold",
    })
    if blend:
        snippets.append(f"**Ensemble / blending:** {', '.join(blend)}")

    # ── GPU ───────────────────────────────────────────────────────────────────
    # Prefer metadata flag; fall back to code heuristics
    gpu_meta = meta.get("enableGpu") or meta.get("enable_gpu")
    if gpu_meta is not None:
        gpu_str = "Yes (metadata flag)" if gpu_meta else "No (metadata flag)"
    else:
        gpu_code = bool(_detect_patterns(code_file, {
            "GPU": r"\.cuda\(\)|device.*cuda|device\s*=\s*['\"]cuda['\"]|\.to\(['\"]cuda",
        }))
        gpu_str = "Likely (code uses .cuda())" if gpu_code else "Not detected"
    snippets.append(f"**GPU:** {gpu_str}")

    # ── Internet access (three levels) ────────────────────────────────────────
    #   a) Notebook metadata flag (authoritative)
    #   b) Code mentions HTTP / download functions (indirect evidence)
    #   c) Code actually downloads data at runtime (high confidence)
    internet_meta = meta.get("enableInternet") or meta.get("enable_internet")
    internet_meta_str = (
        "Yes (notebook metadata flag set)" if internet_meta
        else ("No (metadata flag not set)" if internet_meta is not None else "Unknown (metadata absent)")
    )

    internet_code_hints = _detect_patterns(code_file, {
        "HTTP library imported": r"^import requests|^from requests|^import urllib|^from urllib",
        "wget/curl shell cmd":   r"!wget |!curl |subprocess.*wget|subprocess.*curl",
        "Hub/model download":    r"torch\.hub\.load|timm\.|from_pretrained\(",
    })
    internet_code_downloads = _detect_patterns(code_file, {
        "Actual HTTP fetch in code": r"requests\.get\s*\(['\"]https?://|urllib\.request\.urlopen\s*\(['\"]https?://|!wget\s+https?://|!curl\s+https?://",
    })

    internet_lines = [f"  - Metadata flag: {internet_meta_str}"]
    if internet_code_downloads:
        internet_lines.append(f"  - Code downloads data via HTTP: {', '.join(internet_code_downloads)}")
    if internet_code_hints:
        internet_lines.append(f"  - HTTP-related imports/calls (not necessarily runtime downloads): {', '.join(internet_code_hints)}")
    if not internet_code_hints and not internet_code_downloads:
        internet_lines.append("  - No HTTP-related code detected")
    snippets.append("**Internet access:**\n" + "\n".join(internet_lines))

    # ── External dataset (three levels) ──────────────────────────────────────
    #   a) Kaggle competition input data (/kaggle/input/<slug>/)
    #   b) Kaggle other datasets (/kaggle/input/<other>/)
    #   c) Truly external data (non-Kaggle HTTP or arbitrary paths)
    comp_slug_norm = (slug or "").replace("-", "[-_]").replace("_", "[-_]")

    competition_input = bool(re.search(
        rf"/kaggle/input/{comp_slug_norm}/", code_text, re.IGNORECASE
    )) if comp_slug_norm else False

    # Any /kaggle/input/<X>/ path that is NOT the competition slug
    all_kaggle_inputs = re.findall(r"/kaggle/input/([\w\-]+)/", code_text, re.IGNORECASE)
    other_datasets = sorted(set(
        x for x in all_kaggle_inputs
        if not (comp_slug_norm and re.match(f"^{comp_slug_norm}$", x, re.IGNORECASE))
    ))

    external_http_data = bool(_detect_patterns(code_file, {
        "External HTTP data": r"pd\.read_csv\s*\(['\"]https?://|pd\.read_parquet\s*\(['\"]https?://|requests\.get.*\.csv",
    }))
    external_local_paths = bool(re.search(
        r"pd\.read_csv\s*\(['\"](?!/kaggle/input)(?!['\"])[a-zA-Z/]", code_text
    ))

    ext_lines = []
    ext_lines.append(
        f"  - Competition input (`/kaggle/input/{slug or '<slug>'}/`): {'Yes' if competition_input else 'Not detected'}"
    )
    if other_datasets:
        ext_lines.append(f"  - Other Kaggle datasets used: {', '.join(other_datasets[:5])}")
    else:
        ext_lines.append("  - Other Kaggle datasets: None detected")
    if external_http_data:
        ext_lines.append("  - External HTTP data loaded directly (non-Kaggle URL): Yes")
    if external_local_paths:
        ext_lines.append("  - Local file paths outside /kaggle/input/ referenced: Yes")
    if not external_http_data and not external_local_paths and not other_datasets:
        ext_lines.append("  - No arbitrary external data detected")
    snippets.append("**Dataset sources:**\n" + "\n".join(ext_lines))

    # ── Submission creation ───────────────────────────────────────────────────
    creates_submission = bool(_detect_patterns(code_file, {
        "Creates submission": r"to_csv.*submission|submission.*to_csv",
    }))
    if creates_submission:
        snippets.append("**Creates submission file:** Yes")

    # ── OOF / LB scores from text ─────────────────────────────────────────────
    scores: list[str] = []
    for pattern in [
        r"(?:oof|cv|local)[\s_]*(?:score|auc)[:\s=]+([0-9]+\.[0-9]{3,5})",
        r"(?:lb|leaderboard|public)[:\s=]+([0-9]+\.[0-9]{3,5})",
        r"(?:auc|score)[:\s=]+([0-9]{1}\.[0-9]{4,5})",
    ]:
        for m in re.findall(pattern, combined, re.IGNORECASE)[:3]:
            try:
                if 0.5 < float(m) < 1.0:
                    scores.append(m)
            except ValueError:
                pass
    if scores:
        snippets.append(f"**Score mentions:** {', '.join(sorted(set(scores), reverse=True)[:5])}")

    return "\n\n".join(snippets) if snippets else "_No identifiable patterns found in notebook code._"


_NAV_NOISE_WORDS = {
    "competitions", "datasets", "models", "code", "discussions",
    "sign in", "register", "kaggle", "home", "profile", "search",
    "notifications", "settings",
}


def _is_nav_noise(text: str) -> bool:
    """Return True if the text looks like navigation/chrome noise rather than a real comment."""
    lower = text.lower().strip()
    if len(lower) < 20:
        return True
    if any(lower == w or lower.startswith(w + "\n") for w in _NAV_NOISE_WORDS):
        return True
    return False


def _detect_patterns(file: Path, patterns: dict[str, str]) -> list[str]:
    if not file.exists():
        return []
    text = file.read_text(encoding="utf-8", errors="replace")
    return [name for name, pat in patterns.items() if re.search(pat, text, re.IGNORECASE)]


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
