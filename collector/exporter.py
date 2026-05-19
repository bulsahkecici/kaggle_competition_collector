"""
Report generator.

Combines API metadata and browser-collected content into a single
SUMMARY_REPORT.md that is ready to hand off to an AI assistant for analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("kaggle_collector")

# ── Display labels for API metadata fields ──────────────────────────────────
_META_LABELS: list[tuple[str, str]] = [
    ("title", "Title"),
    ("ref", "Reference slug"),
    ("url", "URL"),
    ("evaluationMetric", "Evaluation metric"),
    ("reward", "Reward / prize"),
    ("teamCount", "Teams entered"),
    ("maxTeamSize", "Max team size"),
    ("maxDailySubmissions", "Max daily submissions"),
    ("enabledDate", "Competition start"),
    ("deadline", "Submission deadline"),
    ("mergerDeadline", "Team merger deadline"),
    ("newEntrantDeadline", "New entrant deadline"),
    ("category", "Category"),
    ("organizationName", "Organizer"),
    ("isKernelsSubmissionsOnly", "Notebook-only submissions"),
    ("userHasEntered", "You have entered"),
    ("userRank", "Your current rank"),
]

# ── Tab display names ────────────────────────────────────────────────────────
_TAB_LABELS: dict[str, str] = {
    "overview": "Overview",
    "data": "Data description",
    "evaluation": "Evaluation",
    "rules": "Rules",
    "leaderboard": "Leaderboard",
    "discussion": "Discussion",
}


class Exporter:
    """
    Assembles all collected information into a structured output directory
    and generates a human- (and AI-) readable SUMMARY_REPORT.md.
    """

    def __init__(
        self,
        slug: str,
        output_dir: Path,
        api_data: dict[str, Any],
        browser_data: dict[str, Any],
    ) -> None:
        self.slug = slug
        self.output_dir = output_dir
        self.api_data = api_data
        self.browser_data = browser_data

    def generate_report(self) -> Path:
        """Write SUMMARY_REPORT.md and return its path."""
        report_path = self.output_dir / "SUMMARY_REPORT.md"
        lines: list[str] = []

        meta = self.api_data.get("metadata", {})
        title = meta.get("title") or self.slug
        collected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._section_header(lines, title, collected_at)
        self._section_metadata(lines, meta)
        self._section_files(lines)
        self._section_pages(lines)
        self._section_screenshots(lines)
        self._section_discussion(lines)
        self._section_tree(lines)
        self._section_footer(lines)

        report_content = "\n".join(lines)
        report_path.write_text(report_content, encoding="utf-8")
        logger.info(f"SUMMARY_REPORT.md written → {report_path}")
        return report_path

    # ──────────────────────────────────────────────────────────────────────────
    # Report sections
    # ──────────────────────────────────────────────────────────────────────────

    def _section_header(self, lines: list[str], title: str, collected_at: str) -> None:
        lines += [
            f"# Kaggle Competition Report: {title}",
            "",
            f"> **Competition slug:** `{self.slug}`  ",
            f"> **Collected at:** {collected_at}  ",
            f"> **Output directory:** `{self.output_dir}`  ",
            "",
            "---",
            "",
        ]

    def _section_metadata(self, lines: list[str], meta: dict[str, Any]) -> None:
        lines += ["## Competition Metadata", ""]
        if not meta:
            lines += [
                "_Metadata could not be retrieved via the Kaggle API. "
                "Check that `~/.kaggle/kaggle.json` is configured correctly._",
                "",
            ]
            return

        for field, label in _META_LABELS:
            val = meta.get(field)
            if val is not None:
                lines.append(f"- **{label}:** {val}")
        lines.append("")

        desc = str(meta.get("description") or "").strip()
        if desc:
            lines += ["### Description", "", desc, ""]

        tags = meta.get("tags")
        if tags:
            tag_list = ", ".join(f"`{t}`" for t in (tags if isinstance(tags, list) else [tags]))
            lines += [f"**Tags:** {tag_list}", ""]

    def _section_files(self, lines: list[str]) -> None:
        lines += ["## Data Files", ""]
        files: list[dict] = self.api_data.get("files_manifest", [])

        if files:
            lines += ["| File | Size | Created |", "|------|------|---------|"]
            for f in files:
                name = f.get("name", "?")
                size = f.get("size", "?")
                created = f.get("creationDate", "")
                lines.append(f"| `{name}` | {size} | {created} |")
            lines.append("")
        elif self.api_data.get("data_downloaded"):
            lines += ["Files downloaded successfully — see the `data/` folder.", ""]
        else:
            lines += [
                "⚠️ **Data files were not downloaded.**",
                "",
                "Possible reasons:",
                f"- You need to accept the competition rules at:  ",
                f"  <https://www.kaggle.com/competitions/{self.slug}/rules>",
                "- Kaggle API credentials are not configured  ",
                "  (`~/.kaggle/kaggle.json` missing or invalid).",
                "",
            ]

        # List actually present data files on disk
        data_dir = self.output_dir / "data"
        if data_dir.exists():
            disk_files = sorted(data_dir.rglob("*"))
            disk_files = [f for f in disk_files if f.is_file()]
            if disk_files:
                lines += ["### Files on disk", ""]
                for f in disk_files:
                    rel = f.relative_to(self.output_dir)
                    size_kb = f.stat().st_size / 1024
                    lines.append(f"- `{rel}`  ({size_kb:,.1f} KB)")
                lines.append("")

    def _section_pages(self, lines: list[str]) -> None:
        lines += ["## Collected Pages", ""]
        any_collected = False

        for tab_key, tab_label in _TAB_LABELS.items():
            tab_data = self.browser_data.get(tab_key, {})
            if not tab_data:
                icon, note = "⬜", "not collected"
            elif "error" in tab_data:
                icon, note = "❌", tab_data["error"]
            elif tab_data.get("requires_acceptance"):
                icon, note = "⚠️ ", "requires rule/competition acceptance"
            else:
                chars = tab_data.get("content_length", 0)
                icon, note = "✅", f"{chars:,} chars → `pages/{tab_key}.md`"
                any_collected = True
            lines.append(f"- {icon} **{tab_label}:** {note}")

        if not any_collected and not self.browser_data:
            lines += [
                "",
                "> Browser collection was skipped (--no-browser flag or login failure).",
            ]
        lines.append("")

    def _section_screenshots(self, lines: list[str]) -> None:
        ss_dir = self.output_dir / "screenshots"
        if not ss_dir.exists():
            return
        pngs = sorted(ss_dir.glob("*.png"))
        if not pngs:
            return
        lines += ["## Screenshots", ""]
        for f in pngs:
            lines.append(f"- `screenshots/{f.name}`")
        lines.append("")

    def _section_discussion(self, lines: list[str]) -> None:
        disc = self.browser_data.get("discussion", {})
        pinned: list[str] = disc.get("pinned_topics", [])
        if not pinned:
            return
        lines += ["## Discussion — Visible Thread Titles", ""]
        for i, title in enumerate(pinned, 1):
            lines.append(f"{i}. {title}")
        lines.append("")

    def _section_tree(self, lines: list[str]) -> None:
        lines += ["## Output Directory Structure", "", "```"]
        lines += self._build_tree(self.output_dir)
        lines += ["```", ""]

    def _section_footer(self, lines: list[str]) -> None:
        lines += [
            "---",
            "",
            "*Generated by [kaggle_competition_collector](https://github.com/)*",
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Directory tree helper
    # ──────────────────────────────────────────────────────────────────────────

    def _build_tree(
        self,
        directory: Path,
        prefix: str = "",
        max_depth: int = 3,
        depth: int = 0,
    ) -> list[str]:
        if not directory.is_dir() or depth > max_depth:
            return []
        children = sorted(directory.iterdir())
        lines: list[str] = []
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{child.name}")
            if child.is_dir() and depth < max_depth:
                extension = "    " if is_last else "│   "
                lines.extend(self._build_tree(child, prefix + extension, max_depth, depth + 1))
        return lines
