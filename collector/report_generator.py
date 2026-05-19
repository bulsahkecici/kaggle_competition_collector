"""
Final report generator.

Assembles all collected information into:
  SUMMARY_REPORT.md        — master AI-ready report
  AI_HANDOFF_INSTRUCTIONS.md — reading guide for the next AI assistant
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("kaggle_collector")


class ReportGenerator:
    """Writes SUMMARY_REPORT.md and AI_HANDOFF_INSTRUCTIONS.md."""

    def __init__(
        self,
        slug: str,
        output_dir: Path,
        api_data: dict[str, Any],
        browser_data: dict[str, Any],
        notebooks: list[dict[str, Any]],
        discussions: list[dict[str, Any]],
        data_profiles: dict[str, Any],
        metric_result: dict[str, Any],
        rules_result: dict[str, Any],
        lb_result: dict[str, Any],
    ) -> None:
        self.slug = slug
        self.out = output_dir
        self.api = api_data
        self.browser = browser_data
        self.notebooks = notebooks
        self.discussions = discussions
        self.profiles = data_profiles
        self.metric = metric_result
        self.rules = rules_result
        self.lb = lb_result

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────────────────────────────────

    def generate_all(self) -> tuple[Path, Path]:
        """Generate both final reports. Returns (summary_path, handoff_path)."""
        summary = self._write_summary()
        handoff = self._write_handoff()
        return summary, handoff

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY_REPORT.md
    # ──────────────────────────────────────────────────────────────────────────

    def _write_summary(self) -> Path:
        meta = self.api.get("metadata", {})
        title = meta.get("title") or self.slug
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        lines += [
            f"# 🏆 Kaggle Competition: {title}",
            "",
            f"> **Slug:** `{self.slug}`  ",
            f"> **Report generated:** {now}  ",
            f"> **Output directory:** `{self.out}`  ",
            "",
            "---",
            "",
        ]

        # ── Competition overview ───────────────────────────────────────────────
        lines.append("## Competition Overview\n")
        for field, label in [
            ("url", "URL"),
            ("deadline", "Deadline"),
            ("enabledDate", "Start date"),
            ("mergerDeadline", "Team merger deadline"),
            ("reward", "Prize / reward"),
            ("teamCount", "Teams entered"),
            ("maxTeamSize", "Max team size"),
            ("maxDailySubmissions", "Max daily submissions"),
            ("evaluationMetric", "Evaluation metric"),
            ("category", "Category"),
            ("organizationName", "Organizer"),
            ("isKernelsSubmissionsOnly", "Notebook-only"),
        ]:
            val = meta.get(field)
            if val is not None:
                lines.append(f"- **{label}:** {val}")
        lines.append("")

        desc = str(meta.get("description") or "").strip()
        if desc:
            lines += ["### Description", "", desc[:2_000], ""]

        tags = meta.get("tags")
        if tags:
            tag_str = ", ".join(f"`{t}`" for t in (tags if isinstance(tags, list) else [tags]))
            lines += [f"**Tags:** {tag_str}", ""]

        # ── Evaluation metric ──────────────────────────────────────────────────
        lines.append("## Evaluation Metric\n")
        raw_m = self.metric.get("raw_metric") or meta.get("evaluationMetric", "")
        entry = self.metric.get("matched_entry")
        if entry:
            hib = "higher is better ✅" if entry["higher_is_better"] else "lower is better ✅"
            lines += [
                f"- **Metric:** {entry['full_name']}",
                f"- **Direction:** {hib}",
                f"- **Task type:** {entry['task_type']}",
                f"- **Prediction format:** {entry['prediction_format']}",
                "",
                f"See **`METRIC_EXPLAINED.md`** for detailed pitfalls, scoring function, and validation tips.",
                "",
            ]
        elif raw_m:
            lines += [
                f"- **Metric:** `{raw_m}` _(not matched to knowledge base)_",
                "- See `METRIC_EXPLAINED.md` and `pages/evaluation.md` for full details.",
                "",
            ]
        else:
            lines += ["_Metric not determined. Check `pages/evaluation.md`._", ""]

        # ── Data files ────────────────────────────────────────────────────────
        lines.append("## Data Files\n")
        files_manifest = self.api.get("files_manifest", [])
        data_dir = self.out / "data"
        disk_files = sorted(data_dir.rglob("*")) if data_dir.exists() else []
        disk_files = [f for f in disk_files if f.is_file()]

        if files_manifest:
            lines += ["| File | Size |", "|------|------|"]
            for f in files_manifest:
                lines.append(f"| `{f.get('name','?')}` | {f.get('size','?')} |")
            lines.append("")
        elif disk_files:
            lines += ["| File | Size (KB) |", "|------|-----------|"]
            for f in disk_files:
                lines.append(f"| `{f.relative_to(self.out)}` | {f.stat().st_size/1024:.1f} |")
            lines.append("")
        else:
            lines += ["_No data files downloaded or listed._", ""]

        # ── Data profile summary ───────────────────────────────────────────────
        if self.profiles:
            lines.append("## Data Profile Summary\n")
            train_k = _fuzzy(self.profiles, "train")
            test_k = _fuzzy(self.profiles, "test")
            sample_k = _fuzzy(self.profiles, "sample_submission")

            for key, label in [(train_k, "Train"), (test_k, "Test"), (sample_k, "Sample Submission")]:
                if key:
                    p = self.profiles[key]
                    lines += [
                        f"### {label}: `{key}`",
                        f"- {p.get('rows', '?'):,} rows × {p.get('columns', '?')} columns",
                        f"- {p.get('file_size_mb', '?')} MB on disk",
                        "",
                    ]
                    has_nulls = any(v > 0 for v in p.get("null_counts", {}).values())
                    if has_nulls:
                        null_cols = [c for c, v in p.get("null_counts", {}).items() if v > 0]
                        lines.append(f"- ⚠️ Columns with missing values: {', '.join(f'`{c}`' for c in null_cols[:10])}")
                        lines.append("")

            # Target column
            target_report = self.out / "data_profile" / "target_detection_report.md"
            if target_report.exists():
                lines += [
                    "### Target Detection",
                    "",
                    target_report.read_text(encoding="utf-8")[:500],
                    "",
                    "_See `data_profile/target_detection_report.md` for full details._",
                    "",
                ]

        # ── Rules quick table ─────────────────────────────────────────────────
        if self.rules:
            lines.append("## Rules Summary\n")
            key_rules = [
                ("external_data", "External data"),
                ("pretrained_models", "Pre-trained models"),
                ("internet_in_notebooks", "Internet in notebooks"),
                ("teams_allowed", "Teams allowed"),
                ("max_team_size", "Max team size"),
                ("code_competition", "Notebook-only competition"),
                ("code_sharing", "Code sharing"),
                ("data_usage_restrictions", "Data usage restrictions"),
            ]
            lines += ["| Rule | Status |", "|------|--------|"]
            for key, label in key_rules:
                status = self.rules.get(key, "❔ NOT FOUND")
                lines.append(f"| {label} | {status} |")
            lines += ["", "See **`RULES_RISK_REPORT.md`** for full rule text and analysis.", ""]

        # ── Submission format ──────────────────────────────────────────────────
        sub_fmt = self.out / "SUBMISSION_FORMAT.md"
        if sub_fmt.exists():
            lines += [
                "## Submission Format",
                "",
                sub_fmt.read_text(encoding="utf-8")[:600],
                "",
                "_(See `SUBMISSION_FORMAT.md` for full details.)_",
                "",
            ]

        # ── Notebooks summary ──────────────────────────────────────────────────
        if self.notebooks:
            lines.append("## Public Notebooks Collected\n")
            lines.append(f"Total: **{len(self.notebooks)}** notebooks  →  see `CODE_NOTEBOOKS_SUMMARY.md`\n")
            top3 = self.notebooks[:3]
            for nb in top3:
                lines += [
                    f"- **{nb.get('title','?')}** by {nb.get('author','?')} "
                    f"({nb.get('totalVotes','?')} votes) [{nb.get('_found_via','')}]"
                ]
            lines.append("")

        # ── Discussion summary ─────────────────────────────────────────────────
        if self.discussions:
            lines.append("## Discussion Highlights\n")
            lines.append(f"Collected: **{len(self.discussions)}** threads  →  see `discussions/`\n")
            for d in self.discussions[:5]:
                title = d.get("title", "?")
                votes = d.get("votes", "?")
                lines.append(f"- **{title}** (votes: {votes})")
            lines.append("")
            # Read insight snippets
            insights_file = self.out / "discussions" / "discussion_insights.md"
            if insights_file.exists():
                lines += [
                    "### Key Discussion Insights",
                    "",
                    insights_file.read_text(encoding="utf-8")[:1_500],
                    "",
                    "_(See `discussions/discussion_insights.md` for full analysis.)_",
                    "",
                ]

        # ── Recommended approach ───────────────────────────────────────────────
        lines += [
            "## Recommended First Steps for a Data Scientist",
            "",
            "1. **Read the evaluation metric carefully** → `METRIC_EXPLAINED.md`",
            "2. **Understand submission format** → `SUBMISSION_FORMAT.md`",
            "3. **Explore the data** → `data_profile/` for shape, nulls, types, targets",
            "4. **Check rules** → `RULES_RISK_REPORT.md` before using external data or models",
            "5. **Scan public notebooks** → `CODE_NOTEBOOKS_SUMMARY.md` for baselines and ideas",
            "6. **Mine discussions** → `discussions/discussion_insights.md` for community wisdom",
            "7. **Set up your validation** aligned with the metric (see `METRIC_EXPLAINED.md`)",
            "8. **Build a minimal baseline** before tuning — track CV and LB correlation",
            "",
            "## Possible Leakage Warnings",
            "",
        ]
        # Leakage mentions from discussions
        disc_text = " ".join(d.get("content", "")[:500] for d in self.discussions)
        if "leak" in disc_text.lower():
            lines += [
                "⚠️ **The word 'leak' or 'leakage' appears in community discussions.**",
                "Review `discussions/discussion_insights.md` section 'Data issues and leakage warnings'.",
                "",
            ]
        else:
            lines.append("No explicit leakage warnings found in collected discussions.\n")

        # ── Known collection gaps ──────────────────────────────────────────────
        lines += [
            "## Known Collection Gaps",
            "",
            "The following limitations apply to this collection. Review before relying on any section:",
            "",
        ]
        pages_dir = self.out / "pages"
        crashed_tabs: list[str] = []
        _crash_phrases = [
            "quality: `failed`", "quality: failed",
            "content extraction failed", "something went wrong and this page crashed",
            "unexpected token '<'", "cloudflare", "navigation failed",
            "page appears to be a crash",
        ]
        for tab in ["overview", "data", "evaluation", "rules", "code", "discussion", "leaderboard"]:
            p = pages_dir / f"{tab}.md"
            if p.exists():
                txt = p.read_text(encoding="utf-8", errors="replace").lower()
                if any(phrase in txt for phrase in _crash_phrases):
                    crashed_tabs.append(tab)

        all_tabs = ["overview", "data", "evaluation", "rules", "code", "discussion", "leaderboard"]
        all_tabs_crashed = len(crashed_tabs) == len(all_tabs)

        if crashed_tabs:
            lines += [
                f"- **Browser tab extraction failed for: {', '.join(f'`{t}`' for t in crashed_tabs)}.**  ",
                "  Kaggle returned a crash / Cloudflare page for these tabs in headless mode.  ",
                "  Re-run with `--headed --refresh-browser-state --pause-on-browser-warning` to attempt manual extraction.",
                "",
            ]

        # Screenshot validity note
        ss_dir = self.out / "screenshots"
        ss_files = sorted(ss_dir.glob("*.png")) if ss_dir.exists() else []
        if ss_files and crashed_tabs:
            if all_tabs_crashed:
                lines += [
                    "- **⚠️ SCREENSHOTS ARE NOT VALID.**  ",
                    f"  Browser tab extraction failed for all {len(crashed_tabs)} tabs.  ",
                    "  All screenshots (`screenshots/*.png`) are crash screens / Cloudflare pages and should **not** be used for any analysis.  ",
                    "  To obtain real screenshots, re-run with `--headed --refresh-browser-state`.",
                    "",
                ]
            else:
                invalid_ss = [f"`screenshots/{t}.png`" for t in crashed_tabs]
                lines += [
                    f"- **Screenshots for crashed tabs are not valid:** {', '.join(invalid_ss)}.  ",
                    "  These show crash screens. Screenshots for non-crashed tabs may be usable.",
                    "",
                ]
        elif not ss_files:
            lines += [
                "- **No screenshots collected** (either `--no-screenshots` was set or browser was unavailable).",
                "",
            ]

        lines += [
            "- **API metadata and downloaded data files are reliable** — metric, leaderboard, submission history, and data CSVs are collected via the Kaggle API, not the browser.",
            "- **Discussion threads** are partially useful but check `discussions/discussion_quality_report.md` for duplicate/comment-anchor removal stats.",
            "- **Notebook comments** are unreliable — Kaggle's comment DOM is difficult to target precisely in headless mode. See each `comments.md` file.",
            "- **`CODE_NOTEBOOKS_SUMMARY.md`** is generated from actual notebook parsing (models, CV, feature engineering, blend patterns, score mentions).",
            "- **Rules analysis** is based on API metadata only if the rules page failed — treat `RULES_RISK_REPORT.md` as incomplete.",
            "",
            "---",
            "",
            "_Generated by kaggle_competition_collector_",
        ]

        path = self.out / "SUMMARY_REPORT.md"
        _write(path, "\n".join(lines))
        logger.info("SUMMARY_REPORT.md written")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # AI_HANDOFF_INSTRUCTIONS.md
    # ──────────────────────────────────────────────────────────────────────────

    def _write_handoff(self) -> Path:
        meta = self.api.get("metadata", {})
        title = meta.get("title") or self.slug
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines: list[str] = [
            "# AI Handoff Instructions",
            "",
            f"This package was collected from Kaggle for competition: **{title}** (`{self.slug}`)  ",
            f"Generated: {now}",
            "",
            "---",
            "",
            "## Purpose",
            "",
            "You are an AI assistant helping a data scientist analyze and solve this Kaggle competition.",
            "All files in this package were collected from the competition's public pages and the Kaggle API.",
            "Nothing private or restricted is included.",
            "",
            "---",
            "",
            "## Recommended Reading Order",
            "",
            "Read the files in this order for full context:\n",
            "### 1. `SUMMARY_REPORT.md`",
            "Start here. Contains the competition overview, metric, data summary, rules, and first-step recommendations.",
            "",
            "### 2. `METRIC_EXPLAINED.md`",
            "Deep-dive on the evaluation metric: direction, pitfalls, scoring function, and validation strategy.",
            "",
            "### 3. `RULES_RISK_REPORT.md`",
            "What is and isn't allowed: external data, pre-trained models, internet, team sizes, etc.",
            "",
            "### 4. `SUBMISSION_FORMAT.md`",
            "Exact structure required for valid submission CSV. Always check this before finalizing predictions.",
            "",
            "### 5. `data_profile/`",
            "- `target_detection_report.md` — what the target column(s) are",
            "- `train_test_column_diff.md` — features available at inference time",
            "- `missing_values_report.md` — nulls to handle",
            "- `file_profiles/*.md` — row/column counts, dtypes, sample rows",
            "",
            "### 6. `CODE_NOTEBOOKS_SUMMARY.md`",
            "Summary of top public notebooks: models used, feature engineering, CV strategy, ensemble ideas.",
            "",
            "### 7. `discussions/discussion_insights.md`",
            "Key insights, warnings, clarifications, and leakage reports from the community.",
            "",
            "### 8. `pages/`",
            "Full raw text from each competition tab (overview, data, evaluation, rules, leaderboard).",
            "Use these as the ground truth if any summary seems incomplete.",
            "",
            "### 9. `code_notebooks/`",
            "Individual notebook files. Read `notebook_code.py` for code and `notebook.md` for explanations.",
            "",
            "### 10. `discussions/threads/`",
            "Full text of the most important discussion threads.",
            "",
            "---",
            "",
            "## Key Questions to Answer",
            "",
            "After reading, you should be able to answer:",
            "- What is the target variable and its format?",
            "- What is the evaluation metric and which direction is better?",
            "- What files are available for training and testing?",
            "- Are there missing values? How many?",
            "- What features are available at test time vs. train time only?",
            "- What approaches have worked in public notebooks?",
            "- Are there any known data leaks or common pitfalls?",
            "- What validation strategy should we use?",
            "- What is a reasonable first baseline model?",
            "",
            "---",
            "",
            "## Important Warnings",
            "",
            "- Do NOT use any information that was not visible on the Kaggle page at collection time.",
            "- Do NOT submit solutions automatically.",
            "- Verify rule compliance before using external data or pre-trained models.",
            "- If a 'requires_acceptance' flag appears in browser data, some content may be incomplete.",
            "",
            "---",
            "",
            "## Files in This Package",
            "",
        ]

        # Directory tree
        lines.append("```")
        lines.extend(_tree(self.out))
        lines += ["```", ""]

        lines += ["---", "", "_Generated by kaggle_competition_collector_"]

        path = self.out / "AI_HANDOFF_INSTRUCTIONS.md"
        _write(path, "\n".join(lines))
        logger.info("AI_HANDOFF_INSTRUCTIONS.md written")
        return path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _fuzzy(d: dict, keyword: str):
    for k in d:
        if keyword.lower().replace("_", "") in k.lower().replace("_", ""):
            return k
    return None


def _tree(directory: Path, prefix: str = "", depth: int = 0, max_depth: int = 3) -> list[str]:
    if not directory.is_dir() or depth > max_depth:
        return []
    children = sorted(directory.iterdir())
    lines = []
    for i, child in enumerate(children):
        if child.name.endswith(".sha256"):
            continue
        connector = "└── " if i == len(children) - 1 else "├── "
        lines.append(f"{prefix}{connector}{child.name}")
        if child.is_dir() and depth < max_depth:
            ext = "    " if i == len(children) - 1 else "│   "
            lines.extend(_tree(child, prefix + ext, depth + 1, max_depth))
    return lines


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
