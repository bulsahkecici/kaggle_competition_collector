"""
Rules and risk analyzer.

Scans the competition rules page text and API metadata for specific questions
a data scientist needs to answer before starting work.

Outputs: RULES_RISK_REPORT.md
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("kaggle_collector")

# Status labels
ALLOWED = "✅ ALLOWED"
NOT_ALLOWED = "❌ NOT ALLOWED"
UNCLEAR = "⚠️  UNCLEAR"
NOT_FOUND = "❔ NOT FOUND"


# ──────────────────────────────────────────────────────────────────────────────
# Rule question definitions
#
# Each entry:
#   key          : machine-readable key
#   label        : human label for the report
#   allowed_re   : if matched → ALLOWED
#   not_allowed_re : if matched → NOT ALLOWED
#   unclear_re   : if matched → UNCLEAR (before allowed/not-allowed check)
#   quote_re     : optional pattern whose match is quoted verbatim in the report
# ──────────────────────────────────────────────────────────────────────────────
_CHECKS: list[dict[str, Any]] = [
    {
        "key": "external_data",
        "label": "External data allowed",
        "allowed_re": r"external\s+data\s+(?:is\s+)?(?:allowed|permitted|accepted|ok)",
        "not_allowed_re": (
            r"no\s+external\s+data|external\s+data\s+(?:is\s+)?(?:not\s+allowed|prohibited|forbidden)"
            r"|(?:only|solely)\s+(?:use\s+)?(?:the\s+)?(?:provided|competition|official)\s+data"
        ),
        "unclear_re": r"external\s+data|outside\s+(?:data|dataset|source)",
        "quote_re": r"external\s+data",
    },
    {
        "key": "pretrained_models",
        "label": "Pre-trained models allowed",
        "allowed_re": r"pre-?trained\s+model[s]?\s+(?:are\s+)?(?:allowed|permitted|ok)",
        "not_allowed_re": (
            r"no\s+pre-?trained|pre-?trained\s+(?:model[s]?\s+)?(?:are\s+)?(?:not\s+allowed|prohibited|forbidden)"
        ),
        "unclear_re": r"pre-?trained|pretrain",
        "quote_re": r"pre-?trained",
    },
    {
        "key": "internet_in_notebooks",
        "label": "Internet access in notebooks",
        "allowed_re": r"internet\s+(?:access\s+)?(?:is\s+)?(?:allowed|enabled|permitted)",
        "not_allowed_re": (
            r"no\s+internet|internet\s+(?:access\s+)?(?:is\s+)?(?:not\s+allowed|disabled|prohibited)"
        ),
        "unclear_re": r"internet\s+access|notebook.*internet",
        "quote_re": r"internet",
    },
    {
        "key": "teams_allowed",
        "label": "Teams allowed",
        "allowed_re": r"team[s]?\s+(?:are\s+)?(?:allowed|permitted|welcome)",
        "not_allowed_re": r"no\s+team[s]?|team[s]?\s+(?:are\s+)?not\s+allowed|solo\s+only",
        "unclear_re": r"\bteam\b",
        "quote_re": None,
    },
    {
        "key": "max_team_size",
        "label": "Maximum team size",
        "allowed_re": None,
        "not_allowed_re": None,
        "unclear_re": r"(?:maximum|max)\s+(?:team\s+)?size|team\s+(?:of\s+)?\d+",
        "quote_re": r"(?:maximum|max)\s+team\s+size|team\s+(?:of\s+)?\d+",
    },
    {
        "key": "team_merger_deadline",
        "label": "Team merger deadline",
        "allowed_re": None,
        "not_allowed_re": None,
        "unclear_re": r"merger\s+deadline|team\s+merger",
        "quote_re": r"merger",
    },
    {
        "key": "submission_limit_final",
        "label": "Final submission limit",
        "allowed_re": None,
        "not_allowed_re": None,
        "unclear_re": r"final\s+submission|select\s+\d+\s+submission",
        "quote_re": r"final\s+submission[s]?",
    },
    {
        "key": "submission_limit_daily",
        "label": "Daily submission limit",
        "allowed_re": None,
        "not_allowed_re": None,
        "unclear_re": r"daily\s+submission|per\s+day",
        "quote_re": r"daily\s+submission[s]?|\d+\s+submission[s]?\s+per\s+day",
    },
    {
        "key": "code_sharing",
        "label": "Code sharing restrictions",
        "allowed_re": r"code\s+sharing\s+(?:is\s+)?(?:allowed|permitted|encouraged)",
        "not_allowed_re": (
            r"(?:private|no)\s+code\s+sharing|code\s+sharing\s+(?:is\s+)?(?:not\s+allowed|prohibited)"
        ),
        "unclear_re": r"code\s+shar|kernel\s+shar|notebook\s+shar",
        "quote_re": r"code\s+shar|kernel\s+shar",
    },
    {
        "key": "private_sharing",
        "label": "Private sharing restrictions",
        "allowed_re": r"private\s+sharing\s+(?:is\s+)?(?:allowed|permitted)",
        "not_allowed_re": (
            r"(?:no\s+private\s+shar|private\s+shar.*not\s+allowed|private\s+shar.*prohibited)"
        ),
        "unclear_re": r"private\s+shar",
        "quote_re": r"private\s+shar",
    },
    {
        "key": "data_usage_restrictions",
        "label": "Data usage restrictions (non-commercial / research only)",
        "allowed_re": r"data\s+(?:may\s+be\s+)?used\s+(?:for\s+)?(?:any|commercial)",
        "not_allowed_re": r"non-?commercial|research\s+only|data\s+(?:must\s+not|may\s+not)\s+be\s+used",
        "unclear_re": r"data\s+(?:usage|use)\s+(?:restrict|condition|licen)|licen[cs]e",
        "quote_re": r"non-?commercial|data.*licen",
    },
    {
        "key": "code_competition",
        "label": "Code-only competition (notebook submission)",
        "allowed_re": r"notebook\s+(?:only|submission)|code\s+competition|kernels\s+only",
        "not_allowed_re": None,
        "unclear_re": r"notebook|kernel\s+submission",
        "quote_re": None,
    },
    {
        "key": "reproducibility",
        "label": "Reproducibility / seed requirements",
        "allowed_re": None,
        "not_allowed_re": None,
        "unclear_re": r"reproducib|random\s+seed|deterministic",
        "quote_re": r"reproducib",
    },
    {
        "key": "data_download_allowed",
        "label": "Downloading competition data to local machine allowed",
        "allowed_re": r"data\s+(?:may\s+be\s+)?downloaded|download.*allowed",
        "not_allowed_re": r"data\s+(?:must\s+not|may\s+not)\s+be\s+downloaded|no\s+downloading",
        "unclear_re": r"download",
        "quote_re": None,
    },
]


class RulesAnalyzer:
    """Parses competition rules and generates the risk report."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def analyze(
        self,
        rules_text: str,
        api_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Analyze rules_text + API metadata, write RULES_RISK_REPORT.md.

        Returns a dict of {key: status_string}.
        """
        findings: dict[str, Any] = {}

        for check in _CHECKS:
            status, quote = self._evaluate(check, rules_text)
            # Augment from API metadata where possible
            status, quote = self._augment_from_api(check["key"], status, quote, api_metadata)
            findings[check["key"]] = {"status": status, "quote": quote, "label": check["label"]}

        self._write_report(rules_text, findings, api_metadata)
        return {k: v["status"] for k, v in findings.items()}

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation logic
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate(
        self, check: dict[str, Any], text: str
    ) -> tuple[str, Optional[str]]:
        """Return (status, optional_quote) for one check."""
        quote = _first_quote(text, check.get("quote_re"))

        if check.get("not_allowed_re") and re.search(check["not_allowed_re"], text, re.IGNORECASE):
            return NOT_ALLOWED, quote
        if check.get("allowed_re") and re.search(check["allowed_re"], text, re.IGNORECASE):
            return ALLOWED, quote
        if check.get("unclear_re") and re.search(check["unclear_re"], text, re.IGNORECASE):
            return UNCLEAR, quote
        return NOT_FOUND, None

    def _augment_from_api(
        self,
        key: str,
        status: str,
        quote: Optional[str],
        meta: dict[str, Any],
    ) -> tuple[str, Optional[str]]:
        """Refine status using structured API metadata when available."""
        if key == "max_team_size" and meta.get("maxTeamSize"):
            val = meta["maxTeamSize"]
            return f"Max team size: **{val}**", None

        if key == "team_merger_deadline":
            val = meta.get("mergerDeadline") or meta.get("merger_deadline")
            if val:
                return f"Merger deadline: **{val}** (from API)", None

        if key == "submission_limit_daily" and meta.get("maxDailySubmissions"):
            val = meta["maxDailySubmissions"]
            return f"Daily limit: **{val}** submission(s)", None

        if key == "code_competition":
            is_kernel_only = meta.get("isKernelsSubmissionsOnly")
            if is_kernel_only is True:
                return ALLOWED + " (Notebooks-only competition)", None
            if is_kernel_only is False:
                return NOT_ALLOWED + " (File submission, not notebook-only)", None

        if key == "teams_allowed" and meta.get("maxTeamSize"):
            sz = meta["maxTeamSize"]
            if sz and int(sz) > 1:
                return ALLOWED, None

        return status, quote

    # ──────────────────────────────────────────────────────────────────────────
    # Report writer
    # ──────────────────────────────────────────────────────────────────────────

    def _write_report(
        self,
        rules_text: str,
        findings: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        rules_page_failed = not rules_text.strip() or any(
            phrase in rules_text.lower()
            for phrase in ["content extraction failed", "crash", "cloudflare", "navigation failed"]
        )

        lines: list[str] = [
            "# Rules and Risk Report\n",
        ]

        if rules_page_failed:
            lines += [
                "> ⚠️ **Rules page extraction failed.**  ",
                "> The browser could not load the rules page (Kaggle crash/Cloudflare block).  ",
                "> **This analysis is based solely on API metadata — it is incomplete and should NOT be used as a final compliance source.**  ",
                "> Visit https://www.kaggle.com/competitions/"
                + meta.get("ref", "").split("/")[-1] + "/rules to read the full rules.",
                "",
            ]

        lines += [
            "Parsed from competition rules page and API metadata.\n",
            "Status codes: ✅ ALLOWED | ❌ NOT ALLOWED | ⚠️  UNCLEAR | ❔ NOT FOUND\n",
            "---\n",
            "## Quick Reference Table\n",
            "| Question | Status |",
            "|----------|--------|",
        ]
        for f in findings.values():
            lines.append(f"| {f['label']} | {f['status']} |")
        lines.append("")

        # Detailed entries
        lines.append("## Detailed Analysis\n")
        for key, f in findings.items():
            lines.append(f"### {f['label']}")
            lines.append(f"**Status:** {f['status']}\n")
            if f.get("quote"):
                lines += [
                    "**Relevant rule text:**",
                    f"> {f['quote']}",
                    "",
                ]

        # API metadata supplement
        api_fields = [
            ("maxTeamSize", "Max team size (API)"),
            ("maxDailySubmissions", "Max daily submissions (API)"),
            ("mergerDeadline", "Team merger deadline (API)"),
            ("deadline", "Competition deadline (API)"),
            ("isKernelsSubmissionsOnly", "Notebooks-only (API)"),
        ]
        lines.append("## Values from Kaggle API Metadata\n")
        for field, label in api_fields:
            val = meta.get(field)
            if val is not None:
                lines.append(f"- **{label}:** {val}")
        lines.append("")

        # Raw rules text
        if rules_text.strip():
            lines += [
                "## Full Rules Page Text\n",
                "---\n",
                rules_text[:10_000],
                "",
            ]

        _write(self.output_dir / "RULES_RISK_REPORT.md", "\n".join(lines))
        logger.info("RULES_RISK_REPORT.md written")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _first_quote(text: str, pattern: Optional[str]) -> Optional[str]:
    """Return the first sentence containing `pattern`, or None."""
    if not pattern:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        if re.search(pattern, s, re.IGNORECASE):
            s = s.strip()
            return s[:300] if len(s) > 300 else s
    return None


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
