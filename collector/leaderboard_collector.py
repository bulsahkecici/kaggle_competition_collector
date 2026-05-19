"""
Leaderboard and user submission collector.

Uses the Kaggle API to:
  - Download the public leaderboard snapshot
  - Fetch the user's own submission history (delegates to submission_scores.py)

Outputs:
  leaderboard/leaderboard_snapshot.csv
  leaderboard/leaderboard_summary.md
  leaderboard/my_submissions.csv   ← written by SubmissionScoreFetcher
  leaderboard/my_submissions.json  ← written by SubmissionScoreFetcher
  leaderboard/my_submission_trend.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .submission_scores import SubmissionScoreFetcher, _all_scores_missing

logger = logging.getLogger("kaggle_collector")


class LeaderboardCollector:
    """Fetches public leaderboard and personal submission history."""

    def __init__(self, slug: str, output_dir: Path) -> None:
        self.slug = slug
        self.lb_dir = output_dir / "leaderboard"
        self.lb_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def collect(self, page=None) -> dict[str, Any]:
        """
        page : optional live Playwright Page for browser-based score fallback.
        """
        result: dict[str, Any] = {}

        # ── Leaderboard (API only) ────────────────────────────────────────────
        api = self._try_auth()
        if api is not None:
            lb_data = self._get_leaderboard(api)
            if lb_data:
                result["leaderboard"] = lb_data
                self._write_leaderboard_csv(lb_data)
                self._write_leaderboard_summary(lb_data)
        else:
            logger.warning("Kaggle API unavailable; skipping leaderboard snapshot")

        # ── Submissions (multi-strategy via SubmissionScoreFetcher) ───────────
        fetcher = SubmissionScoreFetcher(self.slug, self.lb_dir)
        sub_result = await fetcher.fetch(page=page)

        rows = sub_result.get("rows", [])
        result["my_submissions"] = rows
        result["source"] = sub_result.get("source", "none")
        result["warning"] = sub_result.get("warning")

        if rows:
            self._write_submission_trend(rows)

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Leaderboard
    # ──────────────────────────────────────────────────────────────────────────

    def _get_leaderboard(self, api) -> list[dict[str, Any]]:
        try:
            from .kaggle_api import _normalize_list
            response = api.competition_leaderboard_view(self.slug)
            entries = _normalize_list(response) if not isinstance(response, list) else response
            rows = [self._ser_lb(e, rank) for rank, e in enumerate(entries, 1)]
            if rows:
                logger.info(f"Leaderboard: {len(rows)} entries")
            return rows
        except Exception as exc:
            logger.warning(f"Leaderboard fetch failed: {exc}")
            return []

    def _ser_lb(self, entry, rank: int = 0) -> dict[str, Any]:
        d: dict[str, Any] = {}
        # New SDK uses snake_case; old used camelCase — try both
        d["teamId"] = (
            getattr(entry, "team_id", None) or getattr(entry, "teamId", None)
        )
        d["teamName"] = (
            getattr(entry, "team_name", None) or getattr(entry, "teamName", None)
        )
        d["submissionDate"] = str(
            getattr(entry, "submission_date", None)
            or getattr(entry, "submissionDate", None)
            or ""
        )
        d["score"] = getattr(entry, "score", None)
        # New SDK does not return rank in the entry — compute from sorted position
        d["rank"] = getattr(entry, "rank", None) or rank
        return d

    def _write_leaderboard_csv(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        try:
            import csv
            path = self.lb_dir / "leaderboard_snapshot.csv"
            keys = list(rows[0].keys())
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            logger.info(f"Leaderboard snapshot saved: {path}")
        except Exception as exc:
            logger.warning(f"Could not write leaderboard CSV: {exc}")

    def _write_leaderboard_summary(self, rows: list[dict[str, Any]]) -> None:
        top10 = sorted(rows, key=lambda r: r.get("rank") or 9999)[:10]
        lines: list[str] = [
            "# Leaderboard Summary\n",
            f"**Snapshot time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Total teams on leaderboard:** {len(rows)}\n",
            "",
            "## Top 10\n",
            "| Rank | Team | Score | Date |",
            "|------|------|-------|------|",
        ]
        for r in top10:
            lines.append(
                f"| {r.get('rank','?')} | {r.get('teamName','?')} "
                f"| {r.get('score','?')} | {r.get('submissionDate','?')} |"
            )
        _write(self.lb_dir / "leaderboard_summary.md", "\n".join(lines))

    # ──────────────────────────────────────────────────────────────────────────
    # Submission trend report (uses normalised rows from SubmissionScoreFetcher)
    # ──────────────────────────────────────────────────────────────────────────

    def _write_submission_trend(self, rows: list[dict[str, Any]]) -> None:
        sorted_rows = sorted(rows, key=lambda r: str(r.get("date") or ""), reverse=True)

        lines: list[str] = [
            "# My Submission Trend\n",
            f"**Total submissions:** {len(rows)}\n",
            "",
            "## All Submissions (newest first)\n",
            "| # | Date | Public Score | Status | File / Description |",
            "|---|------|--------------|--------|--------------------|",
        ]
        best_score: Optional[float] = None
        for i, r in enumerate(sorted_rows, 1):
            pub = r.get("publicScore") or ""
            try:
                s = float(pub)
                if best_score is None or s > best_score:
                    best_score = s
            except (TypeError, ValueError):
                pass
            desc = (r.get("fileName") or r.get("description") or "")[:40]
            lines.append(
                f"| {i} | {r.get('date','?')} | {pub or '—'} "
                f"| {r.get('status','?')} | {desc} |"
            )
        lines.append("")
        if best_score is not None:
            lines.append(f"**Best public score:** `{best_score}`\n")
        elif rows and _all_scores_missing(rows):
            lines.append(
                "_⚠️ publicScore is empty for all submissions. "
                "Scores may not yet be revealed for this competition._\n"
            )

        _write(self.lb_dir / "my_submission_trend.md", "\n".join(lines))
        logger.info("my_submission_trend.md written")

    # ──────────────────────────────────────────────────────────────────────────
    # Auth
    # ──────────────────────────────────────────────────────────────────────────

    def _try_auth(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
            api = KaggleApi()
            api.authenticate()
            return api
        except Exception as exc:
            logger.warning(f"Kaggle API auth failed for leaderboard: {exc}")
            return None


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
