"""
Deep discussion collector.

Visits the competition discussion tab, scrapes thread listings sorted by votes,
then dives into the highest-priority threads (filtered by keyword relevance).

Outputs:
  discussions/threads/thread_NNN.md
  discussions/top_threads_summary.md
  discussions/discussion_insights.md
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .utils import polite_delay

logger = logging.getLogger("kaggle_collector")

KAGGLE_BASE = "https://www.kaggle.com"

# Threads matching these keywords are prioritised for full collection
PRIORITY_KEYWORDS: list[str] = [
    "solution", "winning", "leak", "leakage", "metric", "evaluation",
    "cv", "lb", "leaderboard", "baseline", "eda", "feature engineering",
    "ensemble", "inference", "submission", "rule", "clarification",
    "data issue", "overfitting", "blending", "stacking", "target",
    "external data", "pretrained", "public score", "private score",
]

# JS to extract thread links from the discussion list page
_LIST_JS = r"""
() => {
    const seen = new Set();
    const items = [];
    document.querySelectorAll('a[href]').forEach(a => {
        const href = a.getAttribute('href') || '';
        if (!href.match(/\/discussion\/\d+/)) return;
        if (seen.has(href)) return;
        seen.add(href);
        // Walk up to find surrounding metadata (votes, comments)
        let el = a;
        let votes = '';
        let comments = '';
        for (let i = 0; i < 6; i++) {
            if (!el.parentElement) break;
            el = el.parentElement;
            const t = el.innerText || '';
            const vm = t.match(/(\d+)\s*(vote|upvote)/i);
            const cm = t.match(/(\d+)\s*(comment|reply|replies)/i);
            if (vm) votes = vm[1];
            if (cm) comments = cm[1];
        }
        items.push({
            href: href.startsWith('http') ? href : 'https://www.kaggle.com' + href,
            title: a.innerText.trim(),
            votes: votes,
            comments: comments,
        });
    });
    return items;
}
"""

# JS to extract full thread content from a single thread page
_THREAD_JS = """
() => {
    const clone = document.body.cloneNode(true);
    clone.querySelectorAll('script,style,nav,footer,header,[role="navigation"],[aria-hidden="true"]')
         .forEach(el => el.remove());
    return clone.innerText;
}
"""


class DiscussionCollector:
    """Collects and saves competition discussion threads."""

    def __init__(
        self,
        slug: str,
        output_dir: Path,
        max_threads: int = 25,
        delay: float = 2.5,
    ) -> None:
        self.slug = slug
        self.output_dir = output_dir
        self.disc_dir = output_dir / "discussions"
        self.threads_dir = self.disc_dir / "threads"
        self.threads_dir.mkdir(parents=True, exist_ok=True)
        self.max_threads = max_threads
        self.delay = delay

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def collect(self, page) -> list[dict[str, Any]]:
        """
        Requires a logged-in Playwright Page.
        Returns list of thread metadata dicts.
        """
        # Always clean stale threads from a previous run BEFORE writing new ones.
        # This prevents old thread_*.md files from being mistaken for current output
        # even when --keep-latest is passed or clean_run_dirs was not called.
        self._clean_stale_threads()

        thread_list, list_stats = await self._get_thread_list(page)
        if not thread_list:
            page_crashed = list_stats.get("page_crashed", False)
            if page_crashed:
                logger.warning(
                    "0 discussions collected because the discussion page could not "
                    "be extracted due to Kaggle crash/Cloudflare page"
                )
            else:
                logger.warning("No discussion threads found (competition may have no discussions)")
            self._write_empty(page_crashed=page_crashed)
            self._write_quality_report([], list_stats, 0)
            return []

        # Rank by keyword relevance + votes
        ranked = _rank_threads(thread_list, self.max_threads)
        logger.info(f"Collecting {len(ranked)} discussion threads …")

        collected: list[dict[str, Any]] = []
        failed = 0
        for i, meta in enumerate(ranked, 1):
            title = meta.get("title", f"thread_{i:03d}")
            url = meta.get("href", "")
            logger.info(f"  [{i}/{len(ranked)}] {title[:70]}")

            content = await self._get_thread_content(page, url)
            meta["content"] = content
            if content.startswith("_Error"):
                failed += 1

            fname = f"thread_{i:03d}.md"
            _write(self.threads_dir / fname, _format_thread(i, meta))
            meta["file"] = fname
            collected.append(meta)
            polite_delay(self.delay)

        self._write_top_summary(collected)
        self._write_insights(collected)
        self._write_quality_report(collected, list_stats, failed)
        return collected

    # ──────────────────────────────────────────────────────────────────────────
    # Browser helpers
    # ──────────────────────────────────────────────────────────────────────────

    _CRASH_PHRASES = [
        "something went wrong",
        "unexpected token '<'",
        "cloudflare",
        "just a moment",
        "checking your browser",
        "navigation failed",
        "an unexpected error",
    ]

    async def _get_thread_list(self, page) -> tuple[list[dict[str, Any]], dict]:
        """
        Navigate to /discussion?sort=votes and extract thread links.
        Returns (deduplicated_threads, stats_dict).
        """
        raw_items: list[dict[str, Any]] = []
        seen_hrefs: set[str] = set()
        stats = {"raw": 0, "comment_anchors": 0, "duplicate_ids": 0, "page_crashed": False}
        crash_count = 0

        for sort in ("votes", "comments", "hot"):
            url = f"{KAGGLE_BASE}/competitions/{self.slug}/discussion?sort={sort}"
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=40_000)
                await page.wait_for_timeout(3_500)
                await _scroll(page)

                # Check for crash/Cloudflare before scraping
                page_text = ""
                try:
                    page_text = (await page.evaluate("() => document.body.innerText")).lower()
                except Exception:
                    pass
                if any(phrase in page_text for phrase in self._CRASH_PHRASES):
                    logger.warning(f"  Discussion list (sort={sort}): crash/Cloudflare page detected — skipping")
                    crash_count += 1
                    continue

                items: list[dict[str, Any]] = await page.evaluate(_LIST_JS)
                for item in items:
                    href = item.get("href", "")
                    title = (item.get("title") or "").strip()
                    stats["raw"] += 1

                    # Skip #comment anchor links (not separate threads)
                    if "#comment" in href:
                        stats["comment_anchors"] += 1
                        continue

                    # Skip entries titled only "comment"
                    if title.lower() in ("comment", "comments", ""):
                        stats["comment_anchors"] += 1
                        continue

                    # Deduplicate by base thread ID (strip query params / anchors)
                    base_href = href.split("#")[0].split("?")[0].rstrip("/")
                    if base_href in seen_hrefs:
                        stats["duplicate_ids"] += 1
                        continue

                    seen_hrefs.add(base_href)
                    item["href"] = base_href  # normalise
                    raw_items.append(item)

                logger.debug(f"  sort={sort}: +{len(items)} raw links")
            except Exception as exc:
                logger.warning(f"  Discussion list (sort={sort}) failed: {exc}")
                crash_count += 1
            polite_delay(1.5)

        # If all 3 sort pages crashed, flag it
        if crash_count >= 3:
            stats["page_crashed"] = True

        return raw_items, stats

    async def _get_thread_content(self, page, url: str) -> str:
        """Navigate to a thread URL and return its full visible text."""
        if not url:
            return ""
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=40_000)
            await page.wait_for_timeout(3_000)
            await _scroll(page)
            text: str = await page.evaluate(_THREAD_JS)
            return text[:30_000]  # cap per thread
        except Exception as exc:
            logger.warning(f"  Thread content failed for {url}: {exc}")
            return f"_Error collecting thread: {exc}_"

    # ──────────────────────────────────────────────────────────────────────────
    # Stale-file cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_stale_threads(self) -> None:
        """
        Remove all thread_*.md files from a previous run.
        Called at the very start of collect() so the output always reflects
        exactly what the current run collected — no stale leftovers.
        """
        removed = 0
        for old_file in sorted(self.threads_dir.glob("thread_*.md")):
            try:
                old_file.unlink()
                removed += 1
            except Exception as exc:
                logger.debug(f"  Could not remove stale thread file {old_file}: {exc}")
        if removed:
            logger.info(f"  Removed {removed} stale thread file(s) from previous run")

        # Also remove previous summary/insight/quality files so they're always regenerated
        for fname in ("top_threads_summary.md", "discussion_insights.md", "discussion_quality_report.md"):
            fpath = self.disc_dir / fname
            if fpath.exists():
                try:
                    fpath.unlink()
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────────────────────────
    # Report writers
    # ──────────────────────────────────────────────────────────────────────────

    def _write_empty(self, page_crashed: bool = False) -> None:
        if page_crashed:
            msg = (
                "_0 discussions collected. The discussion page could not be extracted "
                "due to a Kaggle crash / Cloudflare page. "
                "Re-run with `--headed --refresh-browser-state` to retry._"
            )
        else:
            msg = "_No discussion threads collected (competition may have no discussions yet)._"
        _write(self.disc_dir / "top_threads_summary.md", msg)
        _write(self.disc_dir / "discussion_insights.md", msg)

    def _write_quality_report(
        self,
        collected: list[dict[str, Any]],
        list_stats: dict,
        failed: int,
    ) -> None:
        valid = len(collected) - failed
        page_crashed = list_stats.get("page_crashed", False)
        lines = [
            "# Discussion Quality Report\n",
        ]
        if page_crashed:
            lines += [
                "**⚠️ Root cause: discussion page could not be extracted (Kaggle crash / Cloudflare page).**",
                "0 discussions were collected because the browser returned a crash screen for all sort orders.",
                "Re-run with `--headed --refresh-browser-state` to retry.",
                "",
            ]
        lines += [
            f"**Total raw links found:** {list_stats.get('raw', 0)}",
            f"**Comment-anchor links removed** (`#comment` or title='comment'): {list_stats.get('comment_anchors', 0)}",
            f"**Duplicate thread IDs removed:** {list_stats.get('duplicate_ids', 0)}",
            f"**Unique discussion threads:** {len(collected)}",
            f"**Failed thread fetches:** {failed}",
            f"**Valid threads saved:** {valid}",
            "",
            "---",
            "",
        ]
        if collected:
            lines += ["## Saved Threads\n", "| # | Title | Votes | File |",
                      "|---|-------|-------|------|"]
            for i, t in enumerate(collected, 1):
                title = t.get("title", "?")
                votes = t.get("votes", "?")
                fname = t.get("file", "?")
                ok = "✅" if not t.get("content", "").startswith("_Error") else "❌"
                lines.append(f"| {i} | {title[:60]} | {votes} | {ok} `{fname}` |")
        _write(self.disc_dir / "discussion_quality_report.md", "\n".join(lines))
        logger.info("discussion_quality_report.md written")

    def _write_top_summary(self, threads: list[dict[str, Any]]) -> None:
        lines: list[str] = [
            "# Top Discussion Threads Summary\n",
            f"Collected: **{len(threads)}** threads\n",
            "---\n",
        ]
        for i, t in enumerate(threads, 1):
            title = t.get("title", "?")
            href = t.get("href", "")
            votes = t.get("votes", "?")
            comments = t.get("comments", "?")
            keywords = ", ".join(_matching_keywords(title + " " + t.get("content", "")[:500]))
            lines += [
                f"### {i}. {title}",
                f"- **URL:** {href}",
                f"- **Votes:** {votes}  |  **Comments:** {comments}",
                f"- **Keywords:** {keywords or '—'}",
                f"- **File:** `threads/{t.get('file', '?')}`",
                "",
            ]
        _write(self.disc_dir / "top_threads_summary.md", "\n".join(lines))

    def _write_insights(self, threads: list[dict[str, Any]]) -> None:
        all_text = "\n\n".join(
            f"=== {t.get('title', '')} ===\n{t.get('content', '')}" for t in threads
        )

        sections: dict[str, list[str]] = {
            "Metric / Evaluation clarifications": ["metric", "evaluation", "score", "loss", "auc"],
            "Data issues and leakage warnings": ["leak", "leakage", "data issue", "duplicate"],
            "Baseline references": ["baseline", "simple model", "starter"],
            "EDA insights": ["eda", "exploratory", "distribution", "missing"],
            "Feature engineering ideas": ["feature", "engineerin", "transform", "encoding"],
            "Validation strategy": ["cv", "cross-valid", "kfold", "stratified", "fold"],
            "Ensemble / blending": ["ensemble", "blend", "stack", "voting"],
            "Rule clarifications": ["rule", "allowed", "external data", "pretrained"],
            "Public LB vs CV discussion": ["lb", "leaderboard", "private", "public score"],
            "Common traps / warnings": ["overfitting", "leak", "pitfall", "careful", "warning"],
            "External resources mentioned": ["github", "arxiv", "paper", "blog", "reference"],
        }

        lines: list[str] = [
            "# Discussion Insights\n",
            f"Synthesised from {len(threads)} collected threads.\n",
            "---\n",
        ]

        for section, kws in sections.items():
            hits = [
                t.get("title", "?")
                for t in threads
                if any(
                    kw in (t.get("title", "") + t.get("content", ""))[:3000].lower()
                    for kw in kws
                )
            ]
            lines.append(f"## {section}\n")
            if hits:
                for h in hits[:8]:
                    lines.append(f"- {h}")
            else:
                lines.append("_No relevant threads found._")
            lines.append("")

        # Extract sentences containing key phrases
        lines.append("## Notable Sentences from Threads\n")
        for kw in ["leakage", "winning strategy", "external data", "clarification", "rule"]:
            matches = _extract_sentences(all_text, kw, max_matches=3)
            if matches:
                lines.append(f"### Mentions of '{kw}'\n")
                for m in matches:
                    lines.append(f"> {m}\n")
                lines.append("")

        _write(self.disc_dir / "discussion_insights.md", "\n".join(lines))
        logger.info("discussion_insights.md written")


# ──────────────────────────────────────────────────────────────────────────────
# Ranking and text helpers
# ──────────────────────────────────────────────────────────────────────────────


def _rank_threads(threads: list[dict[str, Any]], max_n: int) -> list[dict[str, Any]]:
    """Score each thread by keyword relevance + vote count, return top-N."""

    def score(t: dict[str, Any]) -> float:
        text = (t.get("title", "") + " " + t.get("content", "")[:200]).lower()
        kw_score = sum(1 for kw in PRIORITY_KEYWORDS if kw in text)
        try:
            vote_score = int(t.get("votes") or 0) / 100
        except (ValueError, TypeError):
            vote_score = 0.0
        return kw_score + vote_score

    ranked = sorted(threads, key=score, reverse=True)
    return ranked[:max_n]


def _matching_keywords(text: str) -> list[str]:
    tl = text.lower()
    return [kw for kw in PRIORITY_KEYWORDS if kw in tl]


def _extract_sentences(text: str, keyword: str, max_matches: int = 3) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matches = [
        s.strip()
        for s in sentences
        if keyword.lower() in s.lower() and 20 < len(s) < 400
    ]
    return matches[:max_matches]


def _format_thread(idx: int, meta: dict[str, Any]) -> str:
    title = meta.get("title", "?")
    href = meta.get("href", "")
    votes = meta.get("votes", "?")
    comments = meta.get("comments", "?")
    content = meta.get("content", "_No content collected._")
    return (
        f"# Thread {idx:03d}: {title}\n\n"
        f"- **URL:** {href}\n"
        f"- **Votes:** {votes}  |  **Comments:** {comments}\n\n"
        f"---\n\n"
        f"{content}\n"
    )


async def _scroll(page) -> None:
    """Scroll to the bottom slowly to trigger lazy-loaded content."""
    try:
        h: int = await page.evaluate("document.body.scrollHeight")
        for pos in range(0, min(h, 8_000), 600):
            await page.evaluate(f"window.scrollTo(0, {pos})")
            await page.wait_for_timeout(120)
        await page.evaluate("window.scrollTo(0, 0)")
    except Exception:
        pass


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
