"""
Page content extraction helpers.

Converts raw HTML or Playwright page handles into clean Markdown text.
Falls back gracefully when optional dependencies (markdownify, bs4) are absent.
"""

from __future__ import annotations

import re


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────


def html_to_markdown(html: str) -> str:
    """
    Convert an HTML string to Markdown.

    Preference order:
    1. markdownify  (rich conversion, preserves tables / headings)
    2. BeautifulSoup plain-text extraction
    3. Regex tag-stripping fallback
    """
    if not html or not html.strip():
        return ""

    # ── Option 1: markdownify ─────────────────────────────────────────────────
    try:
        import markdownify  # type: ignore

        md = markdownify.markdownify(
            html,
            heading_style="ATX",
            strip=["script", "style", "nav", "footer", "header"],
            newline_style="backslash",
        )
        return _clean_text(md)
    except ImportError:
        pass

    # ── Option 2: BeautifulSoup plain text ───────────────────────────────────
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return _clean_text(soup.get_text(separator="\n"))
    except ImportError:
        pass

    # ── Option 3: naive regex strip ──────────────────────────────────────────
    text = re.sub(r"<[^>]+>", "", html)
    return _clean_text(text)


def clean_page_text(raw: str) -> str:
    """
    Post-process the innerText extracted directly from the browser DOM.

    Collapses excessive blank lines and strips leading/trailing whitespace
    on each line, preserving intentional paragraph breaks.
    """
    return _clean_text(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Strip per-line whitespace and collapse runs of blank lines to one."""
    lines = [line.rstrip() for line in text.splitlines()]
    result: list[str] = []
    blank_streak = 0
    for line in lines:
        if line == "":
            blank_streak += 1
            if blank_streak == 1:
                result.append("")
        else:
            blank_streak = 0
            result.append(line)
    return "\n".join(result).strip()
