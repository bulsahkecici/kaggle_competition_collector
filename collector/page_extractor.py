"""
Smart page content extractor for Kaggle competition tabs.

Extraction strategy (tried in order per tab):
  1. __NEXT_DATA__ embedded JSON  — structured, most reliable
  2. Rendered-markdown / content-area DOM selectors
  3. Cleaned full-page innerText as a last resort

Every result carries a ContentResult that describes:
  - the extracted text
  - which strategy produced it
  - quality status: "good" | "acceptable" | "failed"
  - any warning message

Content is REJECTED (quality="failed") when it:
  - is shorter than the minimum threshold for that tab
  - contains known crash / error / Cloudflare / login-wall phrases
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("kaggle_collector")

# ──────────────────────────────────────────────────────────────────────────────
# Known-bad phrase banks
# ──────────────────────────────────────────────────────────────────────────────

# Phrases whose presence means the page is broken / not the real content
_CRASH_PHRASES: list[str] = [
    "this site can't be reached",
    "err_connection_refused",
    "err_name_not_resolved",
    "err_internet_disconnected",
    "net::err_",
    "something went wrong",
    "we couldn't find that page",
    "just a moment...",           # Cloudflare spinner
    "checking your browser",      # Cloudflare
    "please enable javascript",
    "enable cookies",
    "access denied",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "an unexpected error has occurred",
    "reload the page",
    "try again later",
]

# Phrases that indicate a login wall (page is valid but gated)
_LOGIN_WALL_PHRASES: list[str] = [
    "sign in to kaggle",
    "you must be logged in",
    "please log in",
    "register to continue",
]

# Phrases that indicate a competition rule-acceptance wall
_ACCEPT_WALL_PHRASES: list[str] = [
    "you must accept this competition",
    "accept the rules",
    "complete the following to participate",
    "must complete the following",
    "i understand and accept",
]

# Minimum content length (chars) before a result is considered valid per tab
_MIN_LEN: dict[str, int] = {
    "overview":    150,
    "evaluation":  100,
    "rules":       100,
    "data":        100,
    "leaderboard": 200,
    "discussion":  200,
    "code":        200,
    "default":     100,
}

# Pages under this length get quality="acceptable" (WARNING in report) unless
# they clearly contain rich Kaggle-specific content.
_SUSPECT_SHORT_LEN: int = 1000

# Phrases that confirm real Kaggle competition content is present
# (checked when length < _SUSPECT_SHORT_LEN to decide if "good" or just "acceptable")
_USEFUL_CONTENT_PHRASES: list[str] = [
    "team", "score", "rank", "submission", "public leaderboard",
    "discussion", "thread", "notebook", "kernel", "dataset",
    "overview", "evaluation", "rules", "prize", "deadline",
    "competition", "participant", "leaderboard",
]

# Selectors that indicate the Kaggle React app has finished hydrating
_KAGGLE_READY_SELECTORS = [
    "#site-content",
    "[class*='CompetitionHeader']",
    "[class*='competitionHeader']",
    "main[role='main']",
    "[data-testid='competition-header']",
    ".competition-title",
    "h1",  # any h1 as last resort
]

# Tab → JSON keys to search for in __NEXT_DATA__ (longest match wins)
_TAB_NEXTJS_KEYS: dict[str, list[str]] = {
    "overview":   ["overview", "description", "subtitle", "body"],
    "evaluation": ["evaluation", "evaluationSummary", "evaluationDescription", "metric"],
    "rules":      ["rules", "competitionRules", "rule"],
    "data":       ["dataDescription", "data_description", "dataOverview", "dataBody"],
}

# DOM selectors tried in order; first with length > min is used
_CONTENT_SELECTORS: list[str] = [
    ".rendered-markdown",
    "[class*='OverviewDescription']",
    "[class*='CompetitionDescription']",
    "[class*='overview-description']",
    "[class*='competition-overview']",
    "[data-testid*='description']",
    "[data-testid*='overview']",
    "article",
    "main",
]

# JS to read __NEXT_DATA__
_NEXTDATA_JS = "() => { const el = document.getElementById('__NEXT_DATA__'); return el ? el.textContent : null; }"

# JS to read the largest rendered-markdown block or main content block
_DOM_CONTENT_JS = r"""
(selectors) => {
    for (const sel of selectors) {
        try {
            const el = document.querySelector(sel);
            if (el && el.innerText && el.innerText.trim().length > 80) {
                return { html: el.innerHTML, text: el.innerText };
            }
        } catch(e) {}
    }
    return null;
}
"""

# JS to get clean body text after stripping chrome noise
_INNERTEXT_JS = """
() => {
    const clone = document.body.cloneNode(true);
    clone.querySelectorAll(
        'script,style,nav,footer,header,[role="navigation"],' +
        '[role="banner"],[aria-hidden="true"],.site-header,.site-footer,' +
        '[data-testid="site-header"],[data-testid="site-footer"]'
    ).forEach(el => el.remove());
    return clone.innerText;
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContentResult:
    text: Optional[str]                  # The cleaned content (None if failed)
    strategy: str                        # Which strategy produced it
    quality: str                         # "good" | "acceptable" | "failed"
    warning: Optional[str] = None        # Human-readable issue description
    requires_login: bool = False
    requires_acceptance: bool = False
    extracted_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    @property
    def ok(self) -> bool:
        return self.quality in ("good", "acceptable")


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

async def extract_tab(
    page,
    tab_name: str,
    url: str,
    max_attempts: int = 3,
    pause_fn=None,
    cookie_handler=None,
) -> ContentResult:
    """
    Navigate to `url` and extract the content of competition tab `tab_name`.
    Retries up to `max_attempts` times; reloads the page once after each crash.

    Parameters
    ----------
    pause_fn      : async callable(tab_name, attempt, reason, final=False)
                    Invoked on failure when in headed + --pause-on-browser-warning mode.
    cookie_handler: async callable(page, reload_after=True)
                    Called before each attempt to dismiss cookie/consent banners.

    After a manual pause (pause_fn), the page content is re-checked before saving.
    If the page still appears crashed after the user's intervention, the extraction
    is still considered failed (content is NOT saved as valid).
    """
    last_result: ContentResult = ContentResult(
        None, "not_attempted", "failed", warning="No attempt made yet"
    )

    for attempt in range(1, max_attempts + 1):
        logger.debug(f"  [{tab_name}] Attempt {attempt}/{max_attempts} …")

        # Dismiss any cookie/consent banner that may have appeared
        if cookie_handler is not None:
            try:
                await cookie_handler(page, reload_after=False)
            except Exception as cb_exc:
                logger.debug(f"  [{tab_name}] cookie_handler raised: {cb_exc}")

        last_result = await _extract_tab_once(page, tab_name, url)

        if last_result.ok:
            if attempt > 1:
                logger.info(f"  [{tab_name}] Succeeded on attempt {attempt}")
            return last_result

        # Failed — decide what to do before the next attempt
        if attempt < max_attempts:
            logger.warning(
                f"  [{tab_name}] Attempt {attempt} failed "
                f"({last_result.warning or 'unknown reason'}). "
                f"Retrying …"
            )

            # Allow manual intervention in headed/paused mode
            if pause_fn is not None:
                await pause_fn(tab_name, attempt, last_result.warning or "")
                # After the user presses Enter, try reading the page immediately
                # to check whether the manual fix worked, BEFORE the next full attempt.
                try:
                    quick_text = (await page.evaluate("() => document.body.innerText")).lower()
                    still_crashed = any(p in quick_text for p in _CRASH_PHRASES)
                    if not still_crashed:
                        logger.info(
                            f"  [{tab_name}] Page recovered after manual intervention"
                        )
                        # Kick off a proper extraction on the already-loaded page
                        recover_result = await _extract_tab_once(page, tab_name, page.url)
                        if recover_result.ok:
                            return recover_result
                    else:
                        logger.warning(
                            f"  [{tab_name}] Page still crashed after ENTER — will reload"
                        )
                except Exception:
                    pass

            # Reload the page to clear crash/Cloudflare state
            try:
                logger.debug(f"  [{tab_name}] Reloading page …")
                await page.reload(wait_until="load", timeout=45_000)
                await page.wait_for_timeout(4_000)
                await _wait_for_kaggle_ready(page)
                # Dismiss banner again after reload
                if cookie_handler is not None:
                    await cookie_handler(page, reload_after=False)
            except Exception as reload_exc:
                logger.debug(f"  [{tab_name}] Reload failed: {reload_exc}")

    logger.warning(
        f"  [{tab_name}] All {max_attempts} attempt(s) failed. "
        f"Last reason: {last_result.warning}"
    )
    # Final pause opportunity before giving up
    if pause_fn is not None:
        await pause_fn(tab_name, max_attempts, last_result.warning or "", final=True)

    return last_result


async def _extract_tab_once(
    page,
    tab_name: str,
    url: str,
) -> ContentResult:
    """
    Single-attempt extraction for one tab.
    Navigate → wait → detect walls/crashes → try strategies in order.
    """
    # ── Navigate ─────────────────────────────────────────────────────────────
    try:
        # Use domcontentloaded for faster response, then wait for network idle
        await page.goto(url, wait_until="domcontentloaded", timeout=50_000)
    except Exception as exc:
        return ContentResult(
            None, "navigation", "failed",
            warning=f"Navigation failed: {exc}",
        )

    # Wait for network requests to settle (Kaggle fetches JSON via internal APIs)
    try:
        await page.wait_for_load_state("networkidle", timeout=12_000)
    except Exception:
        pass  # networkidle may never fire on live pages; that's OK

    # Extra initial buffer for React hydration
    await page.wait_for_timeout(4_000)
    await _wait_for_kaggle_ready(page)
    await page.wait_for_timeout(2_000)

    # ── Early checks ─────────────────────────────────────────────────────────
    title = await page.title()
    if "404" in title or "not found" in title.lower():
        return ContentResult(None, "404_check", "failed", warning="Page returned 404")

    # Scroll to trigger lazy-loaded content
    await _scroll_page(page)
    await page.wait_for_timeout(1_500)

    # ── Check for login / acceptance walls / crashes ──────────────────────────
    page_lower = ""
    try:
        page_lower = (await page.evaluate("() => document.body.innerText")).lower()
    except Exception:
        pass

    login_wall = any(p in page_lower for p in _LOGIN_WALL_PHRASES)
    accept_wall = any(p in page_lower for p in _ACCEPT_WALL_PHRASES)
    crash = any(p in page_lower for p in _CRASH_PHRASES)

    if crash:
        return ContentResult(
            None, "crash_check", "failed",
            warning="Page appears to be a crash / error / Cloudflare page",
        )

    # ── Strategy 1: __NEXT_DATA__ JSON ────────────────────────────────────────
    result = await _try_nextjs(page, tab_name)
    if result and _passes_min_length(result, tab_name):
        quality, warning = _score_content(result, tab_name)
        cr = ContentResult(result, "nextjs_data", quality, warning=warning)
        if login_wall:
            cr.requires_login = True
            cr.warning = (cr.warning + " | " if cr.warning else "") + "Login wall detected"
        if accept_wall:
            cr.requires_acceptance = True
            cr.warning = (cr.warning + " | " if cr.warning else "") + "Rule acceptance wall"
        return cr

    # ── Strategy 2: Known content DOM selectors ───────────────────────────────
    result = await _try_dom_selectors(page)
    if result and _passes_min_length(result, tab_name):
        quality, warning = _score_content(result, tab_name)
        cr = ContentResult(result, "dom_selector", quality, warning=warning)
        if login_wall:
            cr.requires_login = True
        if accept_wall:
            cr.requires_acceptance = True
        return cr

    # ── Strategy 3: Cleaned full-page innerText ───────────────────────────────
    result = await _try_innertext(page)
    if result and _passes_min_length(result, tab_name):
        quality, warning = _score_content(result, tab_name)
        noise_warn = "Content extracted via full-page innerText (may include noise)"
        if login_wall:
            noise_warn += " | Login wall detected — content may be login form only"
        if accept_wall:
            noise_warn += " | Rule acceptance wall detected"
        final_warn = noise_warn if not warning else f"{noise_warn} | {warning}"
        return ContentResult(result, "innertext", quality, warning=final_warn,
                             requires_login=login_wall, requires_acceptance=accept_wall)

    # ── All strategies failed ─────────────────────────────────────────────────
    reason_parts: list[str] = []
    if login_wall:
        reason_parts.append("login wall")
    if accept_wall:
        reason_parts.append("rule acceptance required")
    if not reason_parts:
        reason_parts.append("content too short or not rendered")
    warning = "Extraction failed: " + " + ".join(reason_parts)

    return ContentResult(
        None, "all_failed", "failed",
        warning=warning,
        requires_login=login_wall,
        requires_acceptance=accept_wall,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Kaggle readiness check
# ──────────────────────────────────────────────────────────────────────────────

async def _wait_for_kaggle_ready(page, timeout_ms: int = 15_000) -> None:
    """
    Wait for the Kaggle React app to finish rendering.
    Tries a list of content selectors and returns as soon as one appears.
    Falls through silently if nothing appears within the timeout.
    """
    # Give each selector an equal share of the total timeout
    per_sel_ms = max(1_500, timeout_ms // len(_KAGGLE_READY_SELECTORS))
    for sel in _KAGGLE_READY_SELECTORS:
        try:
            await page.wait_for_selector(sel, timeout=per_sel_ms)
            logger.debug(f"  Kaggle app ready (matched: {sel})")
            return
        except Exception:
            continue
    logger.debug("  _wait_for_kaggle_ready: no selector matched within timeout")


# ──────────────────────────────────────────────────────────────────────────────
# Strategy implementations
# ──────────────────────────────────────────────────────────────────────────────

async def _try_nextjs(page, tab_name: str) -> Optional[str]:
    """Try to pull content from the embedded __NEXT_DATA__ JSON blob."""
    try:
        raw: Optional[str] = await page.evaluate(_NEXTDATA_JS)
        if not raw:
            return None
        data: Any = json.loads(raw)
        return _search_nextjs_tree(data, tab_name)
    except Exception as exc:
        logger.debug(f"  __NEXT_DATA__ extraction failed: {exc}")
        return None


async def _try_dom_selectors(page) -> Optional[str]:
    """Try known content-area CSS selectors and return the first that yields text."""
    try:
        result: Optional[dict] = await page.evaluate(_DOM_CONTENT_JS, _CONTENT_SELECTORS)
        if result:
            # Prefer HTML → markdownify; fall back to innerText
            html = result.get("html", "")
            text = result.get("text", "")
            if html:
                try:
                    from .extractor import html_to_markdown
                    md = html_to_markdown(html)
                    if md and len(md) > 50:
                        return md
                except Exception:
                    pass
            return text or None
    except Exception as exc:
        logger.debug(f"  DOM selector extraction failed: {exc}")
    return None


async def _try_innertext(page) -> Optional[str]:
    """Fall back to full-page innerText with chrome-noise removed."""
    try:
        raw: str = await page.evaluate(_INNERTEXT_JS)
        return _clean_text(raw or "")
    except Exception as exc:
        logger.debug(f"  innerText extraction failed: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# __NEXT_DATA__ tree search
# ──────────────────────────────────────────────────────────────────────────────

def _search_nextjs_tree(data: Any, tab_name: str) -> Optional[str]:
    """
    Recursively search the Next.js JSON for the longest string value whose
    key matches one of the tab's known key-patterns.
    """
    keys = _TAB_NEXTJS_KEYS.get(tab_name, [tab_name])
    best: Optional[str] = None
    best_len = 0

    def _walk(obj: Any, depth: int = 0) -> None:
        nonlocal best, best_len
        if depth > 12:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                k_lower = k.lower()
                if isinstance(v, str) and len(v) > best_len:
                    for pat in keys:
                        if pat.lower() in k_lower:
                            if len(v) > 30 and len(v) > best_len:
                                best = v
                                best_len = len(v)
                            break
                _walk(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:30]:
                _walk(item, depth + 1)

    _walk(data)
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Content quality check
# ──────────────────────────────────────────────────────────────────────────────

def _passes_min_length(text: Optional[str], tab_name: str) -> bool:
    """Return True when text exceeds the minimum threshold and has no crash phrases."""
    if not text:
        return False
    stripped = text.strip()
    min_len = _MIN_LEN.get(tab_name, _MIN_LEN["default"])
    if len(stripped) < min_len:
        return False
    lower = stripped.lower()
    for phrase in _CRASH_PHRASES:
        if phrase in lower:
            return False
    return True


def _score_content(text: str, tab_name: str) -> tuple[str, Optional[str]]:
    """
    Determine quality tier and optional warning for content that passed _passes_min_length.

    Returns (quality, warning_or_None):
      "good"       — rich content (≥ _SUSPECT_SHORT_LEN chars, or clearly Kaggle-specific)
      "acceptable" — short but contains useful Kaggle phrases
    """
    stripped = text.strip()
    length = len(stripped)
    lower = stripped.lower()

    if length >= _SUSPECT_SHORT_LEN:
        return "good", None

    # Short content — check if it contains genuinely useful Kaggle phrases
    has_useful = any(phrase in lower for phrase in _USEFUL_CONTENT_PHRASES)
    if has_useful:
        return "acceptable", (
            f"Content is short ({length:,} chars) but contains recognisable Kaggle text"
        )

    return "acceptable", (
        f"Content is suspiciously short ({length:,} chars) — may be incomplete or partially rendered"
    )


# Backward-compat alias used by the test path in browser.py
def _passes_quality(text: Optional[str], tab_name: str) -> bool:
    return _passes_min_length(text, tab_name)


def is_crash_or_garbage(text: str) -> bool:
    """Public helper used by the quality validator."""
    if not text or len(text.strip()) < 50:
        return True
    lower = text.lower()
    return any(p in lower for p in _CRASH_PHRASES + _LOGIN_WALL_PHRASES)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_page_markdown(
    tab_name: str,
    slug: str,
    url: str,
    result: ContentResult,
) -> str:
    """
    Format the ContentResult as the final .md file content, including
    metadata headers and warnings.
    """
    lines: list[str] = [
        f"# {tab_name.title()} — {slug}",
        "",
        f"**Source:** {url}  ",
        f"**Extracted:** {result.extracted_at}  ",
        f"**Method:** `{result.strategy}`  ",
        f"**Quality:** `{result.quality}`  ",
    ]
    if result.warning:
        lines.append(f"**Warning:** ⚠️ {result.warning}  ")
    if result.requires_login:
        lines.append("**Login required:** Yes — content may be incomplete  ")
    if result.requires_acceptance:
        lines.append("**Rule acceptance:** Required to see full content  ")
    lines += ["", "---", ""]

    if result.ok and result.text:
        lines.append(result.text)
    else:
        lines += [
            "⚠️ **Content extraction failed.**",
            "",
            f"Reason: {result.warning or 'Unknown'}",
            "",
            "**What to do:**",
            f"1. Visit {url} in your browser",
            "2. Ensure you are logged in to Kaggle",
            "3. If a rule-acceptance wall appears, accept the rules",
            "4. Re-run the collector",
        ]

    return "\n".join(lines)


async def _scroll_page(page) -> None:
    try:
        h: int = await page.evaluate("document.body.scrollHeight")
        for pos in range(0, min(h, 8_000), 600):
            await page.evaluate(f"window.scrollTo(0, {pos})")
            await page.wait_for_timeout(120)
        await page.evaluate("window.scrollTo(0, 0)")
    except Exception:
        pass


def _clean_text(text: str) -> str:
    """Collapse whitespace and blank lines."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    return "\n".join(out).strip()
