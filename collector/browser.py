"""
Playwright-based browser collector.

Public API used by main.py:
  bc = BrowserCollector(slug, out, headless, skip_tabs, collect_screenshots)
  context, page = await bc.setup_context_and_login(browser)
  browser_data   = await bc.collect_tabs(page)

Login strategy (priority order):
  1. Saved state ~/.kaggle_collector/browser_state.json
  2. KAGGLE_USERNAME + KAGGLE_PASSWORD from .env
  3. Manual login in headed mode (user presses Enter when done)
  4. No credentials + headless → clear error, returns (None, None)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from .extractor import clean_page_text
from .page_extractor import (
    extract_tab, build_page_markdown,
    is_crash_or_garbage, _CRASH_PHRASES,
)
from .utils import polite_delay

logger = logging.getLogger("kaggle_collector")

KAGGLE_BASE = "https://www.kaggle.com"
BROWSER_STATE_PATH = Path.home() / ".kaggle_collector" / "browser_state.json"
CHROMIUM_PROFILE_DIR = Path.home() / ".kaggle_collector" / "chromium_profile"

# All competition tabs: (short_name, url_suffix)
COMPETITION_TABS: list[tuple[str, str]] = [
    ("overview", ""),
    ("data", "/data"),
    ("evaluation", "/evaluation"),
    ("rules", "/rules"),
    ("leaderboard", "/leaderboard"),
    ("discussion", "/discussion"),
    ("code", "/code"),
]

_ACCESS_WALL_PHRASES = [
    "you must accept this competition",
    "accept the rules",
    "complete the following to participate",
    "must complete the following",
    "i understand and accept",
]

_EXTRACT_TEXT_JS = """
() => {
    const clone = document.body.cloneNode(true);
    clone.querySelectorAll(
        'script,style,nav,footer,header,[role="navigation"],' +
        '[role="banner"],[aria-hidden="true"],.site-header,.site-footer'
    ).forEach(el => el.remove());
    return clone.innerText;
}
"""

# All tabs now use the smart-extraction pipeline (retry + crash detection + quality scoring)
_SMART_EXTRACT_TABS: frozenset[str] = frozenset(
    {"overview", "evaluation", "rules", "data", "leaderboard", "discussion", "code"}
)


class BrowserCollector:
    def __init__(
        self,
        slug: str,
        output_dir: Path,
        headless: bool = True,
        skip_tabs: Optional[set[str]] = None,
        collect_screenshots: bool = True,
        refresh_browser_state: bool = False,
        interactive_browser: bool = False,
        # Deprecated alias — kept for backward compatibility
        pause_on_browser_warning: bool = False,
    ) -> None:
        self.slug = slug
        self.output_dir = output_dir
        self.headless = headless
        self.skip_tabs = skip_tabs or set()
        self.collect_screenshots = collect_screenshots
        self.refresh_browser_state = refresh_browser_state
        # interactive_browser only meaningful in headed mode
        self.interactive_browser = (interactive_browser or pause_on_browser_warning) and not headless
        self.pages_dir = output_dir / "pages"
        self.screenshots_dir = output_dir / "screenshots"
        self.debug_dir = output_dir / "pages" / "_debug"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._console_errors: list[dict[str, Any]] = []
        self._network_failures: list[dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Setup & login (called from main.py which owns the browser lifecycle)
    # ──────────────────────────────────────────────────────────────────────────

    async def setup_context_and_login(self, browser) -> tuple:
        """
        Create a browser context (loading saved state when available),
        verify login, and return (context, page).
        Returns (None, None) if login cannot be established.
        """
        context = await self._build_context(browser)
        page = await context.new_page()
        self._attach_diagnostics(page)

        if not await self._is_logged_in(page):
            logger.info("Not logged in. Attempting login …")
            ok = await self._perform_login(page, context)
            if not ok:
                await page.close()
                return None, None
            await _save_state(context)

        logger.info("Kaggle login confirmed")

        # Accept cookie banner once after login so it doesn't interfere with tabs
        await self._handle_cookie_banner(page, reload_after=True)

        return context, page

    async def setup_persistent_context(self, context) -> tuple:
        """
        Use an already-created persistent Chromium context.
        Persistent contexts preserve Kaggle login, consent banners, local storage,
        and cookies in CHROMIUM_PROFILE_DIR across runs.
        """
        page = context.pages[0] if context.pages else await context.new_page()
        self._attach_diagnostics(page)

        if not await self._is_logged_in(page):
            logger.warning(
                "Persistent browser profile is not logged in. "
                "Continuing with the persistent profile; run `python main.py browser-login` "
                "if tabs show LOGIN_REQUIRED."
            )

        logger.info("Persistent Chromium profile ready")
        await self._handle_cookie_banner(page, reload_after=True)
        return context, page

    # ──────────────────────────────────────────────────────────────────────────
    # Tab collection
    # ──────────────────────────────────────────────────────────────────────────

    async def collect_tabs(self, page) -> dict[str, Any]:
        """Visit all COMPETITION_TABS (minus skip_tabs), return results dict."""
        self._attach_diagnostics(page)
        results: dict[str, Any] = {}
        tabs = [(name, suffix) for name, suffix in COMPETITION_TABS if name not in self.skip_tabs]

        for tab_name, tab_suffix in tabs:
            url = f"{KAGGLE_BASE}/competitions/{self.slug}{tab_suffix}"
            logger.info(f"  → {tab_name:12s}  {url}")
            try:
                tab_result = await self._collect_tab(page, tab_name, url)
                results[tab_name] = tab_result
            except Exception as exc:
                logger.error(f"Error on tab '{tab_name}': {exc}")
                results[tab_name] = {"url": url, "error": str(exc)}
            polite_delay(2.5)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Single-tab collection
    # ──────────────────────────────────────────────────────────────────────────

    async def _collect_tab(self, page, tab_name: str, url: str) -> dict[str, Any]:
        result: dict[str, Any] = {"url": url}
        md_path = self.pages_dir / f"{tab_name}.md"

        # Build the pause callback if needed
        pause_fn = self._make_pause_fn() if self.interactive_browser else None

        # All tabs now go through the smart extraction pipeline
        # (retries, crash detection, Kaggle-ready wait, quality scoring).
        # We pass the cookie-banner handler so the extractor can call it before
        # each attempt (in case a new banner appears on individual tab pages).
        cr = await extract_tab(
            page, tab_name, url,
            max_attempts=3,
            pause_fn=pause_fn,
            cookie_handler=self._handle_cookie_banner,
        )

        md_content = build_page_markdown(tab_name, self.slug, url, cr)
        md_path.write_text(md_content, encoding="utf-8")

        result["markdown_path"] = str(md_path)
        result["content_length"] = len(cr.text or "")
        result["strategy"] = cr.strategy
        result["quality"] = cr.quality
        result["requires_acceptance"] = cr.requires_acceptance
        result["requires_login"] = cr.requires_login

        if cr.warning:
            result["warning"] = cr.warning

        if not cr.ok:
            result["error"] = cr.warning or "Content extraction failed"
            logger.warning(f"  ⚠️  {tab_name}: {cr.warning or 'extraction failed'}")
            await self._save_debug_page(page, tab_name)
        elif cr.quality == "acceptable":
            char_count = len(cr.text or "")
            logger.warning(
                f"  ⚠️  {tab_name}: accepted with warning "
                f"({char_count:,} chars, strategy={cr.strategy}) — {cr.warning or ''}"
            )
        else:
            logger.info(
                f"  Saved: pages/{tab_name}.md  "
                f"({len(cr.text or ''):,} chars, strategy={cr.strategy})"
            )

        # ── Screenshot (always attempted) ─────────────────────────────────────
        if self.collect_screenshots:
            ss_path = self.screenshots_dir / f"{tab_name}.png"
            try:
                await page.screenshot(path=str(ss_path), full_page=True)
                result["screenshot"] = str(ss_path)
            except Exception as exc:
                logger.warning(f"  Screenshot failed for '{tab_name}': {exc}")

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Cookie banner handler
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_cookie_banner(self, page, reload_after: bool = False) -> bool:
        """
        Detect and dismiss Kaggle / Google consent cookie banner.
        Returns True if a banner was found and dismissed.
        When reload_after is True, reloads the page after dismissal and waits.
        """
        _BANNER_SELECTORS = [
            # Kaggle's own consent dialog (various casing)
            "button:has-text('OK, Got it.')",
            "button:has-text('OK, got it.')",
            "button:has-text('Accept cookies')",
            "button:has-text('Accept all cookies')",
            "button:has-text('I Accept')",
            "button:has-text('Accept')",
            "button:has-text('Accept all')",
            # Google's consent framework (shows on EU IPs)
            "button[aria-label='Accept all']",
            "button[aria-label='Agree to the use of cookies and other data for the purposes described']",
            "[aria-label='Accept all cookies']",
            # Generic modal/dialog confirm buttons
            "[role='dialog'] button:has-text('OK')",
            "[role='dialog'] button:has-text('Accept')",
            # Fallback class-based patterns
            "[class*='cookie'] button",
            "[class*='consent'] button",
            "[class*='CookieBanner'] button",
            "[class*='gdpr'] button",
        ]
        dismissed = False
        for selector in _BANNER_SELECTORS:
            try:
                visible = await page.is_visible(selector, timeout=1_200)
                if visible:
                    await page.click(selector, timeout=3_000)
                    logger.info(f"  Cookie banner dismissed ({selector!r})")
                    dismissed = True
                    await page.wait_for_timeout(800)
                    break
            except Exception:
                continue

        # JavaScript fallback: find any button containing acceptance text
        if not dismissed:
            try:
                clicked = await page.evaluate("""
                () => {
                    const texts = ['OK, Got it', 'Accept all', 'Accept cookies', 'I Accept', 'Accept'];
                    for (const t of texts) {
                        const btns = Array.from(document.querySelectorAll('button, [role="button"]'));
                        const btn = btns.find(b => b.innerText && b.innerText.trim().startsWith(t));
                        if (btn && btn.offsetParent !== null) {
                            btn.click();
                            return btn.innerText.trim();
                        }
                    }
                    return null;
                }
                """)
                if clicked:
                    logger.info(f"  Cookie banner dismissed via JS fallback (text: {clicked!r})")
                    dismissed = True
                    await page.wait_for_timeout(800)
            except Exception as exc:
                logger.debug(f"  JS cookie fallback failed: {exc}")

        if dismissed and reload_after:
            try:
                await page.reload(wait_until="domcontentloaded", timeout=30_000)
                await page.wait_for_timeout(3_000)
            except Exception as exc:
                logger.debug(f"  Reload after cookie banner failed: {exc}")

        return dismissed

    # ──────────────────────────────────────────────────────────────────────────
    # Pause / manual-intervention helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _make_pause_fn(self):
        """
        Return an async callable that pauses execution and prompts the user
        to manually fix the browser before continuing.
        Only constructed when headed + --pause-on-browser-warning are both active.
        """
        async def _pause(tab_name: str, attempt: int, reason: str, final: bool = False) -> None:
            sep = "─" * 62
            if final:
                print(f"\n{sep}")
                print(f"  ⚠️  ALL RETRIES FAILED  —  tab: {tab_name}")
                print(f"  Reason: {reason}")
            else:
                print(f"\n{sep}")
                print(f"  ⚠️  BROWSER WARNING  —  tab: {tab_name}  (attempt {attempt})")
                print(f"  Reason: {reason}")
                print(f"  The browser window is open. You can:")
                print(f"    • Refresh the page manually")
                print(f"    • Solve a Cloudflare challenge")
                print(f"    • Log in again if the session expired")

            print(f"  Kaggle page crashed. Fix it manually in the browser, then press ENTER here.")
            print(f"{sep}\n")
            wait_for_user_interaction(
                prompt="",
                unavailable_msg=(
                    "Interactive mode is unavailable in this terminal. "
                    "Re-run with browser-login first."
                ),
            )
        return _pause

    # ──────────────────────────────────────────────────────────────────────────
    # Debug helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _save_debug_page(self, page, tab_name: str) -> None:
        """Save raw text plus structured diagnostics for a failed tab."""
        try:
            raw: str = await page.evaluate("() => document.body.innerText || ''")
            snippet = (raw or "")[:1000]
            debug_file = self.debug_dir / f"{tab_name}.raw.txt"
            debug_file.write_text(snippet, encoding="utf-8", errors="replace")
            await self._save_diagnostics_json(page, tab_name, raw)
            logger.debug(f"  Debug snapshot saved: pages/_debug/{tab_name}.raw.txt")
            # Append to WARNINGS.md
            self._append_warning(
                tab_name,
                f"pages/{tab_name}.md extraction failed — raw page snippet:\n\n"
                f"```\n{snippet[:500]}\n```\n",
            )
        except Exception as exc:
            logger.debug(f"  Could not save debug page for {tab_name}: {exc}")

    def _attach_diagnostics(self, page) -> None:
        """Attach console/network listeners once per page."""
        if getattr(page, "_kcc_diagnostics_attached", False):
            return
        setattr(page, "_kcc_diagnostics_attached", True)

        def _on_console(msg) -> None:
            try:
                if msg.type == "error":
                    self._console_errors.append({
                        "type": msg.type,
                        "text": msg.text,
                        "location": msg.location,
                    })
                    self._console_errors[:] = self._console_errors[-200:]
            except Exception:
                pass

        def _on_request_failed(request) -> None:
            try:
                failure = request.failure or {}
                self._network_failures.append({
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type,
                    "failure": failure,
                })
                self._network_failures[:] = self._network_failures[-200:]
            except Exception:
                pass

        page.on("console", _on_console)
        page.on("requestfailed", _on_request_failed)

    async def _save_diagnostics_json(self, page, tab_name: str, raw_inner_text: str) -> None:
        """Write pages/_debug/<tab>_diagnostics.json with crash context."""
        try:
            title = await page.title()
        except Exception:
            title = ""
        try:
            body_html = await page.evaluate("() => document.body ? document.body.innerHTML.slice(0, 500) : ''")
        except Exception:
            body_html = ""
        data = {
            "tab": tab_name,
            "current_url": getattr(page, "url", ""),
            "title": title,
            "raw_inner_text": raw_inner_text,
            "body_inner_html_first_500": body_html,
            "console_errors": self._console_errors[-50:],
            "network_failed_requests": self._network_failures[-50:],
            "screenshot": str(self.screenshots_dir / f"{tab_name}.png"),
        }
        (self.debug_dir / f"{tab_name}_diagnostics.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _append_warning(self, tab_name: str, message: str) -> None:
        warnings_path = self.output_dir / "WARNINGS.md"
        try:
            existing = warnings_path.read_text(encoding="utf-8") if warnings_path.exists() else "# Browser Warnings\n\n"
            entry = f"## {tab_name} page extraction failed\n\n{message}\n---\n\n"
            if entry[:60] not in existing:
                warnings_path.write_text(existing + entry, encoding="utf-8")
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # Context builder
    # ──────────────────────────────────────────────────────────────────────────

    async def _build_context(self, browser):
        BROWSER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 900},
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "locale": "en-US",
        }
        if self.refresh_browser_state and BROWSER_STATE_PATH.exists():
            logger.info("--refresh-browser-state: deleting saved browser state")
            try:
                BROWSER_STATE_PATH.unlink()
            except Exception as exc:
                logger.warning(f"Could not delete browser state: {exc}")

        if BROWSER_STATE_PATH.exists():
            logger.info(f"Loading saved browser state from {BROWSER_STATE_PATH}")
            kwargs["storage_state"] = str(BROWSER_STATE_PATH)
        return await browser.new_context(**kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    # Login
    # ──────────────────────────────────────────────────────────────────────────

    async def _is_logged_in(self, page) -> bool:
        try:
            await page.goto(f"{KAGGLE_BASE}/", wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(2_500)
            # Dismiss cookie/consent banner as early as possible on the home page
            await self._handle_cookie_banner(page, reload_after=False)
            await page.wait_for_timeout(1_000)
        except Exception as exc:
            logger.warning(f"Could not reach kaggle.com: {exc}")
            return False
        sign_in = await page.query_selector(
            "a[href*='/account/login'], a[href*='/account/sign-in']"
        )
        return sign_in is None

    async def _perform_login(self, page, context) -> bool:
        username = os.getenv("KAGGLE_USERNAME", "").strip()
        password = os.getenv("KAGGLE_PASSWORD", "").strip()

        if username and password:
            logger.info("Using KAGGLE_USERNAME / KAGGLE_PASSWORD from environment")
            return await self._auto_login(page, username, password)

        if not self.headless:
            return await self._manual_login(page)

        logger.error(
            "\n" + "=" * 62 + "\n"
            "  LOGIN REQUIRED — browser collection cannot proceed.\n\n"
            "  Options:\n"
            "  1. Add to .env:  KAGGLE_USERNAME=...  KAGGLE_PASSWORD=...\n"
            "  2. Re-run with:  python main.py ... --headed\n"
            + "=" * 62
        )
        return False

    async def _auto_login(self, page, username: str, password: str) -> bool:
        try:
            await page.goto(
                f"{KAGGLE_BASE}/account/login",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            await page.wait_for_timeout(2_000)

            email_btn = await page.query_selector(
                "text=Sign in with Email, [data-testid='email-login'], button:has-text('Email')"
            )
            if email_btn:
                await email_btn.click()
                await page.wait_for_timeout(1_000)

            await page.fill("input[name='email']", username)
            await page.fill("input[name='password']", password)

            async with page.expect_navigation(timeout=15_000, wait_until="domcontentloaded"):
                await page.click("button[type='submit']")

            await page.wait_for_timeout(3_000)
            if "login" in page.url or "sign" in page.url.lower():
                two_fa = await page.query_selector("input[name='otp'], input[placeholder*='code']")
                if two_fa:
                    logger.warning("⚠️  2FA required. Disable 2FA or use --headed for manual login.")
                else:
                    logger.warning("Auto-login may have failed (still on login page).")
                return False

            logger.info("Auto-login succeeded")
            return True
        except Exception as exc:
            logger.error(f"Auto-login error: {exc}")
            return False

    async def _manual_login(self, page) -> bool:
        print(
            "\n" + "=" * 62 + "\n"
            "  MANUAL LOGIN  (headed mode)\n\n"
            "  The browser is open on the Kaggle login page.\n"
            "  Please log in, then return here and press Enter.\n"
            + "=" * 62 + "\n"
        )
        try:
            await page.goto(
                f"{KAGGLE_BASE}/account/login",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
        except Exception as exc:
            logger.error(f"Could not open login page: {exc}")
            return False

        wait_for_user_interaction(
            prompt="  [Press Enter once logged in] ",
            unavailable_msg=(
                "Interactive mode is unavailable in this terminal. "
                "Re-run `python main.py browser-login` from an interactive terminal."
            ),
        )

        sign_in = await page.query_selector(
            "a[href*='/account/login'], a[href*='/account/sign-in']"
        )
        if sign_in:
            logger.error("Manual login not confirmed (sign-in link still visible).")
            return False
        logger.info("Manual login confirmed")
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Access-wall detection
    # ──────────────────────────────────────────────────────────────────────────

    async def _detect_wall(
        self, page, tab_name: str, url: str
    ) -> Optional[dict[str, Any]]:
        try:
            text: str = await page.evaluate("() => document.body.innerText.toLowerCase()")
        except Exception:
            return None
        for phrase in _ACCESS_WALL_PHRASES:
            if phrase in text:
                msg = (
                    f"⚠️  Access wall on '{tab_name}' — "
                    "rule acceptance may be required.\n"
                    f"   Visit {url} and accept, then re-run."
                )
                logger.warning(f"  {msg}")
                return {"requires_acceptance": True, "warning": msg}
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────


async def _save_state(context) -> None:
    try:
        BROWSER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=str(BROWSER_STATE_PATH))
        logger.debug(f"Browser state saved → {BROWSER_STATE_PATH}")
    except Exception as exc:
        logger.warning(f"Could not save browser state: {exc}")


def wait_for_user_interaction(prompt: str, unavailable_msg: str) -> bool:
    """
    Wait for a keypress in the main process.

    Uses input() first. On Windows terminals where stdin is awkward, falls back
    to msvcrt.getwch(). Returns False when no interactive input is available.
    """
    attempted_input = False
    stdout_is_tty = bool(sys.stdout and sys.stdout.isatty())
    if sys.stdin and sys.stdin.isatty() and stdout_is_tty:
        attempted_input = True
        try:
            input(prompt)
            return True
        except (EOFError, OSError):
            pass

    if attempted_input and os.name == "nt" and stdout_is_tty:
        try:
            import msvcrt  # type: ignore
            print(prompt or "Press any key to continue...")
            msvcrt.getwch()
            return True
        except Exception:
            pass

    print(unavailable_msg)
    return False


async def _scroll(page) -> None:
    try:
        h: int = await page.evaluate("document.body.scrollHeight")
        for pos in range(0, min(h, 8_000), 600):
            await page.evaluate(f"window.scrollTo(0, {pos})")
            await page.wait_for_timeout(130)
        await page.evaluate("window.scrollTo(0, 0)")
    except Exception:
        pass
