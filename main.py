"""
kaggle_competition_collector — main entry point.

Usage
-----
python main.py --competition playground-series-s6e5
python main.py --competition titanic --headed --max-notebooks 20 --max-discussions 30
python main.py --config config.yaml
python main.py --config config.yaml --competition titanic   # CLI overrides config
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Load .env before anything else so env-vars are available to every import
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("kaggle_collector")


# ──────────────────────────────────────────────────────────────────────────────
# CLI definition
# ──────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kaggle_competition_collector",
        description="Build a complete AI-ready research package from a Kaggle competition page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --competition playground-series-s6e5
  python main.py --competition https://www.kaggle.com/competitions/titanic
  python main.py --competition titanic --headed --max-notebooks 20 --max-discussions 30
  python main.py --config config.yaml
  python main.py --config config.yaml --competition titanic --no-browser
""",
    )

    p.add_argument("--competition", "-c", metavar="SLUG_OR_URL",
                   help="Competition slug or full Kaggle URL.")
    p.add_argument("--config", metavar="FILE",
                   help="Path to config.yaml (CLI flags override values inside).")
    p.add_argument("--output", "-o", metavar="DIR", default=None,
                   help="Root output directory (default: ./competition_archive).")

    # Collection toggles
    p.add_argument("--no-browser", action="store_true",
                   help="Skip all browser automation (API-only run).")
    p.add_argument("--no-download", action="store_true",
                   help="Skip competition data file download.")
    p.add_argument("--no-notebooks", action="store_true",
                   help="Skip notebook collection.")
    p.add_argument("--no-discussions", action="store_true",
                   help="Skip deep discussion collection.")
    p.add_argument("--no-profile", action="store_true",
                   help="Skip data profiling.")
    p.add_argument("--no-screenshots", action="store_true",
                   help="Skip full-page screenshots.")

    # Limits
    p.add_argument("--max-notebooks", type=int, metavar="N",
                   help="Maximum number of notebooks to collect (default: 15).")
    p.add_argument("--max-discussions", type=int, metavar="N",
                   help="Maximum number of discussion threads (default: 25).")

    # Browser mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--headed", action="store_true",
                      help="Show the browser window (required for first-time manual login).")
    mode.add_argument("--headless", action="store_true",
                      help="Force headless browser (default).")

    # Misc
    p.add_argument("--skip-tabs", nargs="*", metavar="TAB", default=None,
                   help="Space-separated tab names to skip: overview data evaluation rules leaderboard discussion code.")
    p.add_argument("--overwrite-cache", action="store_true",
                   help="Ignore cached files and re-download everything.")
    p.add_argument("--refresh-browser-state", action="store_true",
                   help="Delete cached browser session and force a fresh login.")
    p.add_argument("--pause-on-browser-warning", action="store_true",
                   help="Deprecated alias for --interactive-browser. Kept for backward compatibility.")
    p.add_argument("--interactive-browser", action="store_true",
                   help=(
                       "When a browser tab crashes/Cloudflare blocks, pause and wait for "
                       "manual ENTER in the terminal. Requires --headed AND an interactive "
                       "terminal (not a background process). If stdin is not a TTY, prints "
                       "a clear 'Interactive pause unavailable' message and continues."
                   ))
    p.add_argument("--pause-before-tabs", action="store_true",
                   help=(
                       "In headed mode, open the competition overview page and pause before "
                       "collecting tabs so you can accept cookies, log in, or solve prompts."
                   ))
    p.add_argument("--clean-latest", action="store_true",
                   help=(
                       "Delete the entire latest/ directory before starting, "
                       "then rebuild from scratch. Downloaded data files are "
                       "preserved unless --no-download is also given."
                   ))
    p.add_argument("--keep-latest", action="store_true",
                   help=(
                       "Do NOT clean run-generated directories (pages/, "
                       "screenshots/, discussions/, data_profile/, leaderboard/) "
                       "at the start of the run. Keeps stale files. "
                       "Use only if you want to diff against a previous run."
                   ))

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Main async orchestrator
# ──────────────────────────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> int:  # noqa: C901
    from collector.config_loader import load_config
    from collector.utils import parse_competition_input, setup_logging
    from collector.archiver import Archiver, collect_pkg_versions
    from collector.kaggle_api import KaggleAPICollector
    from collector.browser import BrowserCollector, COMPETITION_TABS
    from collector.notebooks import NotebookCollector
    from collector.discussions import DiscussionCollector
    from collector.data_profiler import DataProfiler
    from collector.metric_analyzer import MetricAnalyzer
    from collector.rules_analyzer import RulesAnalyzer
    from collector.leaderboard_collector import LeaderboardCollector
    from collector.report_generator import ReportGenerator
    from collector.quality_validator import QualityValidator

    # ── Build config ──────────────────────────────────────────────────────────
    cli_overrides = {
        "competition_slug": args.competition,
        "output_dir": args.output,
        "collect_notebooks": False if args.no_notebooks else None,
        "collect_discussions": False if args.no_discussions else None,
        "profile_data": False if args.no_profile else None,
        "collect_screenshots": False if args.no_screenshots else None,
        "headless_browser": False if args.headed else (True if args.headless else None),
        "max_notebooks": args.max_notebooks,
        "max_discussion_threads": args.max_discussions,
        "overwrite_cache": args.overwrite_cache if args.overwrite_cache else None,
        "skip_tabs": args.skip_tabs,
    }
    # Remove None values so they don't overwrite YAML defaults
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    cfg = load_config(args.config, cli_overrides)

    if not cfg.competition_slug:
        print("Error: --competition is required (or set competition_slug in config.yaml).",
              file=sys.stderr)
        return 1

    slug = parse_competition_input(cfg.competition_slug)
    base_dir = Path(cfg.output_dir).resolve()

    # ── Archiver ──────────────────────────────────────────────────────────────
    archiver = Archiver(base_dir, slug)
    out = archiver.latest_dir  # All collectors write here

    # ── Clean stale run artifacts BEFORE opening the log file (avoids Windows file-lock) ─
    if not getattr(args, "keep_latest", False):
        archiver.clean_run_dirs(
            clean_latest_completely=getattr(args, "clean_latest", False),
            keep_data=(not args.no_download and not args.overwrite_cache),
            keep_notebooks=(not args.no_notebooks and not args.overwrite_cache),
        )
    else:
        archiver.write_run_id_marker()

    # Ensure latest/ exists (clean_run_dirs may have recreated it but always confirm)
    out.mkdir(parents=True, exist_ok=True)

    # Setup logging (opens collection_log.txt — must happen AFTER clean_run_dirs)
    setup_logging(out)

    headless = cfg.headless_browser and not getattr(args, "headed", False)

    # --interactive-browser: pause on crash (new flag; --pause-on-browser-warning is alias)
    interactive_browser = (
        getattr(args, "interactive_browser", False)
        or getattr(args, "pause_on_browser_warning", False)
    ) and not headless

    archiver.log(
        f"Run started. slug={slug} run_id={archiver.run_id} "
        f"max_notebooks={cfg.max_notebooks} max_discussions={cfg.max_discussion_threads} "
        f"headless={headless} interactive_browser={interactive_browser}"
    )

    skip_tabs = set(cfg.skip_tabs or [])
    download_data = not args.no_download

    # Counters for the final summary
    counters: dict[str, int] = {
        "pages_extracted": 0,
        "screenshots_saved": 0,
        "notebooks_collected": 0,
        "discussion_threads_collected": 0,
        "data_files_downloaded": 0,
    }

    # ── Phase 1: Kaggle API ───────────────────────────────────────────────────
    _phase("1/9", "Kaggle API — metadata + data download")
    api_data: dict = {}
    try:
        api_collector = KaggleAPICollector(slug, out)
        api_data = await api_collector.collect(download_data=download_data)
        data_dir = out / "data"
        if data_dir.exists():
            counters["data_files_downloaded"] = sum(1 for f in data_dir.rglob("*") if f.is_file())
        archiver.log(f"API collection done. files={counters['data_files_downloaded']}")
    except Exception as exc:
        logger.error(f"API collection error: {exc}")
        archiver.error(f"Phase 1 (API): {exc}")

    # ── Phase 2: Browser — page text + screenshots ───────────────────────────
    browser_data: dict = {}
    live_page = None          # shared Playwright page for subsequent phases
    live_context = None
    live_browser = None
    live_pw = None

    # Check playwright availability before attempting browser phase
    _playwright_ok = _check_playwright()

    skip_browser = args.no_browser or not _playwright_ok
    if not _playwright_ok and not args.no_browser:
        _remediation = (
            "Playwright is not installed. Browser collection will be skipped.\n"
            "To enable browser collection, install it:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium\n"
            "Then re-run without --no-browser."
        )
        logger.warning(f"  ⚠️  {_remediation}")
        archiver.error(f"Phase 2 (browser): {_remediation}")

    if not skip_browser:
        _phase("2/8", f"Browser — competition tabs ({'headed' if not headless else 'headless'})")
        try:
            from playwright.async_api import async_playwright  # type: ignore
            from collector.browser import CHROMIUM_PROFILE_DIR

            live_pw = await async_playwright().start()
            bc = BrowserCollector(
                slug, out,
                headless=headless,
                skip_tabs=skip_tabs,
                collect_screenshots=cfg.collect_screenshots,
                refresh_browser_state=getattr(args, "refresh_browser_state", False),
                interactive_browser=interactive_browser,
            )

            if not headless:
                CHROMIUM_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using persistent Chromium profile: {CHROMIUM_PROFILE_DIR}")
                try:
                    live_context = await _launch_persistent_chromium(live_pw, CHROMIUM_PROFILE_DIR)
                except Exception as first_exc:
                    logger.warning(
                        "Persistent Chromium profile launch failed once; "
                        "terminating stale Playwright Chromium processes and retrying. "
                        f"Reason: {first_exc}"
                    )
                    _terminate_stale_playwright_chromium()
                    await asyncio.sleep(1)
                    live_context = await _launch_persistent_chromium(live_pw, CHROMIUM_PROFILE_DIR)
                live_context, live_page = await bc.setup_persistent_context(live_context)
            else:
                live_browser = await live_pw.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                live_context, live_page = await bc.setup_context_and_login(live_browser)

            if live_page is not None:
                if getattr(args, "pause_before_tabs", False) and not headless:
                    await _pause_before_tabs(live_page, slug)
                browser_data = await bc.collect_tabs(live_page)
                counters["pages_extracted"] = sum(
                    1 for v in browser_data.values() if isinstance(v, dict) and v.get("markdown_path")
                )
                counters["screenshots_saved"] = sum(
                    1 for v in browser_data.values() if isinstance(v, dict) and v.get("screenshot")
                )
                archiver.log(f"Browser pages done. pages={counters['pages_extracted']}")
            else:
                archiver.warning("Browser login failed; skipping tab collection")
        except Exception as exc:
            logger.error(f"Browser collection error: {exc}")
            archiver.error(f"Phase 2 (browser): {exc}")
    else:
        _phase("2/8", "Browser skipped (--no-browser or playwright not installed)")

    # ── Phase 3: Notebooks ────────────────────────────────────────────────────
    notebooks: list = []
    if cfg.collect_notebooks and not args.no_browser or (cfg.collect_notebooks and not args.no_notebooks):
        if not args.no_notebooks:
            _phase("3/8", "Notebooks — public code collection")
            try:
                nb_collector = NotebookCollector(
                    slug, out,
                    max_notebooks=cfg.max_notebooks,
                    delay=cfg.polite_delay_seconds,
                )
                notebooks = await nb_collector.collect(page=live_page)
                counters["notebooks_collected"] = len(notebooks)
                archiver.log(f"Notebooks done. count={len(notebooks)}")
            except Exception as exc:
                logger.error(f"Notebook collection error: {exc}")
                archiver.error(f"Phase 3 (notebooks): {exc}")

    # ── Phase 4: Discussions ──────────────────────────────────────────────────
    discussions: list = []
    if cfg.collect_discussions and not args.no_discussions and live_page is not None:
        _phase("4/8", "Discussions — deep thread collection")
        try:
            disc_collector = DiscussionCollector(
                slug, out,
                max_threads=cfg.max_discussion_threads,
                delay=cfg.polite_delay_seconds,
            )
            discussions = await disc_collector.collect(live_page)
            counters["discussion_threads_collected"] = len(discussions)
            archiver.log(f"Discussions done. threads={len(discussions)}")
        except Exception as exc:
            logger.error(f"Discussion collection error: {exc}")
            archiver.error(f"Phase 4 (discussions): {exc}")
    elif cfg.collect_discussions and live_page is None:
        archiver.warning("Discussion collection skipped (browser not available)")

    # ── Phase 5: Leaderboard ──────────────────────────────────────────────────
    _phase("5/8", "Leaderboard + submission history")
    lb_result: dict = {}
    try:
        lb_collector = LeaderboardCollector(slug, out)
        # Pass live_page so SubmissionScoreFetcher can use browser fallback for scores
        lb_result = await lb_collector.collect(page=live_page)
        archiver.log("Leaderboard collection done")
    except Exception as exc:
        logger.error(f"Leaderboard error: {exc}")
        archiver.error(f"Phase 5 (leaderboard): {exc}")

    # ── Phase 6: Data profiling ───────────────────────────────────────────────
    data_profiles: dict = {}
    if cfg.profile_data and not args.no_profile:
        _phase("6/8", "Data profiling")
        try:
            profiler = DataProfiler(out)
            data_profiles = profiler.profile_all()
            archiver.log(f"Data profiling done. files={len(data_profiles)}")
        except Exception as exc:
            logger.error(f"Data profiling error: {exc}")
            archiver.error(f"Phase 6 (data profile): {exc}")

    # ── Phase 7: Metric + Rules analysis ─────────────────────────────────────
    _phase("7/8", "Metric + Rules analysis")
    metric_result: dict = {}
    rules_result: dict = {}
    try:
        eval_text = _read_page(browser_data, "evaluation")
        overview_text = _read_page(browser_data, "overview")
        rules_text = _read_page(browser_data, "rules")
        meta = api_data.get("metadata", {})

        # If this run's API failed (meta is empty), load from saved competition_metadata.json
        # so MetricAnalyzer can fall back to a previously-collected evaluationMetric.
        if not meta:
            _saved_meta_path = out / "competition_metadata.json"
            if _saved_meta_path.exists():
                try:
                    import json as _json
                    meta = _json.loads(_saved_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

        # evaluationMetric may come from old SDK (camelCase) or new SDK via serialized dict
        api_metric_raw = meta.get("evaluationMetric") or meta.get("evaluation_metric")
        metric_result = MetricAnalyzer(out).analyze(
            api_metric=api_metric_raw,
            eval_page_text=eval_text,
            overview_page_text=overview_text,
            existing_metadata=meta,
            competition_slug=slug,
        )
        rules_result = RulesAnalyzer(out).analyze(rules_text, meta)
        archiver.log("Metric and rules analysis done")
    except Exception as exc:
        logger.error(f"Analysis error: {exc}")
        archiver.error(f"Phase 7 (analysis): {exc}")

    # ── Phase 8: Final reports ────────────────────────────────────────────────
    _phase("8/9", "Final reports")
    summary_path = out / "SUMMARY_REPORT.md"
    handoff_path = out / "AI_HANDOFF_INSTRUCTIONS.md"
    try:
        rg = ReportGenerator(
            slug=slug,
            output_dir=out,
            api_data=api_data,
            browser_data=browser_data,
            notebooks=notebooks,
            discussions=discussions,
            data_profiles=data_profiles,
            metric_result=metric_result,
            rules_result=rules_result,
            lb_result=lb_result,
        )
        summary_path, handoff_path = rg.generate_all()
        archiver.log("Final reports written")
    except Exception as exc:
        logger.error(f"Report generation error: {exc}")
        archiver.error(f"Phase 8 (reports): {exc}")

    # ── Phase 9: Quality validation ───────────────────────────────────────────
    _phase("9/9", "Collection quality validation")
    quality_results: list = []
    try:
        validator = QualityValidator(
            output_dir=out,
            browser_data=browser_data,
            metric_result=metric_result,
            lb_result=lb_result,
            data_profiles=data_profiles,
            archiver=archiver,
        )
        quality_results = validator.validate()
        fail_count = sum(1 for r in quality_results if r.status == "FAILED")
        warn_count = sum(1 for r in quality_results if r.status == "WARNING")
        archiver.log(
            f"Quality validation done. "
            f"FAILED={fail_count} WARNINGS={warn_count}"
        )
        if fail_count:
            logger.warning(
                f"  ⚠️  Quality check: {fail_count} FAILED item(s). "
                "See collection_quality_report.md"
            )
    except Exception as exc:
        logger.error(f"Quality validation error: {exc}")
        archiver.error(f"Phase 9 (quality): {exc}")

    # ── Close browser ─────────────────────────────────────────────────────────
    try:
        if live_context:
            try:
                await live_context.storage_state(
                    path=str(Path.home() / ".kaggle_collector" / "browser_state.json")
                )
            except Exception:
                pass
            await live_context.close()
        if live_browser:
            await live_browser.close()
        if live_pw:
            await live_pw.stop()
    except Exception:
        pass

    # ── Archive + finalize ────────────────────────────────────────────────────
    run_config = {
        "max_notebooks": cfg.max_notebooks,
        "max_discussion_threads": cfg.max_discussion_threads,
        "headless": headless,
        "interactive_browser": interactive_browser,
        "pause_before_tabs": getattr(args, "pause_before_tabs", False),
        "browser_profile": str(Path.home() / ".kaggle_collector" / "chromium_profile"),
        "collect_notebooks": cfg.collect_notebooks,
        "collect_discussions": cfg.collect_discussions,
        "download_data": download_data,
        "profile_data": cfg.profile_data,
        "collect_screenshots": cfg.collect_screenshots,
        "overwrite_cache": cfg.overwrite_cache,
        "clean_latest": getattr(args, "clean_latest", False),
        "keep_latest": getattr(args, "keep_latest", False),
        "skip_tabs": list(skip_tabs),
    }
    archive_path = archiver.finalize(counters, collect_pkg_versions(), run_config)

    # ── Final success message ─────────────────────────────────────────────────
    _print_summary(
        out, archive_path, counters,
        summary_path, handoff_path,
        quality_results=quality_results,
    )

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _check_playwright() -> bool:
    """Return True if playwright is importable and chromium is installed."""
    try:
        import importlib
        importlib.import_module("playwright")
        return True
    except ImportError:
        return False


def _banner(slug: str, out: Path) -> None:
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  kaggle_competition_collector")
    print(f"  Competition : {slug}")
    print(f"  Output      : {out}")
    print(f"{sep}\n")


def _phase(label: str, desc: str) -> None:
    logger.info(f"\n── Phase {label}: {desc}")


async def _launch_persistent_chromium(playwright, profile_dir: Path):
    """Launch a headed persistent Chromium profile for Kaggle browser work."""
    return await playwright.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=False,
        viewport={"width": 1280, "height": 900},
        locale="en-US",
        args=["--disable-blink-features=AutomationControlled"],
    )


def _terminate_stale_playwright_chromium() -> None:
    """
    Best-effort cleanup for orphaned Playwright Chromium processes on Windows.
    Does not touch the user's regular Google Chrome install.
    """
    if sys.platform != "win32":
        return
    try:
        import subprocess
        cmd = (
            "Get-Process chrome -ErrorAction SilentlyContinue | "
            "Where-Object { $_.Path -like '*ms-playwright*chromium*chrome.exe' } | "
            "Stop-Process -Force"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        logger.debug(f"Could not terminate stale Playwright Chromium processes: {exc}")


async def _pause_before_tabs(page, slug: str) -> None:
    """Open overview and pause for manual cookie/challenge handling."""
    from collector.browser import wait_for_user_interaction

    url = f"https://www.kaggle.com/competitions/{slug}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(2_000)
    except Exception as exc:
        logger.warning(f"Could not open overview before tab pause: {exc}")

    wait_for_user_interaction(
        prompt=(
            "Please make sure Kaggle page is fully loaded. "
            "Accept cookies or solve any prompt. Then press ENTER."
        ),
        unavailable_msg=(
            "Interactive mode is unavailable in this terminal. "
            "Re-run with browser-login first."
        ),
    )


def _read_page(browser_data: dict, tab_name: str) -> str:
    """Read saved markdown for a tab if it exists."""
    tab = browser_data.get(tab_name, {})
    if not isinstance(tab, dict):
        return ""
    md_path = tab.get("markdown_path", "")
    if md_path:
        p = Path(md_path)
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    return ""


def _print_summary(
    out: Path,
    archive_path: Path,
    counters: dict,
    summary_path: Path,
    handoff_path: Path,
    quality_results: list = [],
) -> None:
    ok = sum(1 for r in quality_results if r.status == "OK")
    warn = sum(1 for r in quality_results if r.status == "WARNING")
    fail = sum(1 for r in quality_results if r.status == "FAILED")

    sep = "═" * 62
    print(f"\n{sep}")
    if fail == 0:
        print("  ✅  Collection complete!")
    else:
        print(f"  ⚠️   Collection complete with {fail} quality FAILURE(s)")
    print(f"{sep}")
    print(f"  Output folder     : {out}")
    print(f"  Archive copy      : {archive_path}")
    print(f"  Data files        : {counters['data_files_downloaded']}")
    print(f"  Pages extracted   : {counters['pages_extracted']}")
    print(f"  Screenshots saved : {counters['screenshots_saved']}")
    print(f"  Notebooks         : {counters['notebooks_collected']}")
    print(f"  Discussions       : {counters['discussion_threads_collected']}")
    print(f"  SUMMARY_REPORT    : {summary_path}")
    print(f"  AI_HANDOFF        : {handoff_path}")
    if quality_results:
        print(f"  Quality           : ✅ {ok} OK  |  ⚠️  {warn} WARN  |  ❌ {fail} FAIL")
        print(f"  Quality report    : {out / 'collection_quality_report.md'}")
    if fail:
        print(f"\n  Failed items:")
        for r in quality_results:
            if r.status == "FAILED":
                print(f"    ❌ {r.item}: {r.details}")
    print(f"{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Doctor command
# ──────────────────────────────────────────────────────────────────────────────

_DOCTOR_PACKAGES = [
    "kaggle", "playwright", "pandas", "pyarrow",
    "markdownify", "beautifulsoup4", "pyyaml",
    "openpyxl", "tabulate", "python-dotenv",
]


def _doctor_check(label: str, ok: bool, detail: str = "") -> bool:
    symbol = "✅" if ok else "❌"
    suffix = f"  — {detail}" if detail else ""
    print(f"  {symbol}  {label}{suffix}")
    return ok


async def _doctor_browser_check() -> tuple[bool, bool]:
    """Try to open Kaggle in a headless browser. Returns (launched, loaded)."""
    try:
        from playwright.async_api import async_playwright  # type: ignore
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(
                    "https://www.kaggle.com/competitions",
                    wait_until="domcontentloaded",
                    timeout=20_000,
                )
                content = await page.content()
                loaded = "kaggle" in content.lower()

                # Check for cookie banner
                banner_visible = await page.is_visible("text=OK, Got it.", timeout=2000)
                if banner_visible:
                    await page.click("text=OK, Got it.")
                    banner_ok = True
                else:
                    banner_ok = None  # not shown (may already be accepted)
            except Exception:
                loaded = False
                banner_ok = False
            await browser.close()
            return True, loaded
    except Exception:
        return False, False


def run_doctor() -> None:
    """Run diagnostic checks and print a health report."""
    print("\n" + "═" * 60)
    print("  kaggle_competition_collector — Doctor")
    print("═" * 60)

    all_ok = True

    # 1. Python
    print("\n[1/6] Python environment")
    py_ok = sys.version_info >= (3, 9)
    all_ok &= _doctor_check(
        f"Python {sys.version.split()[0]}",
        py_ok,
        sys.executable,
    )

    # 2. Packages
    print("\n[2/6] Required packages")
    import importlib.metadata as _im
    for pkg in _DOCTOR_PACKAGES:
        try:
            ver = _im.version(pkg)
            _doctor_check(f"{pkg} {ver}", True)
        except Exception:
            _doctor_check(pkg, False, "NOT installed")
            all_ok = False

    # 3. Kaggle auth
    print("\n[3/6] Kaggle authentication")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    local_kaggle_json = Path(".kaggle") / "kaggle.json"
    found_creds = kaggle_json.exists() or local_kaggle_json.exists()
    all_ok &= _doctor_check(
        "kaggle.json",
        found_creds,
        str(kaggle_json) if kaggle_json.exists() else str(local_kaggle_json),
    )
    if found_creds:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
            api = KaggleApi()
            api.authenticate()
            _doctor_check("API authentication", True)
        except Exception as exc:
            _doctor_check("API authentication", False, str(exc)[:80])
            all_ok = False

    # 4. Playwright installation
    print("\n[4/6] Playwright")
    pw_installed = _check_playwright()
    all_ok &= _doctor_check("playwright package", pw_installed)
    if pw_installed:
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
                capture_output=True, text=True, timeout=10,
            )
            chromium_ok = result.returncode == 0 or "chromium" in (result.stdout + result.stderr).lower()
            _doctor_check("Chromium browser", chromium_ok,
                          "run: python -m playwright install chromium" if not chromium_ok else "")
            if not chromium_ok:
                all_ok = False
        except Exception as exc:
            _doctor_check("Chromium browser", False, str(exc)[:80])
            all_ok = False

    # 5. Browser state
    print("\n[5/6] Browser state")
    state_path = Path.home() / ".kaggle_collector" / "browser_state.json"
    state_exists = state_path.exists()
    _doctor_check(
        "browser_state.json",
        state_exists,
        str(state_path) if state_exists else "not found — run with --headed to create",
    )

    # 6. Live browser test
    print("\n[6/6] Live browser test (headless Kaggle page)")
    if pw_installed:
        launched, loaded = asyncio.run(_doctor_browser_check())
        all_ok &= _doctor_check("Browser launch", launched)
        if launched:
            _doctor_check(
                "Kaggle page loaded",
                loaded,
                "page returned Kaggle content" if loaded else "got crash/empty page",
            )
    else:
        _doctor_check("Browser launch", False, "playwright not installed — skipped")
        all_ok = False

    # Summary
    print("\n" + "═" * 60)
    if all_ok:
        print("  ✅  All checks passed — collector is ready to run.")
    else:
        print("  ⚠️   Some checks failed — see items marked ❌ above.")
        print("  Tip: fix the ❌ items, then re-run: python main.py doctor")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Browser setup/test commands
# ──────────────────────────────────────────────────────────────────────────────

async def run_browser_login() -> int:
    """Open Kaggle with the persistent Chromium profile and wait for manual login."""
    from collector.browser import CHROMIUM_PROFILE_DIR, wait_for_user_interaction
    from playwright.async_api import async_playwright  # type: ignore

    CHROMIUM_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using persistent Chromium profile: {CHROMIUM_PROFILE_DIR}")

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(CHROMIUM_PROFILE_DIR),
            headless=False,
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto("https://www.kaggle.com/", wait_until="domcontentloaded", timeout=45_000)
        print(
            "Login to Kaggle manually in the opened browser. "
            "After login is complete, return here and press ENTER."
        )
        wait_for_user_interaction(
            prompt="",
            unavailable_msg=(
                "Interactive mode is unavailable in this terminal. "
                "Please run `python main.py browser-login` in a real terminal."
            ),
        )

        await page.goto("https://www.kaggle.com/account", wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(2_000)
        text = (await page.evaluate("() => document.body.innerText || ''")).lower()
        logged_in = "sign in" not in text and "login" not in page.url.lower()
        if logged_in:
            print("Kaggle login verified. Persistent profile saved.")
            code = 0
        else:
            print("Kaggle login could not be verified. The profile was kept, but login may be incomplete.")
            code = 1
        await context.close()
        return code


async def run_browser_test(args: argparse.Namespace) -> int:
    """Open each competition tab with the persistent profile and write browser_test_report.md."""
    from collector.browser import CHROMIUM_PROFILE_DIR, COMPETITION_TABS
    from collector.page_extractor import _CRASH_PHRASES, _LOGIN_WALL_PHRASES
    from collector.utils import parse_competition_input
    from playwright.async_api import async_playwright  # type: ignore

    if not args.competition:
        print("Error: browser-test requires --competition", file=sys.stderr)
        return 1

    slug = parse_competition_input(args.competition)
    out = Path(args.output or "./competition_archive").resolve() / slug / "latest"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "browser_test_report.md"
    CHROMIUM_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, int, str]] = []
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(CHROMIUM_PROFILE_DIR),
            headless=False,
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.pages[0] if context.pages else await context.new_page()

        for tab, suffix in COMPETITION_TABS:
            url = f"https://www.kaggle.com/competitions/{slug}{suffix}"
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10_000)
                except Exception:
                    pass
                await page.wait_for_timeout(3_000)
                text = await page.evaluate("() => document.body.innerText || ''")
                lower = text.lower()
                if any(p in lower for p in _LOGIN_WALL_PHRASES):
                    status = "LOGIN_REQUIRED"
                elif any(p in lower for p in _CRASH_PHRASES):
                    status = "CRASH"
                elif len(text.strip()) < 1000:
                    status = "TOO_SHORT"
                else:
                    status = "OK"
                rows.append((tab, status, len(text.strip()), url))
                print(f"{tab:12s} {status:14s} {len(text.strip()):6d} chars")
            except Exception as exc:
                rows.append((tab, "CRASH", 0, f"{url} ({exc})"))
                print(f"{tab:12s} CRASH          0 chars  {exc}")

        await context.close()

    lines = [
        "# Browser Test Report",
        "",
        f"Competition: `{slug}`",
        f"Persistent profile: `{CHROMIUM_PROFILE_DIR}`",
        "",
        "| Tab | Status | Chars | URL |",
        "|-----|--------|-------|-----|",
    ]
    for tab, status, chars, url in rows:
        lines.append(f"| `{tab}` | `{status}` | {chars} | {url} |")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"browser_test_report.md written: {report_path}")

    return 0 if all(status == "OK" for _, status, _, _ in rows) else 2


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # Handle `python main.py doctor` before normal argparse
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        run_doctor()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "browser-login":
        try:
            sys.exit(asyncio.run(run_browser_login()))
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(1)
    if len(sys.argv) > 1 and sys.argv[1] == "browser-test":
        browser_test_parser = argparse.ArgumentParser(prog="kaggle_competition_collector browser-test")
        browser_test_parser.add_argument("--competition", "-c", required=True)
        browser_test_parser.add_argument("--output", "-o", default="./competition_archive")
        try:
            sys.exit(asyncio.run(run_browser_test(browser_test_parser.parse_args(sys.argv[2:]))))
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(1)

    parser = build_parser()
    args = parser.parse_args()

    try:
        code = asyncio.run(run(args))
        sys.exit(code)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
