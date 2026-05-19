"""
Configuration loading.

Priority (highest wins):
  CLI flags  >  config.yaml values  >  dataclass defaults
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Config:
    # ── Identity ──────────────────────────────────────────────────────────────
    competition_slug: str = ""
    output_dir: str = "./competition_archive"

    # ── Collection toggles ────────────────────────────────────────────────────
    download_data: bool = True
    collect_screenshots: bool = True
    collect_notebooks: bool = True
    collect_discussions: bool = True
    profile_data: bool = True

    # ── Limits ────────────────────────────────────────────────────────────────
    max_notebooks: int = 15
    max_discussion_threads: int = 25

    # ── Browser ───────────────────────────────────────────────────────────────
    headless_browser: bool = True
    skip_tabs: list[str] = field(default_factory=list)

    # ── Behaviour ─────────────────────────────────────────────────────────────
    polite_delay_seconds: float = 2.5
    overwrite_cache: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Public loader
# ──────────────────────────────────────────────────────────────────────────────


def load_config(
    config_path: Optional[str],
    cli_overrides: dict[str, Any],
) -> Config:
    """Return a Config merged from YAML file and CLI overrides."""
    cfg = Config()

    if config_path:
        cfg = _apply_yaml(Path(config_path), cfg)

    for key, value in cli_overrides.items():
        if value is None:
            continue
        normalized = key.replace("-", "_")
        if hasattr(cfg, normalized):
            setattr(cfg, normalized, value)

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _apply_yaml(path: Path, cfg: Config) -> Config:
    try:
        import yaml  # type: ignore
    except ImportError:
        print(
            "PyYAML is not installed; ignoring config file. "
            "Run: pip install pyyaml",
            file=sys.stderr,
        )
        return cfg

    if not path.exists():
        print(f"[config] File not found: {path}", file=sys.stderr)
        return cfg

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for key, value in data.items():
        normalized = key.replace("-", "_")
        if hasattr(cfg, normalized):
            setattr(cfg, normalized, value)
        else:
            print(f"[config] Unknown key ignored: '{key}'", file=sys.stderr)

    return cfg
