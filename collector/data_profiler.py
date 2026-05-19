"""
Data profiler — inspects downloaded competition files and generates markdown reports.

Supports: CSV, Parquet, Feather, JSON (records), XLSX.
Handles large files by sampling when memory would be an issue.

Outputs:
  data_profile/file_profiles/<name>_profile.md
  data_profile/data_schema.json
  data_profile/train_test_column_diff.md
  data_profile/missing_values_report.md
  data_profile/target_detection_report.md
  data_profile/SUBMISSION_FORMAT.md          ← at output root
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("kaggle_collector")

# Max rows to load for profiling very large files (avoids OOM)
_MAX_PROFILE_ROWS = 500_000

# Extensions we know how to profile
_READABLE_EXTS = {".csv", ".parquet", ".feather", ".json", ".xlsx", ".xls"}


class DataProfiler:
    """Profiles every readable data file in the data/ folder."""

    def __init__(self, output_dir: Path) -> None:
        self.data_dir = output_dir / "data"
        self.profile_dir = output_dir / "data_profile"
        self.file_profiles_dir = self.profile_dir / "file_profiles"
        self.output_dir = output_dir
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.file_profiles_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def profile_all(self) -> dict[str, Any]:
        """
        Profile every supported file in data/.
        Returns a results dict with profile metadata for each file.
        """
        try:
            import pandas as pd  # type: ignore  # noqa: F401
        except ImportError:
            logger.error(
                "pandas is required for data profiling. "
                "Run: pip install pandas pyarrow openpyxl"
            )
            return {}

        if not self.data_dir.exists():
            logger.warning("data/ directory not found; skipping data profiling")
            return {}

        files = [
            f for f in self.data_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in _READABLE_EXTS
        ]
        if not files:
            logger.warning("No readable data files found in data/")
            return {}

        logger.info(f"Profiling {len(files)} data file(s) …")
        all_profiles: dict[str, Any] = {}

        for f in sorted(files):
            logger.info(f"  → {f.name}")
            try:
                profile = self._profile_file(f)
            except Exception as exc:
                logger.warning(f"  Profiling failed for {f.name}: {exc}")
                profile = {"error": str(exc), "file": f.name}
            if profile:
                all_profiles[f.name] = profile
                self._write_file_profile(f.name, profile)

        self._write_schema(all_profiles)
        self._write_missing_report(all_profiles)
        self._write_column_diff(all_profiles)
        self._write_target_report(all_profiles)
        self._write_submission_format(all_profiles)
        return all_profiles

    # ──────────────────────────────────────────────────────────────────────────
    # File loading
    # ──────────────────────────────────────────────────────────────────────────

    def _profile_file(self, path: Path) -> Optional[dict[str, Any]]:
        import pandas as pd

        ext = path.suffix.lower()
        try:
            file_size_mb = path.stat().st_size / 1_048_576

            if ext == ".csv":
                if file_size_mb > 200:
                    df = pd.read_csv(path, nrows=_MAX_PROFILE_ROWS, low_memory=False)
                    truncated = True
                else:
                    df = pd.read_csv(path, low_memory=False)
                    truncated = False
            elif ext == ".parquet":
                df = pd.read_parquet(path)
                truncated = False
            elif ext == ".feather":
                df = pd.read_feather(path)
                truncated = False
            elif ext in (".json",):
                df = pd.read_json(path, orient="records", lines=True)
                truncated = False
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(path)
                truncated = False
            else:
                return None

        except Exception as exc:
            logger.warning(f"  Could not read {path.name}: {exc}")
            return {"error": str(exc), "file": path.name}

        return self._build_profile(df, path, file_size_mb, truncated)

    # ──────────────────────────────────────────────────────────────────────────
    # Profile building
    # ──────────────────────────────────────────────────────────────────────────

    def _build_profile(self, df, path: Path, size_mb: float, truncated: bool) -> dict[str, Any]:
        import pandas as pd

        null_counts = df.isnull().sum().to_dict()
        null_pct = (df.isnull().sum() / max(len(df), 1) * 100).round(2).to_dict()
        unique_counts = _safe_unique_counts(df)
        dtypes = df.dtypes.astype(str).to_dict()
        mem_mb = df.memory_usage(deep=True).sum() / 1_048_576

        # Column classification
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # Heuristic: ID columns (name contains "id", unique == row count)
        n = len(df)
        id_cols = [
            c for c in df.columns
            if re.search(r"\bid\b|_id$|^id_", c, re.IGNORECASE)
            or (unique_counts.get(c, 0) == n and n > 0)
        ]

        # Date column detection by column name
        date_name_cols = [
            c for c in df.columns
            if re.search(r"date|time|year|month|week|day", c, re.IGNORECASE)
            and c not in dt_cols
        ]
        dt_cols = list(set(dt_cols + date_name_cols))

        # Safe head (convert to json-serializable)
        try:
            head_rows = json.loads(df.head(5).to_json(orient="records", default_handler=str))
        except Exception:
            head_rows = []

        return {
            "file": path.name,
            "file_size_mb": round(size_mb, 2),
            "in_memory_mb": round(mem_mb, 2),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": dtypes,
            "null_counts": {k: int(v) for k, v in null_counts.items()},
            "null_pct": null_pct,
            "unique_counts": {k: int(v) for k, v in unique_counts.items()},
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
            "datetime_columns": dt_cols,
            "id_columns": id_cols,
            "head": head_rows,
            "truncated_at": _MAX_PROFILE_ROWS if truncated else None,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Report writers
    # ──────────────────────────────────────────────────────────────────────────

    def _write_file_profile(self, filename: str, p: dict[str, Any]) -> None:
        safe = re.sub(r"\W+", "_", filename)
        lines: list[str] = [
            f"# Data Profile: {filename}\n",
            f"- **File size:** {p.get('file_size_mb', '?')} MB",
            f"- **In-memory:** {p.get('in_memory_mb', '?')} MB",
            f"- **Rows:** {p.get('rows', '?'):,}",
            f"- **Columns:** {p.get('columns', '?')}",
        ]
        if p.get("truncated_at"):
            lines.append(f"- ⚠️ **Profiled on first {p['truncated_at']:,} rows only (file is large)**")
        lines.append("")

        if "error" in p:
            lines.append(f"⚠️ **Error loading file:** {p['error']}")
            _write(self.file_profiles_dir / f"{safe}_profile.md", "\n".join(lines))
            return

        # Column overview table
        cols = p.get("column_names", [])
        dtypes = p.get("dtypes", {})
        nulls = p.get("null_counts", {})
        null_pct = p.get("null_pct", {})
        uniq = p.get("unique_counts", {})

        lines += [
            "## Columns\n",
            "| Column | Dtype | Nulls | Null% | Unique |",
            "|--------|-------|-------|-------|--------|",
        ]
        for c in cols:
            lines.append(
                f"| `{c}` | {dtypes.get(c,'?')} | {nulls.get(c,0)} "
                f"| {null_pct.get(c,0):.1f}% | {uniq.get(c,0)} |"
            )
        lines.append("")

        # Classified columns
        if p.get("id_columns"):
            lines.append(f"**Likely ID columns:** {', '.join(f'`{c}`' for c in p['id_columns'])}\n")
        if p.get("datetime_columns"):
            lines.append(f"**Date/time columns:** {', '.join(f'`{c}`' for c in p['datetime_columns'])}\n")
        if p.get("numeric_columns"):
            lines.append(f"**Numeric columns:** {', '.join(f'`{c}`' for c in p['numeric_columns'][:20])}")
            if len(p["numeric_columns"]) > 20:
                lines.append(f" _(+{len(p['numeric_columns'])-20} more)_")
            lines.append("")
        if p.get("categorical_columns"):
            lines.append(f"**Categorical columns:** {', '.join(f'`{c}`' for c in p['categorical_columns'][:20])}")
            if len(p["categorical_columns"]) > 20:
                lines.append(f" _(+{len(p['categorical_columns'])-20} more)_")
            lines.append("")

        # First 5 rows
        head = p.get("head", [])
        if head:
            lines.append("## First 5 Rows\n")
            if cols:
                lines.append("| " + " | ".join(str(c) for c in cols) + " |")
                lines.append("| " + " | ".join("---" for _ in cols) + " |")
                for row in head:
                    lines.append("| " + " | ".join(str(row.get(c, ""))[:40] for c in cols) + " |")
            lines.append("")

        _write(self.file_profiles_dir / f"{safe}_profile.md", "\n".join(lines))

    def _write_schema(self, profiles: dict[str, Any]) -> None:
        schema: dict[str, Any] = {}
        for fname, p in profiles.items():
            schema[fname] = {
                "rows": p.get("rows"),
                "columns": p.get("columns"),
                "dtypes": p.get("dtypes", {}),
            }
        _write(
            self.profile_dir / "data_schema.json",
            json.dumps(schema, indent=2, default=str),
        )

    def _write_missing_report(self, profiles: dict[str, Any]) -> None:
        lines = ["# Missing Values Report\n"]
        for fname, p in profiles.items():
            if "error" in p:
                continue
            nulls = {c: v for c, v in p.get("null_counts", {}).items() if v > 0}
            lines.append(f"## {fname}\n")
            if not nulls:
                lines.append("No missing values detected.\n")
            else:
                null_pct = p.get("null_pct", {})
                lines += ["| Column | Nulls | %Missing |", "|--------|-------|---------|"]
                for c, n in sorted(nulls.items(), key=lambda x: -x[1]):
                    lines.append(f"| `{c}` | {n:,} | {null_pct.get(c, 0):.1f}% |")
                lines.append("")
        _write(self.profile_dir / "missing_values_report.md", "\n".join(lines))

    def _write_column_diff(self, profiles: dict[str, Any]) -> None:
        train_key = _fuzzy_match(profiles, "train")
        test_key = _fuzzy_match(profiles, "test")

        if not train_key or not test_key:
            _write(
                self.profile_dir / "train_test_column_diff.md",
                "_Could not identify train and test files for column comparison._",
            )
            return

        train_cols = set(profiles[train_key].get("column_names", []))
        test_cols = set(profiles[test_key].get("column_names", []))

        only_train = sorted(train_cols - test_cols)
        only_test = sorted(test_cols - train_cols)
        shared = sorted(train_cols & test_cols)

        lines = [
            "# Train vs Test Column Difference\n",
            f"- **Train file:** `{train_key}`  ({len(train_cols)} columns)",
            f"- **Test file:** `{test_key}`  ({len(test_cols)} columns)\n",
            f"**Columns only in train ({len(only_train)}):** "
            + (", ".join(f"`{c}`" for c in only_train) or "_none_"),
            "",
            f"**Columns only in test ({len(only_test)}):** "
            + (", ".join(f"`{c}`" for c in only_test) or "_none_"),
            "",
            f"**Shared columns ({len(shared)}):** "
            + (", ".join(f"`{c}`" for c in shared[:30]) or "_none_"),
            "",
        ]
        if only_train:
            lines += [
                "## Target / Label Candidates (in train, not in test)\n",
                "These columns exist in train but not in test — likely the target(s):\n",
            ]
            for c in only_train:
                lines.append(f"- `{c}`")
            lines.append("")

        _write(self.profile_dir / "train_test_column_diff.md", "\n".join(lines))

    def _write_target_report(self, profiles: dict[str, Any]) -> None:
        train_key = _fuzzy_match(profiles, "train")
        test_key = _fuzzy_match(profiles, "test")
        sample_key = _fuzzy_match(profiles, "sample_submission")

        lines = ["# Target Detection Report\n"]

        # Derive target from sample_submission
        if sample_key:
            s = profiles[sample_key]
            sub_cols = s.get("column_names", [])
            id_cols = s.get("id_columns", [])
            target_cols = [c for c in sub_cols if c not in id_cols]
            lines += [
                f"**Source:** `{sample_key}` (columns in submission define the target)\n",
                f"**All submission columns:** {', '.join(f'`{c}`' for c in sub_cols)}\n",
                f"**Inferred ID column(s):** {', '.join(f'`{c}`' for c in id_cols) or '_none_'}\n",
                f"**Inferred target column(s):** {', '.join(f'`{c}`' for c in target_cols) or '_none_'}\n",
            ]
        elif train_key and test_key:
            train_cols = set(profiles[train_key].get("column_names", []))
            test_cols = set(profiles[test_key].get("column_names", []))
            only_train = sorted(train_cols - test_cols)
            lines += [
                "_No sample_submission.csv found. Using train/test column diff._\n",
                f"**Columns in train but not test (likely target):** "
                f"{', '.join(f'`{c}`' for c in only_train) or '_none_'}\n",
            ]
        else:
            lines.append("_Could not determine target columns from available files._\n")

        _write(self.profile_dir / "target_detection_report.md", "\n".join(lines))

    def _write_submission_format(self, profiles: dict[str, Any]) -> None:
        sample_key = _fuzzy_match(profiles, "sample_submission")
        if not sample_key:
            _write(
                self.output_dir / "SUBMISSION_FORMAT.md",
                "# Submission Format\n\n_No sample_submission file found._",
            )
            return

        p = profiles[sample_key]
        cols = p.get("column_names", [])
        dtypes = p.get("dtypes", {})
        head = p.get("head", [])
        id_cols = p.get("id_columns", [])
        target_cols = [c for c in cols if c not in id_cols]

        lines = [
            "# Submission Format\n",
            f"Based on `{sample_key}`\n",
            f"**Required columns:** {', '.join(f'`{c}`' for c in cols)}\n",
            f"**ID column(s):** {', '.join(f'`{c}`' for c in id_cols) or '_none_'}\n",
            f"**Target column(s):** {', '.join(f'`{c}`' for c in target_cols) or '_unknown_'}\n",
            "",
            "## Column Types\n",
            "| Column | Dtype |",
            "|--------|-------|",
        ]
        for c in cols:
            lines.append(f"| `{c}` | {dtypes.get(c, '?')} |")
        lines.append("")

        if head:
            lines.append("## Example Rows\n")
            lines.append("| " + " | ".join(str(c) for c in cols) + " |")
            lines.append("| " + " | ".join("---" for _ in cols) + " |")
            for row in head[:3]:
                lines.append("| " + " | ".join(str(row.get(c, ""))[:30] for c in cols) + " |")
            lines.append("")

        lines += [
            "## Notes for Building a Valid Submission\n",
            "1. Your submission CSV must contain **exactly** these columns (no extras, no missing).",
            "2. The row count must match the test set.",
            f"3. ID column(s) must match `test.csv` exactly: {', '.join(f'`{c}`' for c in id_cols)}",
            "4. Target column(s) must contain predictions in the correct format.",
        ]

        _write(self.output_dir / "SUBMISSION_FORMAT.md", "\n".join(lines))
        logger.info("SUBMISSION_FORMAT.md written")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _fuzzy_match(profiles: dict[str, Any], keyword: str) -> Optional[str]:
    """Return the profile key whose filename best matches the keyword."""
    for k in profiles:
        if keyword.lower().replace("_", "") in k.lower().replace("_", ""):
            return k
    return None


def _safe_unique_counts(df) -> dict[str, int]:
    """
    Return per-column unique counts even when cells contain unhashable lists/dicts.
    Pandas nunique() cannot hash nested JSON objects, which is common in task files.
    """
    counts: dict[str, int] = {}
    for col in df.columns:
        series = df[col]
        try:
            counts[col] = int(series.nunique(dropna=False))
            continue
        except TypeError:
            pass

        def _stable_value(value: Any) -> str:
            try:
                return json.dumps(value, sort_keys=True, default=str)
            except Exception:
                return str(value)

        try:
            counts[col] = int(series.map(_stable_value).nunique(dropna=False))
        except Exception:
            counts[col] = 0
    return counts


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
