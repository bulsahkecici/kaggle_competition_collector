"""
Evaluation metric analyzer.

Extracts the metric name from the competition metadata and evaluation page text,
looks it up in a built-in knowledge base, and writes METRIC_EXPLAINED.md.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("kaggle_collector")

# ──────────────────────────────────────────────────────────────────────────────
# Knowledge base — common Kaggle evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────
# Each entry: name aliases → explanation dict
_KB: list[dict[str, Any]] = [
    {
        "aliases": ["auc", "roc auc", "area under roc", "auroc", "roc_auc"],
        "full_name": "Area Under the ROC Curve (AUC-ROC)",
        "higher_is_better": True,
        "task_type": "Binary classification",
        "prediction_format": "Probability for the positive class (float 0–1)",
        "pitfalls": [
            "Measures ranking quality only — a model that perfectly ranks samples scores 1.0 even if its raw probabilities are miscalibrated.",
            "Constant predictions (e.g. all 0.5) provide no ranking signal and yield AUC ≈ 0.5, not 1.0. Always output per-sample probabilities, not a single value.",
            "Hard 0/1 labels instead of probabilities collapse ranking resolution and typically hurt AUC. Submit `predict_proba()[:, 1]`, not `predict()`.",
            "With highly imbalanced data, PR-AUC or log-loss can be more informative alongside ROC AUC.",
        ],
        "scoring_snippet": (
            "from sklearn.metrics import roc_auc_score\n"
            "score = roc_auc_score(y_true, y_pred_proba)\n"
            "# y_pred_proba should be the probability of the POSITIVE class"
        ),
        "validation_tip": "Use StratifiedKFold to preserve the positive-class ratio across folds.",
    },
    {
        "aliases": ["log loss", "logloss", "binary crossentropy", "binary logloss", "logarithmic loss"],
        "full_name": "Logarithmic Loss (Log Loss)",
        "higher_is_better": False,
        "task_type": "Binary or multi-class classification",
        "prediction_format": "Probability for each class (float 0–1, rows must sum to 1 for multi-class)",
        "pitfalls": [
            "Predictions exactly 0 or 1 cause infinite loss — clip probabilities (e.g. 1e-7 to 1-1e-7).",
            "Sensitive to confidence: overconfident wrong predictions are heavily penalised.",
            "Class imbalance inflates the metric; consider using class weights.",
        ],
        "scoring_snippet": (
            "from sklearn.metrics import log_loss\n"
            "score = log_loss(y_true, y_pred_proba)\n"
            "# Clip predictions to avoid log(0): np.clip(y_pred, 1e-7, 1 - 1e-7)"
        ),
        "validation_tip": "StratifiedKFold; monitor both OOF and LB log-loss carefully.",
    },
    {
        "aliases": ["rmse", "root mean square error", "root mean squared error"],
        "full_name": "Root Mean Squared Error (RMSE)",
        "higher_is_better": False,
        "task_type": "Regression",
        "prediction_format": "Continuous numeric prediction",
        "pitfalls": [
            "Sensitive to outliers — a single large error can dominate the score.",
            "The scale depends on the target range; not comparable across datasets.",
            "Consider whether the evaluation is on raw or log-transformed targets.",
        ],
        "scoring_snippet": (
            "from sklearn.metrics import mean_squared_error\n"
            "import numpy as np\n"
            "score = np.sqrt(mean_squared_error(y_true, y_pred))"
        ),
        "validation_tip": "KFold regression; ensure predictions are in the same scale as the target.",
    },
    {
        "aliases": ["mae", "mean absolute error"],
        "full_name": "Mean Absolute Error (MAE)",
        "higher_is_better": False,
        "task_type": "Regression",
        "prediction_format": "Continuous numeric prediction",
        "pitfalls": [
            "Optimal prediction is the conditional median, not the mean.",
            "Less sensitive to outliers than RMSE.",
        ],
        "scoring_snippet": (
            "from sklearn.metrics import mean_absolute_error\n"
            "score = mean_absolute_error(y_true, y_pred)"
        ),
        "validation_tip": "KFold regression; optimise with MAE-friendly objectives (e.g. Huber, MAE objective in LGB).",
    },
    {
        "aliases": ["rmsle", "root mean squared log error", "root mean squared logarithmic error"],
        "full_name": "Root Mean Squared Logarithmic Error (RMSLE)",
        "higher_is_better": False,
        "task_type": "Regression (positive targets)",
        "prediction_format": "Positive continuous numeric prediction",
        "pitfalls": [
            "Penalises under-prediction more than over-prediction.",
            "Predictions must be positive (≥ 0).",
            "Train on log1p(target), then predict and apply expm1 to reverse.",
        ],
        "scoring_snippet": (
            "import numpy as np\n"
            "score = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))"
        ),
        "validation_tip": "Transform target with log1p before training; use RMSE loss internally.",
    },
    {
        "aliases": ["accuracy", "classification accuracy"],
        "full_name": "Classification Accuracy",
        "higher_is_better": True,
        "task_type": "Classification",
        "prediction_format": "Predicted class label (integer or string)",
        "pitfalls": [
            "Misleading on imbalanced datasets — a model always predicting the majority class can score highly.",
            "Does not differentiate confidence levels.",
        ],
        "scoring_snippet": (
            "from sklearn.metrics import accuracy_score\n"
            "score = accuracy_score(y_true, y_pred_classes)"
        ),
        "validation_tip": "StratifiedKFold; check per-class accuracy if classes are imbalanced.",
    },
    {
        "aliases": ["f1", "f1 score", "f1-score", "macro f1", "micro f1", "weighted f1", "binary f1"],
        "full_name": "F1 Score",
        "higher_is_better": True,
        "task_type": "Binary or multi-class classification",
        "prediction_format": "Predicted class label",
        "pitfalls": [
            "Macro-F1 treats all classes equally regardless of support.",
            "Threshold matters for binary F1 — optimise the decision threshold.",
            "Not differentiable; gradient-boosting models need a surrogate loss.",
        ],
        "scoring_snippet": (
            "from sklearn.metrics import f1_score\n"
            "score = f1_score(y_true, y_pred, average='macro')  # adjust average as needed"
        ),
        "validation_tip": "StratifiedKFold; tune decision threshold on OOF predictions.",
    },
    {
        "aliases": ["qwk", "quadratic weighted kappa", "cohen kappa", "kappa"],
        "full_name": "Quadratic Weighted Kappa (QWK)",
        "higher_is_better": True,
        "task_type": "Ordinal regression / multi-class classification",
        "prediction_format": "Ordinal class label (integer)",
        "pitfalls": [
            "Very sensitive to the optimal thresholds used to convert continuous predictions to ordinal labels.",
            "Optimise thresholds on OOF predictions.",
            "Using regression with rounding often beats classification approaches.",
        ],
        "scoring_snippet": (
            "from sklearn.metrics import cohen_kappa_score\n"
            "score = cohen_kappa_score(y_true, y_pred_rounded, weights='quadratic')"
        ),
        "validation_tip": "KFold or StratifiedKFold; optimise rounding thresholds via scipy.optimize.",
    },
    {
        "aliases": ["map", "mean average precision", "map@k", "map@3", "map@5", "map@12"],
        "full_name": "Mean Average Precision (MAP / MAP@K)",
        "higher_is_better": True,
        "task_type": "Ranking / multi-label retrieval",
        "prediction_format": "Ranked list of predictions per query (top-K)",
        "pitfalls": [
            "Order of predictions matters — the highest-confidence predictions must appear first.",
            "MAP@K ignores relevant items ranked beyond K.",
        ],
        "scoring_snippet": (
            "def apk(actual, predicted, k=5):\n"
            "    if len(predicted) > k: predicted = predicted[:k]\n"
            "    score, hits = 0.0, 0\n"
            "    for i, p in enumerate(predicted):\n"
            "        if p in actual and p not in predicted[:i]:\n"
            "            hits += 1\n"
            "            score += hits / (i + 1)\n"
            "    return score / min(len(actual), k) if actual else 0.0\n\n"
            "def mapk(actual, predicted, k=5):\n"
            "    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)"
        ),
        "validation_tip": "GroupKFold on query ID; confirm K value from competition description.",
    },
    {
        "aliases": ["dice", "dice coefficient", "dice loss", "sorensen dice"],
        "full_name": "Dice Coefficient",
        "higher_is_better": True,
        "task_type": "Image segmentation / binary overlap",
        "prediction_format": "Binary mask (pixel predictions)",
        "pitfalls": [
            "Sensitive to small regions — a missed small object heavily penalises the score.",
            "Requires run-length encoding (RLE) for submission in many competitions.",
        ],
        "scoring_snippet": (
            "def dice_coef(y_true, y_pred, smooth=1):\n"
            "    y_true_f = y_true.flatten()\n"
            "    y_pred_f = y_pred.flatten()\n"
            "    intersection = (y_true_f * y_pred_f).sum()\n"
            "    return (2 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)"
        ),
        "validation_tip": "Use image-level or fold-by-patient split to avoid data leakage in medical imaging.",
    },
    {
        "aliases": ["iou", "intersection over union", "jaccard", "jaccard index"],
        "full_name": "Intersection over Union (IoU / Jaccard)",
        "higher_is_better": True,
        "task_type": "Object detection / segmentation",
        "prediction_format": "Bounding boxes or binary masks",
        "pitfalls": [
            "Mean IoU can be misleadingly high if background class dominates.",
            "Threshold for positive match (e.g. IoU > 0.5) matters for mAP.",
        ],
        "scoring_snippet": (
            "def iou(y_true, y_pred):\n"
            "    intersection = (y_true & y_pred).sum()\n"
            "    union = (y_true | y_pred).sum()\n"
            "    return intersection / union if union > 0 else 0.0"
        ),
        "validation_tip": "Stratify by class frequency; ensure train/val splits preserve object distribution.",
    },
    {
        "aliases": ["mcc", "matthews correlation coefficient", "matthew correlation"],
        "full_name": "Matthews Correlation Coefficient (MCC)",
        "higher_is_better": True,
        "task_type": "Binary classification (imbalanced)",
        "prediction_format": "Predicted class label",
        "pitfalls": [
            "Range is −1 to +1; 0 means no better than random.",
            "Requires careful threshold optimisation.",
        ],
        "scoring_snippet": (
            "from sklearn.metrics import matthews_corrcoef\n"
            "score = matthews_corrcoef(y_true, y_pred_classes)"
        ),
        "validation_tip": "StratifiedKFold; optimise threshold on OOF predictions.",
    },
    {
        "aliases": ["spearman", "spearman correlation", "spearman rank"],
        "full_name": "Spearman's Rank Correlation Coefficient",
        "higher_is_better": True,
        "task_type": "Ranking / regression",
        "prediction_format": "Continuous numeric prediction",
        "pitfalls": [
            "Measures rank correlation, not value accuracy — even perfectly monotone but scaled wrong predictions score 1.0.",
            "Not differentiable; use a surrogate loss during training.",
        ],
        "scoring_snippet": (
            "from scipy.stats import spearmanr\n"
            "score, pval = spearmanr(y_true, y_pred)"
        ),
        "validation_tip": "KFold; ensembling typically helps rank-based metrics.",
    },
    {
        "aliases": ["mape", "mean absolute percentage error"],
        "full_name": "Mean Absolute Percentage Error (MAPE)",
        "higher_is_better": False,
        "task_type": "Regression (positive targets)",
        "prediction_format": "Positive continuous numeric prediction",
        "pitfalls": [
            "Undefined when y_true = 0 — exclude or clip zero-target rows.",
            "Asymmetric: under-predictions are penalised more than over-predictions of the same magnitude.",
        ],
        "scoring_snippet": (
            "import numpy as np\n"
            "score = np.mean(np.abs((y_true - y_pred) / y_true))"
        ),
        "validation_tip": "KFold; consider SMAPE if some targets are near zero.",
    },
    {
        "aliases": ["ndcg", "normalized discounted cumulative gain", "ndcg@k"],
        "full_name": "Normalized Discounted Cumulative Gain (NDCG)",
        "higher_is_better": True,
        "task_type": "Ranking",
        "prediction_format": "Relevance scores for each item (float)",
        "pitfalls": [
            "Items ranked lower contribute exponentially less — top positions matter most.",
            "Confirm whether K is fixed or varies per query.",
        ],
        "scoring_snippet": (
            "from sklearn.metrics import ndcg_score\n"
            "score = ndcg_score([y_true], [y_pred], k=10)"
        ),
        "validation_tip": "GroupKFold on query/user ID.",
    },
]


class MetricAnalyzer:
    """Extracts and explains the competition evaluation metric."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def analyze(
        self,
        api_metric: Optional[str],
        eval_page_text: str,
        overview_page_text: str = "",
        existing_metadata: Optional[dict[str, Any]] = None,
        competition_slug: str = "",
    ) -> dict[str, Any]:
        """
        Determine the metric, look it up in the KB, and write METRIC_EXPLAINED.md.
        Also updates competition_metadata.json with the parsed metric.

        Parameters
        ----------
        api_metric         : raw metric string from the Kaggle API (may be None)
        eval_page_text     : full text of the evaluation page
        overview_page_text : overview page text as secondary source
        existing_metadata  : existing competition_metadata.json content to augment
        competition_slug   : used for slug_known_fallback when all else fails
        """
        source = "unknown"

        # 1. Try API metric first (from this run's live API call)
        raw = api_metric.strip() if api_metric else None
        if raw:
            source = "api_metadata"

        # 1.5. If this run's API returned no metric, check existing_metadata
        #      (it may carry evaluationMetric from a previous successful API run)
        if not raw and existing_metadata:
            em = (existing_metadata.get("evaluationMetric")
                  or existing_metadata.get("evaluation_metric"))
            if em and str(em).strip() not in ("None", "null", ""):
                raw = str(em).strip()
                source = "api_metadata"
                logger.info(f"  Metric from existing_metadata.evaluationMetric: {raw}")

        # 2. Search evaluation page — full text, no truncation
        if not raw and eval_page_text:
            raw = _extract_metric_from_text(eval_page_text)
            if raw:
                source = "evaluation_page"

        # 3. Fall back to overview page
        if not raw and overview_page_text:
            raw = _extract_metric_from_text(overview_page_text)
            if raw:
                source = "overview_page"

        # 4. Direct alias scan of evaluation text (catches e.g. bare "RMSE" headings)
        if not raw and eval_page_text:
            raw = _scan_for_known_alias(eval_page_text)
            if raw:
                source = "evaluation_page"

        if not raw and overview_page_text:
            raw = _scan_for_known_alias(overview_page_text)
            if raw:
                source = "overview_page"

        # 5. Try loading evaluationMetric from an existing competition_metadata.json
        #    (covers the case where this run's API call failed but a prior run saved it)
        if not raw:
            cached_raw, cached_source = self._load_cached_metric()
            if cached_raw:
                raw = cached_raw
                source = cached_source or "api_metadata"
                logger.info(
                    f"  Metric recovered from cached competition_metadata.json: {raw} "
                    f"(source={source})"
                )

        # 6. Slug-known fallback — hardcoded for well-known competition series
        if not raw and competition_slug:
            known = _slug_known_metric(competition_slug)
            if known:
                raw = known
                source = "slug_known_fallback"

        entry = _lookup(raw) if raw else None

        # Determine confidence
        if source == "api_metadata":
            confidence = "high"
        elif source in ("evaluation_page", "overview_page"):
            confidence = "high"
        elif source == "slug_known_fallback":
            confidence = "medium"
        else:
            confidence = "low"

        result: dict[str, Any] = {
            "raw_metric": raw,
            "matched_entry": entry,
            "source": source,
            "confidence": confidence,
        }

        self._write_report(raw, eval_page_text, entry)
        self._update_metadata_json(raw, entry, existing_metadata, source, confidence)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Report writer
    # ──────────────────────────────────────────────────────────────────────────

    def _load_cached_metric(self) -> tuple[Optional[str], str]:
        """
        Read metric from an existing competition_metadata.json.
        Returns (raw_metric_string, source_string) or (None, "").

        Priority:
          1. evaluationMetric / evaluation_metric (from Kaggle API)
          2. parsed_metric.raw + parsed_metric.source (from a prior successful analysis)
        """
        meta_path = self.output_dir / "competition_metadata.json"
        if not meta_path.exists():
            return None, ""
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            val = data.get("evaluationMetric") or data.get("evaluation_metric")
            if val and str(val).strip() not in ("None", "null", ""):
                return str(val).strip(), "api_metadata"
            # Fall back to previously parsed metric (preserves prior run's work)
            pm = data.get("parsed_metric") or {}
            raw = pm.get("raw") or pm.get("name")
            prev_source = pm.get("source", "")
            if raw and str(raw).strip() not in ("None", "null", ""):
                return str(raw).strip(), prev_source or "api_metadata"
        except Exception:
            pass
        return None, ""

    def _write_report(
        self,
        raw_metric: Optional[str],
        eval_text: str,
        entry: Optional[dict[str, Any]],
    ) -> None:
        lines: list[str] = ["# Metric Explained\n"]

        if raw_metric:
            lines.append(f"**Raw metric string (from Kaggle API / page):** `{raw_metric}`\n")
        else:
            lines.append("_Could not determine metric from API or evaluation page._\n")

        if entry:
            lines += [
                f"## {entry['full_name']}\n",
                f"- **Higher is better:** {'Yes ✅' if entry['higher_is_better'] else 'No — lower is better ✅'}",
                f"- **Task type:** {entry['task_type']}",
                f"- **Prediction format:** {entry['prediction_format']}",
                "",
                "## Common Pitfalls\n",
            ]
            for p in entry["pitfalls"]:
                lines.append(f"- {p}")
            lines += [
                "",
                "## Minimal Python Scoring Function\n",
                "```python",
                entry["scoring_snippet"],
                "```",
                "",
                "## Validation Strategy Recommendation\n",
                entry["validation_tip"],
                "",
            ]
        else:
            lines += [
                "## Not Found in Knowledge Base\n",
                "The metric was not matched to the built-in knowledge base.",
                "Please refer to the evaluation page text below for the exact definition.",
                "",
            ]

        # Always include the raw evaluation page text
        if eval_text.strip():
            lines += [
                "## Evaluation Page Text (from Kaggle)\n",
                "---\n",
                eval_text[:8_000],
                "",
            ]

        _write(self.output_dir / "METRIC_EXPLAINED.md", "\n".join(lines))
        logger.info("METRIC_EXPLAINED.md written")

    def _update_metadata_json(
        self,
        raw_metric: Optional[str],
        entry: Optional[dict[str, Any]],
        existing: Optional[dict[str, Any]],
        source: str = "unknown",
        confidence: str = "low",
    ) -> None:
        """Merge metric findings into competition_metadata.json."""
        meta_path = self.output_dir / "competition_metadata.json"

        # Load existing metadata (from API or previous run).
        # Always start from the saved file to preserve fields from a prior successful run,
        # then overlay the current run's data (skipping None values so we don't erase
        # previously collected fields like evaluationMetric).
        base: dict[str, Any] = {}
        if meta_path.exists():
            try:
                base = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                base = {}
        if existing:
            for k, v in existing.items():
                if v is not None:
                    base[k] = v

        # Normalize the metric name for display
        display_name = None
        if entry:
            display_name = entry.get("full_name")
        elif raw_metric:
            # Try to produce a clean name from the raw string
            display_name = raw_metric.strip().title()

        base["parsed_metric"] = {
            "name": display_name or raw_metric,
            "raw": raw_metric,
            "full_name": entry.get("full_name") if entry else display_name,
            "higher_is_better": entry.get("higher_is_better") if entry else None,
            "task_type": entry.get("task_type") if entry else None,
            "prediction_format": entry.get("prediction_format") if entry else None,
            "matched": entry is not None,
            "source": source,
            "confidence": confidence,
        }

        _write(meta_path, json.dumps(base, indent=2, default=str))
        logger.info("competition_metadata.json updated with parsed metric")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _lookup(raw: str) -> Optional[dict[str, Any]]:
    """Find the best matching KB entry for a raw metric string."""
    canon = raw.lower().strip()
    for entry in _KB:
        for alias in entry["aliases"]:
            if alias in canon or canon in alias:
                return entry
    return None


def _scan_for_known_alias(text: str) -> Optional[str]:
    """
    Scan the full text for any known metric alias as a standalone word/phrase.
    Returns the matched alias string if found (longest match wins).
    """
    canon = text.lower()
    best: Optional[str] = None
    best_len = 0
    for entry in _KB:
        for alias in entry["aliases"]:
            if len(alias) <= best_len:
                continue
            # Require word boundary around the alias
            pat = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
            if re.search(pat, canon):
                best = alias
                best_len = len(alias)
    return best


def _extract_metric_from_text(text: str) -> Optional[str]:
    """
    Extract a metric name from evaluation/overview page text using
    common sentence patterns. Searches the FULL text (no truncation).
    """
    patterns = [
        r"scored\s+(?:on|using|with|by)\s+([A-Za-z0-9@_ ()/-]{2,50}?)(?:[.,;\n]|$)",
        r"evaluated\s+(?:on|using|with)\s+([A-Za-z0-9@_ ()/-]{2,50}?)(?:[.,;\n]|$)",
        r"evaluation\s+metric\s*[:\-–]\s*([A-Za-z0-9@_ ()/-]{2,50}?)(?:[.,;\n]|$)",
        r"metric\s*[:\-–]\s*([A-Za-z0-9@_ ()/-]{2,50}?)(?:[.,;\n]|$)",
        r"performance\s+(?:is\s+)?measured\s+(?:by|using|with)\s+([A-Za-z0-9@_ ()/-]{2,50}?)(?:[.,;\n]|$)",
        r"optimized\s+for\s+([A-Za-z0-9@_ ()/-]{2,50}?)(?:[.,;\n]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            candidate = m.group(1).strip().rstrip(".,;: ")
            if 2 < len(candidate) < 60:
                return candidate
    return None


def _slug_known_metric(slug: str) -> Optional[str]:
    """
    Return a known metric string for well-known competition slugs/series.
    Used as last-resort fallback when API and page extraction both fail.
    """
    slug_lower = slug.lower()
    # Playground series season 6: AUC-based (s6e5 is binary classification)
    if slug_lower.startswith("playground-series-s6"):
        return "roc auc"
    # Titanic, most binary classification playgrounds
    if "titanic" in slug_lower:
        return "accuracy"
    return None


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
