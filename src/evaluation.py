"""Shared evaluation metrics and diagnostic plots for binary damage prediction.

Designed so every model family (baselines, logistic regression, random forest,
gradient boosting, neural network) reports the exact same metrics on the same split.

Rules: evaluate on validation during development/tuning; touch test once at the end.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_precision(y_true, y_prob, min_precision: float) -> float:
    """Highest recall achieved on the PR curve where precision >= min_precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # Exclude the final sentinel point (precision=1, recall=0) added by sklearn
    precision, recall = precision[:-1], recall[:-1]
    mask = precision >= min_precision
    if not np.any(mask):
        return 0.0
    return float(recall[mask].max())


def best_f1_on_curve(y_true, y_prob) -> tuple[float, float]:
    """Return (best_f1, threshold) over all thresholds on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    if len(thresholds) == 0:
        return 0.0, 0.5
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(thresholds[idx])


def evaluate_predictions(
    y_true,
    y_prob,
    model_name: str,
    train_seconds: float | None = None,
) -> dict:
    """Compute the full proposal metric suite on a single evaluation split."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    f1_best, best_thr = best_f1_on_curve(y_true, y_prob)

    row = {
        "model": model_name,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "recall_at_p20": recall_at_precision(y_true, y_prob, 0.20),
        "recall_at_p30": recall_at_precision(y_true, y_prob, 0.30),
        "recall_at_p50": recall_at_precision(y_true, y_prob, 0.50),
        "f1_at_05": float(f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)),
        "f1_best": f1_best,
        "best_f1_threshold": best_thr,
        "mean_pred_prob": float(y_prob.mean()),
    }
    if train_seconds is not None:
        row["train_seconds"] = round(float(train_seconds), 2)
    return row


def sanitize_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    """Replace characters disallowed by LightGBM/XGBoost feature-name parsers."""
    X = X.copy()
    seen: dict[str, int] = {}
    new_cols = []
    for col in X.columns:
        clean = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") or "feat"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}__{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols.append(clean)
    X.columns = new_cols
    return X


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    base_rate: float,
    path: Path,
    title: str = "Validation Precision-Recall Curves (2023-2024)",
) -> None:
    """Plot precision-recall curves for multiple models with the no-skill prevalence line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, (y_true, y_prob) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.step(recall, precision, where="post", linewidth=2.0, label=f"{label} (PR-AUC={ap:.4f})")
    ax.axhline(
        base_rate,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"No-skill prevalence ({base_rate:.4f})",
    )
    ax.set_xlabel("Recall (sensitivity on damaging strikes)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    n_bins: int = 10,
    title: str = "Validation Calibration Curves (2023-2024)",
) -> None:
    """Plot reliability diagrams (quantile-binned predicted vs observed damage rate)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Perfect calibration")
    for label, (y_true, y_prob) in curves.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(y_true, y_prob)
        ax.plot(prob_pred, prob_true, marker="o", linewidth=1.8, label=f"{label} (Brier={brier:.4f})")
    ax.set_xlabel("Mean predicted damage probability")
    ax.set_ylabel("Observed damage rate")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
