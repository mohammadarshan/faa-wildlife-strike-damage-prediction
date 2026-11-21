"""Phase 2 baselines: majority-class/prior-rate predictor and one-feature SIZE rule.

Run from the repo root:
    python -m src.baselines

Both baselines are fitted on TRAIN only and evaluated on VALIDATION (2023-2024).
Test (2025) is never touched here.
"""
import time

import numpy as np
import pandas as pd

from src.data_loading import ROOT
from src.evaluation import evaluate_predictions
from src.splits import load_split_data

RESULTS_DIR = ROOT / "reports" / "results"


class MajorityBaseline:
    """Always predicts the training damage prevalence (no-skill benchmark)."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.prior_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.prior_, dtype=float)


class OneFeatureRateBaseline:
    """Predicts empirical P(damage | column) learned on train, with '(blank)' for missing values."""

    def __init__(self, column: str = "SIZE", smoothing: float = 1.0):
        self.column = column
        self.smoothing = smoothing

    @staticmethod
    def _clean(s: pd.Series) -> pd.Series:
        return s.astype(object).where(s.notna(), "(blank)").astype(str)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.prior_ = float(np.mean(y))
        col = self._clean(X[self.column])
        stats = (
            pd.DataFrame({"val": col, "y": np.asarray(y, dtype=int)})
            .groupby("val")["y"]
            .agg(n="size", damaged="sum", raw_rate="mean")
        )
        stats["rate"] = (stats["damaged"] + self.smoothing * self.prior_) / (stats["n"] + self.smoothing)
        self.table_ = stats.sort_values("rate", ascending=False)
        self.rates_ = self.table_["rate"].to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        col = self._clean(X[self.column])
        return col.map(self.rates_).fillna(self.prior_).to_numpy(dtype=float)


def run_baselines(scheme: str = "main", save: bool = True) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Fit baselines on train and evaluate on validation. Returns (metrics_df, val_probs, size_table)."""
    splits = load_split_data(scheme)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]

    rows = []
    val_probs = {}

    t0 = time.perf_counter()
    maj = MajorityBaseline().fit(X_train, y_train)
    t_maj = time.perf_counter() - t0
    p_maj = maj.predict_proba(X_val)
    rows.append(evaluate_predictions(y_val, p_maj, "Baseline: Majority / Prior", train_seconds=t_maj))
    val_probs["Baseline: Majority / Prior"] = (np.asarray(y_val, dtype=int), p_maj)

    t0 = time.perf_counter()
    size_rule = OneFeatureRateBaseline(column="SIZE").fit(X_train, y_train)
    t_size = time.perf_counter() - t0
    p_size = size_rule.predict_proba(X_val)
    rows.append(evaluate_predictions(y_val, p_size, "Baseline: One-Feature (SIZE)", train_seconds=t_size))
    val_probs["Baseline: One-Feature (SIZE)"] = (np.asarray(y_val, dtype=int), p_size)

    metrics_df = pd.DataFrame(rows)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(RESULTS_DIR / "baselines_val.csv", index=False)
        size_rule.table_.to_csv(RESULTS_DIR / "baseline_size_rates.csv")
    return metrics_df, val_probs, size_rule.table_


def main():
    metrics_df, _, size_table = run_baselines("main", save=True)
    print("=== One-Feature SIZE Rule (learned on train, 2014-2022) ===")
    print(size_table.round(4).to_string())
    print("\n=== Validation Metrics (2023-2024) ===")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR / 'baselines_val.csv'}")


if __name__ == "__main__":
    main()
