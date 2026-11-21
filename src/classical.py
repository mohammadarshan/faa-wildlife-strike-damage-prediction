"""Phase 3 classical models: elastic-net logistic regression and random forest.

Run from the repo root:
    python -m src.classical                  # both models + missing-indicator ablation
    python -m src.classical --quick          # 100-tree forest, to check that everything runs
    python -m src.classical --skip-ablation  # skip the 'no missing indicators' refits
    python -m src.classical --tune-lr        # pick C for the logistic regression on validation
    python -m src.classical --balanced       # optional class-weighted comparison

Both models are trained on TRAIN (2014-2022) and evaluated on VALIDATION (2023-2024), with the same
metrics and preprocessing as the baselines and boosting runs. Test (2025) is never touched here.
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.baselines import run_baselines
from src.data_loading import ROOT
from src.evaluation import evaluate_predictions, plot_calibration_curves, plot_pr_curves
from src.preprocessing import make_matrices

RESULTS_DIR = ROOT / "reports" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
SEED = 42
C_GRID = [0.01, 0.1, 1.0]


def _sklearn_version() -> tuple:
    major, minor = sklearn.__version__.split(".")[:2]
    return int(major), int(minor)


def make_logreg(C: float, balanced: bool = False) -> LogisticRegression:
    """Elastic-net logistic regression (saga). scikit-learn 1.8+ takes l1_ratio alone;
    older versions need penalty='elasticnet' as well."""
    kwargs = dict(solver="saga", C=C, l1_ratio=0.5, max_iter=200, tol=1e-3, random_state=SEED,
                  class_weight="balanced" if balanced else None)
    if _sklearn_version() < (1, 8):
        kwargs["penalty"] = "elasticnet"
    return LogisticRegression(**kwargs)


def fit_logreg(X_train, y_train, X_val, C: float = 0.1, balanced: bool = False):
    model = make_logreg(C, balanced)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        print(f"  note: logistic regression (C={C}) hit max_iter without fully converging")
    return model, model.predict_proba(X_val)[:, 1], elapsed


def fit_random_forest(X_train, y_train, X_val, n_trees: int = 300, balanced: bool = False):
    model = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample" if balanced else None,
        n_jobs=-1,
        random_state=SEED,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    return model, model.predict_proba(X_val)[:, 1], elapsed


def _top_features(names, importances, model_name: str, top_k: int = 15) -> pd.DataFrame:
    order = np.argsort(importances)[::-1][:top_k]
    return pd.DataFrame({
        "model": model_name,
        "rank": np.arange(1, len(order) + 1),
        "feature": [names[i] for i in order],
        "importance": [float(importances[i]) for i in order],
    })


def markdown_rows(df: pd.DataFrame, owner: str) -> str:
    """Rows in the same column order as the checkpoint results table."""
    lines = []
    for r in df.itertuples():
        lines.append(
            f"| {r.model} | {owner} | {r.pr_auc:.4f} | {r.roc_auc:.4f} | {r.brier:.4f} | "
            f"{r.recall_at_p20:.4f} | {r.recall_at_p30:.4f} | {r.recall_at_p50:.4f} | "
            f"{r.f1_at_05:.4f} | {r.f1_best:.4f} | {r.train_seconds:.2f} |"
        )
    return "\n".join(lines)


def run_classical(scheme="main", balanced=False, save=True, ablation=True, tune_lr=False, n_trees=300):
    baseline_df, baseline_curves, _ = run_baselines(scheme, save=save)

    # Scaled matrix for both models (a forest is unaffected by per-column scaling)
    mat = make_matrices(scheme, add_missing_indicators=True, scale=True, seed=SEED)
    X_tr, y_tr = mat["train"][0], np.asarray(mat["train"][1], dtype=int)
    X_va, y_va = mat["validation"][0], np.asarray(mat["validation"][1], dtype=int)

    pr_curves = {k: v for k, v in baseline_curves.items() if "Majority" not in k}
    cal_curves, fi_tables, rows = {}, [], []

    # ---- Logistic regression (optionally choose C on validation) ----
    best_C = 0.1
    if tune_lr:
        scores = {}
        for C in C_GRID:
            print(f"Logistic regression, C={C} ...")
            _, p, _ = fit_logreg(X_tr, y_tr, X_va, C=C, balanced=balanced)
            scores[C] = float(evaluate_predictions(y_va, p, "tmp")["pr_auc"])
            print(f"  validation PR-AUC {scores[C]:.4f}")
        best_C = max(scores, key=scores.get)
        print(f"Chosen C = {best_C}")
    print(f"Fitting logistic regression (elastic-net, C={best_C}) ...")
    lr_m, lr_p, lr_t = fit_logreg(X_tr, y_tr, X_va, C=best_C, balanced=balanced)
    rows.append(evaluate_predictions(y_va, lr_p, "Logistic Regression (Elastic-Net)", train_seconds=lr_t))
    pr_curves["Logistic Regression"] = (y_va, lr_p)
    cal_curves["Logistic Regression"] = (y_va, lr_p)
    fi_tables.append(_top_features(list(X_tr.columns), np.abs(lr_m.coef_[0]), "Logistic Regression"))
    n_zero = int((lr_m.coef_[0] == 0).sum())
    print(f"  {n_zero} of {X_tr.shape[1]} coefficients shrunk to exactly zero by the L1 part")

    # ---- Random forest ----
    print(f"Fitting random forest ({n_trees} trees) ...")
    rf_m, rf_p, rf_t = fit_random_forest(X_tr, y_tr, X_va, n_trees=n_trees, balanced=balanced)
    rows.append(evaluate_predictions(y_va, rf_p, "Random Forest", train_seconds=rf_t))
    pr_curves["Random Forest"] = (y_va, rf_p)
    cal_curves["Random Forest"] = (y_va, rf_p)
    fi_tables.append(_top_features(list(X_tr.columns), rf_m.feature_importances_, "Random Forest"))

    # ---- Missing-indicator ablation ----
    if ablation:
        print("Ablation: refitting without missing indicators ...")
        m2 = make_matrices(scheme, add_missing_indicators=False, scale=True, seed=SEED)
        X2_tr, y2_tr = m2["train"][0], np.asarray(m2["train"][1], dtype=int)
        X2_va, y2_va = m2["validation"][0], np.asarray(m2["validation"][1], dtype=int)
        _, p, t = fit_logreg(X2_tr, y2_tr, X2_va, C=best_C, balanced=balanced)
        rows.append(evaluate_predictions(y2_va, p, "Logistic Regression (no missing ind.)", train_seconds=t))
        _, p, t = fit_random_forest(X2_tr, y2_tr, X2_va, n_trees=n_trees, balanced=balanced)
        rows.append(evaluate_predictions(y2_va, p, "Random Forest (no missing ind.)", train_seconds=t))

    classical_df = pd.DataFrame(rows)
    all_df = pd.concat([baseline_df, classical_df], ignore_index=True)
    fi_df = pd.concat(fi_tables, ignore_index=True)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_balanced" if balanced else ""
        all_df.to_csv(RESULTS_DIR / f"classical_val{suffix}.csv", index=False)
        fi_df.to_csv(RESULTS_DIR / f"classical_top_features{suffix}.csv", index=False)
        plot_pr_curves(pr_curves, base_rate=float(y_va.mean()),
                       path=FIGURES_DIR / f"val_pr_curves_classical{suffix}.png")
        plot_calibration_curves(cal_curves, path=FIGURES_DIR / f"val_calibration_classical{suffix}.png")
        # One table with everything, if the boosting results are already there
        boost_path = RESULTS_DIR / f"boosting_val{suffix}.csv"
        if boost_path.exists():
            boost = pd.read_csv(boost_path)
            boost = boost[~boost["model"].str.startswith("Baseline")]
            pd.concat([baseline_df, classical_df, boost], ignore_index=True).to_csv(
                RESULTS_DIR / f"val_comparison_all{suffix}.csv", index=False)
    return classical_df, all_df, fi_df


def main():
    parser = argparse.ArgumentParser(description="Phase 3 classical models on validation.")
    parser.add_argument("--scheme", default="main", choices=["main", "all_years", "random"])
    parser.add_argument("--balanced", action="store_true", help="Enable balanced class weights.")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--tune-lr", action="store_true")
    parser.add_argument("--quick", action="store_true", help="100 trees instead of 300")
    args = parser.parse_args()

    classical_df, all_df, fi_df = run_classical(
        scheme=args.scheme, balanced=args.balanced, ablation=not args.skip_ablation,
        tune_lr=args.tune_lr, n_trees=100 if args.quick else 300)

    print(f"\n=== Validation Comparison ({args.scheme}, balanced={args.balanced}) ===")
    cols = ["model", "pr_auc", "roc_auc", "brier", "recall_at_p20", "recall_at_p30", "recall_at_p50",
            "f1_at_05", "f1_best", "train_seconds"]
    print(all_df[cols].round(4).to_string(index=False))

    print("\n=== Top 10 Features per Model (LR: |standardized coefficient|, RF: impurity importance) ===")
    for model_name, grp in fi_df.groupby("model", sort=False):
        names = ", ".join(f"{r.feature} ({r.importance:.2f})" for r in grp.head(10).itertuples())
        print(f"- {model_name}: {names}")

    print("\n=== Rows for the checkpoint results table ===")
    print(markdown_rows(classical_df, "Arshan"))
    print(f"\nSaved results to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()