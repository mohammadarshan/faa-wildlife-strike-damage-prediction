"""Phase 4 small hyperparameter search for logistic regression and the random forest.

Run from the repo root:
    python -m src.classical_tuning --quick      # 2 configs per model, 100 trees: checks that everything runs
    python -m src.classical_tuning              # 9 logistic-regression configs and 6 random-forest configs
    python -m src.classical_tuning --skip-lr    # random forest only (or --skip-rf for logistic regression only)

Same protocol as the boosting and neural-network tuning: fit on TRAIN (2014-2022), score validation PR-AUC
(2023-2024), keep the best configuration. Test (2025) is never touched here. The default settings from
src/classical.py (C = 0.1, l1_ratio = 0.5; 300 trees, min leaf 10, sqrt features) are inside each grid, so the
table shows directly what the search gained.
"""
import argparse
import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.classical import _sklearn_version, markdown_rows
from src.data_loading import ROOT
from src.evaluation import evaluate_predictions, plot_calibration_curves, plot_pr_curves
from src.preprocessing import make_matrices

RESULTS_DIR = ROOT / "reports" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
SEED = 42

LR_GRID = list(itertools.product([0.1, 0.03, 0.3], [0.5, 0.1, 0.9]))          # (C, l1_ratio); default first
RF_GRID = list(itertools.product([10, 3, 30], ["sqrt", 0.2]))                  # (min_samples_leaf, max_features)


def fit_lr(X_tr, y_tr, X_va, C, l1_ratio):
    kwargs = dict(solver="saga", C=C, l1_ratio=l1_ratio, max_iter=200, tol=1e-3, random_state=SEED)
    if _sklearn_version() < (1, 8):
        kwargs["penalty"] = "elasticnet"
    model = LogisticRegression(**kwargs)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, model.predict_proba(X_va)[:, 1], elapsed, converged


def fit_rf(X_tr, y_tr, X_va, min_leaf, max_features, n_trees):
    model = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=min_leaf, max_features=max_features,
                                   n_jobs=-1, random_state=SEED)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_va)[:, 1], time.perf_counter() - t0


def run_search(scheme="main", quick=False, do_lr=True, do_rf=True, save=True):
    mat = make_matrices(scheme, add_missing_indicators=True, scale=True, seed=SEED)
    X_tr, y_tr = mat["train"][0], np.asarray(mat["train"][1], dtype=int)
    X_va, y_va = mat["validation"][0], np.asarray(mat["validation"][1], dtype=int)
    n_trees = 100 if quick else 300

    grid_rows, final_rows, curves, best_params = [], [], {}, {}

    if do_lr:
        results = []
        for C, l1 in (LR_GRID[:2] if quick else LR_GRID):
            print(f"Logistic regression: C={C}, l1_ratio={l1} ...")
            _, p, t, ok = fit_lr(X_tr, y_tr, X_va, C, l1)
            ap = float(average_precision_score(y_va, p))
            print(f"  validation PR-AUC {ap:.4f} ({t:.0f}s{'' if ok else ', did not fully converge'})")
            results.append((ap, C, l1, p, t))
            grid_rows.append(dict(model="Logistic Regression", setting=f"C={C}, l1_ratio={l1}", pr_auc=ap, seconds=round(t, 1)))
        ap, C, l1, p, t = max(results, key=lambda r: r[0])
        final_rows.append(evaluate_predictions(y_va, p, "Logistic Regression (tuned)", train_seconds=t))
        curves["Logistic Regression (tuned)"] = (y_va, p)
        best_params["logistic_regression"] = dict(C=C, l1_ratio=l1, val_pr_auc=ap, n_configs=len(results))

    if do_rf:
        results = []
        for leaf, mf in (RF_GRID[:2] if quick else RF_GRID):
            print(f"Random forest: min_samples_leaf={leaf}, max_features={mf} ...")
            _, p, t = fit_rf(X_tr, y_tr, X_va, leaf, mf, n_trees)
            ap = float(average_precision_score(y_va, p))
            print(f"  validation PR-AUC {ap:.4f} ({t:.0f}s)")
            results.append((ap, leaf, mf, p, t))
            grid_rows.append(dict(model="Random Forest", setting=f"min_samples_leaf={leaf}, max_features={mf}", pr_auc=ap, seconds=round(t, 1)))
        ap, leaf, mf, p, t = max(results, key=lambda r: r[0])
        final_rows.append(evaluate_predictions(y_va, p, "Random Forest (tuned)", train_seconds=t))
        curves["Random Forest (tuned)"] = (y_va, p)
        best_params["random_forest"] = dict(min_samples_leaf=leaf, max_features=mf, n_estimators=n_trees, val_pr_auc=ap, n_configs=len(results))

    grid_df, tuned_df = pd.DataFrame(grid_rows), pd.DataFrame(final_rows)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        grid_df.to_csv(RESULTS_DIR / "classical_tuning_grid.csv", index=False)
        tuned_df.to_csv(RESULTS_DIR / "classical_tuned_val.csv", index=False)
        (RESULTS_DIR / "classical_tuning_best_params.json").write_text(json.dumps(best_params, indent=2) + "\n")
        plot_pr_curves(curves, base_rate=float(y_va.mean()), path=FIGURES_DIR / "val_pr_curves_classical_tuned.png",
                       title="Validation PR Curves: Tuned Logistic Regression and Random Forest (2023-2024)")
        plot_calibration_curves(curves, path=FIGURES_DIR / "val_calibration_classical_tuned.png",
                                title="Validation Calibration: Tuned Logistic Regression and Random Forest (2023-2024)")
        # One table with every model, if the earlier result files exist
        parts = []
        for name in ["val_comparison_all_with_nn.csv", "val_comparison_all.csv"]:
            if (RESULTS_DIR / name).exists():
                parts.append(pd.read_csv(RESULTS_DIR / name))
                break
        if (RESULTS_DIR / "boosting_tuning_summary.csv").exists():
            parts.append(pd.read_csv(RESULTS_DIR / "boosting_tuning_summary.csv"))
        if parts:
            pd.concat(parts + [tuned_df], ignore_index=True).drop_duplicates("model", keep="last").to_csv(
                RESULTS_DIR / "val_comparison_all_final.csv", index=False)
    return grid_df, tuned_df, best_params


def main():
    parser = argparse.ArgumentParser(description="Small validation-scored search for logistic regression and random forest.")
    parser.add_argument("--scheme", default="main", choices=["main", "all_years", "random"])
    parser.add_argument("--quick", action="store_true", help="2 configs per model and 100 trees: a fast check.")
    parser.add_argument("--skip-lr", action="store_true")
    parser.add_argument("--skip-rf", action="store_true")
    args = parser.parse_args()

    grid_df, tuned_df, best = run_search(args.scheme, args.quick, not args.skip_lr, not args.skip_rf)

    print(f"\n=== Search results, validation PR-AUC ({args.scheme}, quick={args.quick}) ===")
    for model, grp in grid_df.groupby("model", sort=False):
        print(f"\n{model}")
        print(grp.sort_values("pr_auc", ascending=False)[["setting", "pr_auc", "seconds"]].round(4).to_string(index=False))
    print("\n=== Best settings ===")
    for k, v in best.items():
        print(f"- {k}: {v}")
    print("\n=== Tuned models, validation ===")
    cols = ["model", "pr_auc", "roc_auc", "brier", "recall_at_p20", "recall_at_p30", "recall_at_p50",
            "f1_at_05", "f1_best", "train_seconds"]
    print(tuned_df[cols].round(4).to_string(index=False))
    print("\n=== Rows for the checkpoint results table ===")
    print(markdown_rows(tuned_df, "Arshan"))
    print(f"\nSaved results to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()