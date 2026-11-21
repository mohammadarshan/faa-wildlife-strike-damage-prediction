"""Phase 3 first gradient-boosting runs: XGBoost, LightGBM, and CatBoost.

Run from the repo root:
    python -m src.boosting
    python -m src.boosting --balanced    # optional class-weighted comparison

All models are trained on TRAIN (2014-2022) and evaluated on VALIDATION (2023-2024).
Test (2025) is never touched here.

Models included:
1. XGBoost (encoded matrix, add_missing_indicators=True)
2. LightGBM (encoded matrix, add_missing_indicators=True)
3. CatBoost (encoded matrix, add_missing_indicators=True)
4. CatBoost (native categoricals on LOW_CARD + HIGH_CARD)
5. Missing-indicator ablation: XGBoost & LightGBM with add_missing_indicators=False
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

from src.baselines import run_baselines
from src.data_loading import ROOT
from src.evaluation import (
    evaluate_predictions,
    plot_calibration_curves,
    plot_pr_curves,
    sanitize_feature_names,
)
from src.preprocessing import BINARY, HIGH_CARD, LOW_CARD, NUMERIC, add_features, make_matrices
from src.splits import load_split_data

RESULTS_DIR = ROOT / "reports" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
SEED = 42


def _prepare_catboost_native(scheme: str = "main"):
    """Build train/validation frames for CatBoost native categorical handling."""
    splits = load_split_data(scheme)
    cat_cols = LOW_CARD + HIGH_CARD
    feature_cols = NUMERIC + cat_cols + BINARY
    prepared = {}
    for name in ("train", "validation"):
        X_raw, y = splits[name]
        X = add_features(X_raw)[feature_cols].copy()
        for col in cat_cols:
            X[col] = X[col].astype(object).where(X[col].notna(), "(blank)").astype(str)
        prepared[name] = (X, np.asarray(y, dtype=int))
    return prepared["train"], prepared["validation"], cat_cols


def _top_features(names, importances, model_name: str, top_k: int = 15) -> pd.DataFrame:
    order = np.argsort(importances)[::-1][:top_k]
    return pd.DataFrame(
        {
            "model": model_name,
            "rank": np.arange(1, len(order) + 1),
            "feature": [names[i] for i in order],
            "importance": [float(importances[i]) for i in order],
        }
    )


def fit_xgboost(X_train, y_train, X_val, y_val, balanced: bool = False):
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum()) if balanced else 1.0
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.perf_counter() - t0
    val_prob = model.predict_proba(X_val)[:, 1]
    return model, val_prob, elapsed


def fit_lightgbm(X_train, y_train, X_val, y_val, balanced: bool = False):
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced" if balanced else None,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    elapsed = time.perf_counter() - t0
    val_prob = model.predict_proba(X_val)[:, 1]
    return model, val_prob, elapsed


def fit_catboost(X_train, y_train, X_val, y_val, cat_features=None, balanced: bool = False):
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        eval_metric="PRAUC",
        auto_class_weights="Balanced" if balanced else None,
        early_stopping_rounds=50,
        random_seed=SEED,
        verbose=False,
        thread_count=-1,
    )
    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    elapsed = time.perf_counter() - t0
    val_prob = model.predict_proba(X_val)[:, 1]
    return model, val_prob, elapsed


def run_boosting(scheme: str = "main", balanced: bool = False, save: bool = True):
    """Train first gradient-boosting models on train and evaluate on validation."""
    baseline_df, baseline_curves, _ = run_baselines(scheme, save=save)

    # 1. Preprocessed matrices WITH missing indicators (primary representation)
    mat_mi = make_matrices(scheme, add_missing_indicators=True, scale=False, seed=SEED)
    X_tr = sanitize_feature_names(mat_mi["train"][0])
    y_tr = np.asarray(mat_mi["train"][1], dtype=int)
    X_va = sanitize_feature_names(mat_mi["validation"][0])
    y_va = np.asarray(mat_mi["validation"][1], dtype=int)

    rows = []
    # Majority baseline is shown by the horizontal no-skill prevalence line in plot_pr_curves
    pr_curves = {k: v for k, v in baseline_curves.items() if "Majority" not in k}
    cal_curves = {}
    fi_tables = []

    xgb_m, xgb_p, xgb_t = fit_xgboost(X_tr, y_tr, X_va, y_va, balanced=balanced)
    rows.append(evaluate_predictions(y_va, xgb_p, "XGBoost (encoded)", train_seconds=xgb_t))
    pr_curves["XGBoost (encoded)"] = (y_va, xgb_p)
    cal_curves["XGBoost (encoded)"] = (y_va, xgb_p)
    fi_tables.append(_top_features(list(X_tr.columns), xgb_m.feature_importances_, "XGBoost (encoded)"))

    lgb_m, lgb_p, lgb_t = fit_lightgbm(X_tr, y_tr, X_va, y_va, balanced=balanced)
    rows.append(evaluate_predictions(y_va, lgb_p, "LightGBM (encoded)", train_seconds=lgb_t))
    pr_curves["LightGBM (encoded)"] = (y_va, lgb_p)
    cal_curves["LightGBM (encoded)"] = (y_va, lgb_p)
    fi_tables.append(_top_features(list(X_tr.columns), lgb_m.feature_importances_, "LightGBM (encoded)"))

    cb_enc_m, cb_enc_p, cb_enc_t = fit_catboost(X_tr, y_tr, X_va, y_va, balanced=balanced)
    rows.append(evaluate_predictions(y_va, cb_enc_p, "CatBoost (encoded)", train_seconds=cb_enc_t))
    pr_curves["CatBoost (encoded)"] = (y_va, cb_enc_p)
    cal_curves["CatBoost (encoded)"] = (y_va, cb_enc_p)
    fi_tables.append(_top_features(list(X_tr.columns), cb_enc_m.get_feature_importance(), "CatBoost (encoded)"))

    # 2. CatBoost with native high-cardinality categorical handling
    (X_tr_nat, y_tr_nat), (X_va_nat, y_va_nat), cat_cols = _prepare_catboost_native(scheme)
    cb_nat_m, cb_nat_p, cb_nat_t = fit_catboost(
        X_tr_nat, y_tr_nat, X_va_nat, y_va_nat, cat_features=cat_cols, balanced=balanced
    )
    rows.append(evaluate_predictions(y_va_nat, cb_nat_p, "CatBoost (native cat)", train_seconds=cb_nat_t))
    pr_curves["CatBoost (native cat)"] = (y_va_nat, cb_nat_p)
    cal_curves["CatBoost (native cat)"] = (y_va_nat, cb_nat_p)
    fi_tables.append(_top_features(list(X_tr_nat.columns), cb_nat_m.get_feature_importance(), "CatBoost (native cat)"))

    # 3. Missing-indicator ablation (add_missing_indicators=False)
    mat_no_mi = make_matrices(scheme, add_missing_indicators=False, scale=False, seed=SEED)
    X_tr_nomi = sanitize_feature_names(mat_no_mi["train"][0])
    y_tr_nomi = np.asarray(mat_no_mi["train"][1], dtype=int)
    X_va_nomi = sanitize_feature_names(mat_no_mi["validation"][0])
    y_va_nomi = np.asarray(mat_no_mi["validation"][1], dtype=int)

    _, xgb_nomi_p, xgb_nomi_t = fit_xgboost(X_tr_nomi, y_tr_nomi, X_va_nomi, y_va_nomi, balanced=balanced)
    rows.append(evaluate_predictions(y_va_nomi, xgb_nomi_p, "XGBoost (no missing ind.)", train_seconds=xgb_nomi_t))

    _, lgb_nomi_p, lgb_nomi_t = fit_lightgbm(X_tr_nomi, y_tr_nomi, X_va_nomi, y_va_nomi, balanced=balanced)
    rows.append(evaluate_predictions(y_va_nomi, lgb_nomi_p, "LightGBM (no missing ind.)", train_seconds=lgb_nomi_t))

    boosting_df = pd.DataFrame(rows)
    all_df = pd.concat([baseline_df, boosting_df], ignore_index=True)
    fi_df = pd.concat(fi_tables, ignore_index=True)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_balanced" if balanced else ""
        all_df.to_csv(RESULTS_DIR / f"boosting_val{suffix}.csv", index=False)
        fi_df.to_csv(RESULTS_DIR / f"boosting_top_features{suffix}.csv", index=False)
        plot_pr_curves(
            pr_curves,
            base_rate=float(y_va.mean()),
            path=FIGURES_DIR / f"val_pr_curves_checkpoint{suffix}.png",
        )
        plot_calibration_curves(
            cal_curves,
            path=FIGURES_DIR / f"val_calibration_checkpoint{suffix}.png",
        )

    return all_df, fi_df


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 first gradient-boosting models on validation.")
    parser.add_argument("--scheme", default="main", choices=["main", "all_years", "random"])
    parser.add_argument("--balanced", action="store_true", help="Enable balanced class weights.")
    args = parser.parse_args()

    all_df, fi_df = run_boosting(scheme=args.scheme, balanced=args.balanced, save=True)
    print(f"=== Validation Comparison ({args.scheme}, balanced={args.balanced}) ===")
    cols = [
        "model", "pr_auc", "roc_auc", "brier",
        "recall_at_p20", "recall_at_p30", "recall_at_p50",
        "f1_at_05", "f1_best", "train_seconds",
    ]
    print(all_df[cols].round(4).to_string(index=False))

    print("\n=== Top 10 Features per Boosting Model ===")
    for model_name, grp in fi_df.groupby("model", sort=False):
        top10 = grp.head(10)
        names = ", ".join(f"{r.feature} ({r.importance:.2f})" for r in top10.itertuples())
        print(f"- {model_name}: {names}")

    print(f"\nSaved results to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
