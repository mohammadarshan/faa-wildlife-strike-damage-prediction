"""Phase 4 Optuna hyperparameter tuning setup for gradient-boosting models.

Run from the repo root:
    .venv/bin/python -m src.boosting_tuning --quick
    .venv/bin/python -m src.boosting_tuning --n-trials 30
    .venv/bin/python -m src.boosting_tuning --models xgb lgb --n-trials 40

All tuning fits on TRAIN (2014-2022) and optimizes validation PR-AUC on
VALIDATION (2023-2024). Test (2025) is never touched here.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from src.boosting import _prepare_catboost_native
from src.data_loading import ROOT
from src.evaluation import (
    evaluate_predictions,
    plot_calibration_curves,
    plot_pr_curves,
    sanitize_feature_names,
)
from src.preprocessing import make_matrices

RESULTS_DIR = ROOT / "reports" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
SEED = 42


def _make_xgb_model(params: dict, scale_pos: float, n_estimators: int, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        gamma=params["gamma"],
        scale_pos_weight=scale_pos,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
    )


def _make_lgb_model(params: dict, balanced: bool, n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        subsample_freq=1,
        colsample_bytree=params["colsample_bytree"],
        min_child_samples=params["min_child_samples"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        class_weight="balanced" if balanced else None,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _make_catboost_model(params: dict, balanced: bool, iterations: int, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_strength=params["random_strength"],
        bagging_temperature=params["bagging_temperature"],
        border_count=params["border_count"],
        eval_metric="PRAUC",
        auto_class_weights="Balanced" if balanced else None,
        early_stopping_rounds=50,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def tune_xgboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    n_trials: int = 30,
    timeout: int | None = None,
    balanced: bool = False,
    max_estimators: int = 1500,
    seed: int = SEED,
):
    """Tune XGBoost on validation PR-AUC with Optuna."""
    scale_pos = float((y_tr == 0).sum() / (y_tr == 1).sum()) if balanced else 1.0

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        model = _make_xgb_model(params, scale_pos=scale_pos, n_estimators=max_estimators, seed=seed)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        val_prob = model.predict_proba(X_va)[:, 1]
        return float(average_precision_score(y_va, val_prob))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="xgb_pr_auc",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_model = _make_xgb_model(best_params, scale_pos=scale_pos, n_estimators=max_estimators, seed=seed)
    t0 = time.perf_counter()
    best_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elapsed = time.perf_counter() - t0
    val_prob = best_model.predict_proba(X_va)[:, 1]
    return best_model, val_prob, elapsed, study


def tune_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    n_trials: int = 30,
    timeout: int | None = None,
    balanced: bool = False,
    max_estimators: int = 1500,
    seed: int = SEED,
):
    """Tune LightGBM on validation PR-AUC with Optuna."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = _make_lgb_model(params, balanced=balanced, n_estimators=max_estimators, seed=seed)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        val_prob = model.predict_proba(X_va)[:, 1]
        return float(average_precision_score(y_va, val_prob))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="lgb_pr_auc",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_model = _make_lgb_model(best_params, balanced=balanced, n_estimators=max_estimators, seed=seed)
    t0 = time.perf_counter()
    best_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    elapsed = time.perf_counter() - t0
    val_prob = best_model.predict_proba(X_va)[:, 1]
    return best_model, val_prob, elapsed, study


def tune_catboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_features: list[str],
    n_trials: int = 20,
    timeout: int | None = None,
    balanced: bool = False,
    max_iterations: int = 1500,
    seed: int = SEED,
):
    """Tune CatBoost (native categoricals) on validation PR-AUC with Optuna."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 254),
        }
        model = _make_catboost_model(params, balanced=balanced, iterations=max_iterations, seed=seed)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_features,
            eval_set=(X_va, y_va),
            use_best_model=True,
        )
        trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
        val_prob = model.predict_proba(X_va)[:, 1]
        return float(average_precision_score(y_va, val_prob))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="catboost_pr_auc",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_model = _make_catboost_model(best_params, balanced=balanced, iterations=max_iterations, seed=seed)
    t0 = time.perf_counter()
    best_model.fit(
        X_tr,
        y_tr,
        cat_features=cat_features,
        eval_set=(X_va, y_va),
        use_best_model=True,
    )
    elapsed = time.perf_counter() - t0
    val_prob = best_model.predict_proba(X_va)[:, 1]
    return best_model, val_prob, elapsed, study


def run_tuning(
    models: list[str],
    scheme: str = "main",
    n_trials: int = 30,
    timeout: int | None = None,
    balanced: bool = False,
    quick: bool = False,
    seed: int = SEED,
    save: bool = True,
):
    """Run Optuna tuning for selected boosting models and save summary artifacts."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    max_trees = 200 if quick else 1500
    trials_per_model = min(n_trials, 3) if quick else n_trials

    rows = []
    pr_curves = {}
    cal_curves = {}
    best_configs = {}
    y_va_ref = None

    if "xgb" in models or "lgb" in models:
        mat_mi = make_matrices(scheme, add_missing_indicators=True, scale=False, seed=seed)
        X_tr = sanitize_feature_names(mat_mi["train"][0])
        y_tr = np.asarray(mat_mi["train"][1], dtype=int)
        X_va = sanitize_feature_names(mat_mi["validation"][0])
        y_va = np.asarray(mat_mi["validation"][1], dtype=int)
        y_va_ref = y_va

        if "xgb" in models:
            xgb_m, xgb_p, xgb_t, xgb_study = tune_xgboost(
                X_tr, y_tr, X_va, y_va,
                n_trials=trials_per_model,
                timeout=timeout,
                balanced=balanced,
                max_estimators=max_trees,
                seed=seed,
            )
            label = "XGBoost (tuned)"
            rows.append(evaluate_predictions(y_va, xgb_p, label, train_seconds=xgb_t))
            pr_curves[label] = (y_va, xgb_p)
            cal_curves[label] = (y_va, xgb_p)
            best_configs["xgb"] = {
                "best_val_pr_auc": float(xgb_study.best_value),
                "best_iteration": int(xgb_m.best_iteration),
                "n_trials": len(xgb_study.trials),
                "params": dict(xgb_study.best_params),
            }
            if save:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                suffix = "_balanced" if balanced else ""
                xgb_study.trials_dataframe().to_csv(
                    RESULTS_DIR / f"optuna_trials_xgb{suffix}.csv", index=False
                )

        if "lgb" in models:
            lgb_m, lgb_p, lgb_t, lgb_study = tune_lightgbm(
                X_tr, y_tr, X_va, y_va,
                n_trials=trials_per_model,
                timeout=timeout,
                balanced=balanced,
                max_estimators=max_trees,
                seed=seed,
            )
            label = "LightGBM (tuned)"
            rows.append(evaluate_predictions(y_va, lgb_p, label, train_seconds=lgb_t))
            pr_curves[label] = (y_va, lgb_p)
            cal_curves[label] = (y_va, lgb_p)
            best_configs["lgb"] = {
                "best_val_pr_auc": float(lgb_study.best_value),
                "best_iteration": int(lgb_m.best_iteration_),
                "n_trials": len(lgb_study.trials),
                "params": dict(lgb_study.best_params),
            }
            if save:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                suffix = "_balanced" if balanced else ""
                lgb_study.trials_dataframe().to_csv(
                    RESULTS_DIR / f"optuna_trials_lgb{suffix}.csv", index=False
                )

    if "catboost" in models:
        (X_tr_nat, y_tr_nat), (X_va_nat, y_va_nat), cat_cols = _prepare_catboost_native(scheme)
        if y_va_ref is None:
            y_va_ref = y_va_nat
        cb_m, cb_p, cb_t, cb_study = tune_catboost(
            X_tr_nat, y_tr_nat, X_va_nat, y_va_nat,
            cat_features=cat_cols,
            n_trials=trials_per_model,
            timeout=timeout,
            balanced=balanced,
            max_iterations=max_trees,
            seed=seed,
        )
        label = "CatBoost native (tuned)"
        rows.append(evaluate_predictions(y_va_nat, cb_p, label, train_seconds=cb_t))
        pr_curves[label] = (y_va_nat, cb_p)
        cal_curves[label] = (y_va_nat, cb_p)
        best_configs["catboost"] = {
            "best_val_pr_auc": float(cb_study.best_value),
            "best_iteration": int(cb_m.get_best_iteration()),
            "n_trials": len(cb_study.trials),
            "params": dict(cb_study.best_params),
        }
        if save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            suffix = "_balanced" if balanced else ""
            cb_study.trials_dataframe().to_csv(
                RESULTS_DIR / f"optuna_trials_catboost{suffix}.csv", index=False
            )

    summary_df = pd.DataFrame(rows)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_balanced" if balanced else ""
        summary_df.to_csv(RESULTS_DIR / f"boosting_tuning_summary{suffix}.csv", index=False)
        payload = {
            "scheme": scheme,
            "balanced": balanced,
            "quick": quick,
            "seed": seed,
            "models": best_configs,
        }
        Path(RESULTS_DIR / f"boosting_tuning_best_params{suffix}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        if pr_curves and y_va_ref is not None:
            plot_pr_curves(
                pr_curves,
                base_rate=float(y_va_ref.mean()),
                path=FIGURES_DIR / f"val_pr_curves_tuned{suffix}.png",
                title="Validation PR Curves — Tuned Boosting Models (2023-2024)",
            )
            plot_calibration_curves(
                cal_curves,
                path=FIGURES_DIR / f"val_calibration_tuned{suffix}.png",
                title="Validation Calibration — Tuned Boosting Models (2023-2024)",
            )

    return summary_df, best_configs


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Optuna tuning setup for gradient-boosting models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgb", "lgb", "catboost"],
        choices=["xgb", "lgb", "catboost"],
        help="Which boosting models to tune.",
    )
    parser.add_argument("--scheme", default="main", choices=["main", "all_years", "random"])
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per model.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional per-model timeout in seconds.")
    parser.add_argument("--balanced", action="store_true", help="Enable balanced class weights.")
    parser.add_argument("--quick", action="store_true", help="Smoke-test mode (3 trials, 200 max trees).")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    summary_df, best_configs = run_tuning(
        models=args.models,
        scheme=args.scheme,
        n_trials=args.n_trials,
        timeout=args.timeout,
        balanced=args.balanced,
        quick=args.quick,
        seed=args.seed,
        save=True,
    )

    print(f"=== Tuned Boosting Validation Summary ({args.scheme}, balanced={args.balanced}, quick={args.quick}) ===")
    cols = [
        "model", "pr_auc", "roc_auc", "brier",
        "recall_at_p20", "recall_at_p30", "recall_at_p50",
        "f1_at_05", "f1_best", "train_seconds",
    ]
    print(summary_df[cols].round(4).to_string(index=False))

    print("\n=== Best Hyperparameters ===")
    for name, cfg in best_configs.items():
        print(f"- {name}: PR-AUC={cfg['best_val_pr_auc']:.4f}, best_iter={cfg['best_iteration']}, params={cfg['params']}")

    print(f"\nSaved tuning artifacts to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
