"""Phase 4 neural network: a feed-forward network (MLP) for tabular data, with embeddings for the
categorical columns (including the high-cardinality ones) and standardized numeric columns.

Run from the repo root:
    python -m src.neural --quick       # 3 epochs, one seed, no ablation: checks that everything runs
    python -m src.neural               # default config, 3 seeds, plus the 'no missing indicators' refit
    python -m src.neural --tune        # try 5 small configs on validation first, then use the best
    python -m src.neural --balanced    # optional class-weighted comparison (changes the probabilities)

Trained on TRAIN (2014-2022), early-stopped and compared on VALIDATION (2023-2024), the same protocol as
the boosting runs. Test (2025) is never touched here.
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from src.baselines import run_baselines
from src.data_loading import ROOT
from src.evaluation import evaluate_predictions, plot_calibration_curves, plot_pr_curves
from src.preprocessing import BINARY, HIGH_CARD, LOW_CARD, NUMERIC, add_features
from src.splits import load_split_data

RESULTS_DIR = ROOT / "reports" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
SEED = 42
CAT_COLS = LOW_CARD + HIGH_CARD
MIN_COUNT = 20  # categories with fewer training rows share one "rare" embedding

DEFAULT_CFG = dict(name="base", hidden=(256, 128), dropout=0.3, emb_dropout=0.1, lr=1e-3, wd=1e-5, batch=1024)
TUNE_CFGS = [
    DEFAULT_CFG,
    dict(DEFAULT_CFG, name="small", hidden=(128, 64), dropout=0.2),
    dict(DEFAULT_CFG, name="wide", hidden=(512, 256), dropout=0.4),
    dict(DEFAULT_CFG, name="slow", lr=3e-4),
    dict(DEFAULT_CFG, name="deep", hidden=(256, 128, 64)),
]


# ----------------------------------------------------------------------------------------------
# Encoding: everything that learns from data is fitted on TRAIN only
# ----------------------------------------------------------------------------------------------
class NNEncoder:
    """Categorical columns -> integer codes (0 = blank, 1 = rare or unseen, 2+ = known categories).
    Numeric columns -> median-filled, optional 'was blank' indicators, standardized.
    add_missing_indicators=False: blanks are filled with the median / most common category instead."""

    def __init__(self, add_missing_indicators: bool = True, min_count: int = MIN_COUNT):
        self.add_missing_indicators = add_missing_indicators
        self.min_count = min_count

    def fit(self, X: pd.DataFrame):
        self.medians_, self.means_, self.stds_, self.ind_cols_ = {}, {}, {}, []
        filled = {}
        for c in NUMERIC:
            col = pd.to_numeric(X[c], errors="coerce")
            med = float(col.median()) if col.notna().any() else 0.0
            self.medians_[c] = med
            if self.add_missing_indicators and col.isna().any():
                self.ind_cols_.append(c)
            filled[c] = col.fillna(med)
        for c in NUMERIC:
            self.means_[c] = float(filled[c].mean())
            sd = float(filled[c].std())
            self.stds_[c] = sd if sd > 0 else 1.0
        self.vocab_, self.blank_id_ = {}, {}
        for c in CAT_COLS:
            counts = X[c].dropna().astype(str).value_counts()
            keep = counts[counts >= self.min_count].index.tolist()
            self.vocab_[c] = {v: i + 2 for i, v in enumerate(keep)}
            mode = counts.index[0] if len(counts) else None
            self.blank_id_[c] = 0 if self.add_missing_indicators else self.vocab_[c].get(mode, 1)
        self.cardinalities_ = [len(self.vocab_[c]) + 2 for c in CAT_COLS]
        return self

    def transform(self, X: pd.DataFrame):
        num_cols = []
        for c in NUMERIC:
            col = pd.to_numeric(X[c], errors="coerce")
            num_cols.append(((col.fillna(self.medians_[c]) - self.means_[c]) / self.stds_[c]).to_numpy(dtype=np.float32))
        for c in self.ind_cols_:
            num_cols.append(pd.to_numeric(X[c], errors="coerce").isna().to_numpy(dtype=np.float32))
        for c in BINARY:
            num_cols.append(pd.to_numeric(X[c], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
        X_num = np.column_stack(num_cols)
        cat_cols = []
        for c in CAT_COLS:
            s = X[c].astype(object)
            s = s.where(s.isna(), s.astype(str))                  # non-blank values as text, like the vocabulary
            blank = s.isna().to_numpy()
            mapped = s.map(self.vocab_[c]).to_numpy(dtype=float)  # NaN for blank or unseen/rare values
            ids = np.where(np.isnan(mapped), 1, mapped).astype(np.int64)   # unseen or rare -> 1
            ids[blank] = self.blank_id_[c]                        # blank -> 0 (or the most common category)
            cat_cols.append(ids)
        X_cat = np.column_stack(cat_cols)
        return X_cat, X_num


def build_data(scheme: str = "main", add_missing_indicators: bool = True) -> dict:
    """Split with src.splits, build features, fit the encoder on TRAIN only, encode all three splits."""
    splits = load_split_data(scheme)
    feats = {name: (add_features(X), np.asarray(y, dtype=int)) for name, (X, y) in splits.items()}
    enc = NNEncoder(add_missing_indicators).fit(feats["train"][0])
    out = {}
    for name, (X, y) in feats.items():
        X_cat, X_num = enc.transform(X)
        out[name] = (X_cat, X_num, y)
    out["encoder"] = enc
    return out


# ----------------------------------------------------------------------------------------------
# Model and training
# ----------------------------------------------------------------------------------------------
def emb_dim(card: int) -> int:
    return int(min(32, max(2, round(1.6 * card ** 0.56))))


class TabularMLP(nn.Module):
    def __init__(self, cardinalities, n_num, hidden=(256, 128), dropout=0.3, emb_dropout=0.1):
        super().__init__()
        dims = [emb_dim(c) for c in cardinalities]
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, dims)])
        self.emb_drop = nn.Dropout(emb_dropout)
        layers, d_in = [], sum(dims) + n_num
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        return self.net(torch.cat([self.emb_drop(e), x_num], dim=1)).squeeze(1)


def predict(model, X_cat, X_num, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_cat), batch):
            xc = torch.as_tensor(X_cat[i:i + batch], dtype=torch.long)
            xn = torch.as_tensor(X_num[i:i + batch], dtype=torch.float32)
            out.append(torch.sigmoid(model(xc, xn)).numpy())
    return np.concatenate(out)


def train_one(data: dict, cfg: dict, seed: int, pos_weight=None, max_epochs: int = 40, patience: int = 5):
    """Train one network; keep the epoch with the best validation PR-AUC. Returns (model, val_prob, info)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xc_tr, Xn_tr, y_tr = data["train"]
    Xc_va, Xn_va, y_va = data["validation"]
    model = TabularMLP(data["encoder"].cardinalities_, Xn_tr.shape[1], cfg["hidden"], cfg["dropout"], cfg["emb_dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    pw = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    xc_t = torch.as_tensor(Xc_tr, dtype=torch.long)
    xn_t = torch.as_tensor(Xn_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.float32)
    n, bs = len(y_t), cfg["batch"]

    best_ap, best_state, best_epoch, since_best = -1.0, None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:          # BatchNorm needs at least 2 rows
                continue
            opt.zero_grad()
            loss = loss_fn(model(xc_t[idx], xn_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
        ap = float(average_precision_score(y_va, predict(model, Xc_va, Xn_va)))
        if ap > best_ap:
            best_ap, best_epoch, since_best = ap, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                break
    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return model, predict(model, Xc_va, Xn_va), dict(best_epoch=best_epoch, epochs_run=epoch, val_pr_auc=best_ap, seconds=elapsed)


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


def run_neural(scheme="main", balanced=False, tune=False, n_seeds=3, ablation=True, max_epochs=40, save=True):
    baseline_df, baseline_curves, _ = run_baselines(scheme, save=save)
    data = build_data(scheme, add_missing_indicators=True)
    y_tr, y_va = data["train"][2], data["validation"][2]
    n_feat = data["train"][1].shape[1]
    print(f"Training rows {len(y_tr):,} | {len(data['encoder'].cardinalities_)} embedded columns | {n_feat} numeric inputs")
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum()) if balanced else None

    cfg = DEFAULT_CFG
    if tune:
        scores = {}
        for c in TUNE_CFGS:
            print(f"Config '{c['name']}' ...")
            _, _, info = train_one(data, c, SEED, pos_weight, max_epochs)
            scores[c["name"]] = info["val_pr_auc"]
            print(f"  validation PR-AUC {info['val_pr_auc']:.4f} (best epoch {info['best_epoch']}, {info['seconds']:.0f}s)")
        best = max(scores, key=scores.get)
        cfg = next(c for c in TUNE_CFGS if c["name"] == best)
        print(f"Chosen config: {best}")

    rows, seed_rows, probs = [], [], {}
    for k in range(n_seeds):
        seed = SEED + k
        print(f"Training network, seed {seed} ({cfg['name']}) ...")
        _, p, info = train_one(data, cfg, seed, pos_weight, max_epochs)
        probs[seed] = p
        seed_rows.append(dict(seed=seed, **info, pr_auc=float(average_precision_score(y_va, p))))
        print(f"  validation PR-AUC {info['val_pr_auc']:.4f} at epoch {info['best_epoch']} of {info['epochs_run']} run")
        if k == 0:
            rows.append(evaluate_predictions(y_va, p, "Neural Network (MLP + embeddings)", train_seconds=info["seconds"]))
    seeds_df = pd.DataFrame(seed_rows)

    if ablation:
        print("Ablation: refitting without missing indicators ...")
        d2 = build_data(scheme, add_missing_indicators=False)
        _, p2, info2 = train_one(d2, cfg, SEED, pos_weight, max_epochs)
        rows.append(evaluate_predictions(d2["validation"][2], p2, "Neural Network (no missing ind.)", train_seconds=info2["seconds"]))

    nn_df = pd.DataFrame(rows)
    all_df = pd.concat([baseline_df, nn_df], ignore_index=True)
    first = probs[SEED]
    pr_curves = {k: v for k, v in baseline_curves.items() if "Majority" not in k}
    pr_curves["Neural Network"] = (y_va, first)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_balanced" if balanced else ""
        all_df.to_csv(RESULTS_DIR / f"neural_val{suffix}.csv", index=False)
        seeds_df.to_csv(RESULTS_DIR / f"neural_seeds{suffix}.csv", index=False)
        plot_pr_curves(pr_curves, base_rate=float(y_va.mean()), path=FIGURES_DIR / f"val_pr_curves_neural{suffix}.png")
        plot_calibration_curves({"Neural Network": (y_va, first)}, path=FIGURES_DIR / f"val_calibration_neural{suffix}.png")
        prev = RESULTS_DIR / f"val_comparison_all{suffix}.csv"
        if prev.exists():
            old = pd.read_csv(prev)
            old = old[~old["model"].str.startswith("Neural")]
            pd.concat([old, nn_df], ignore_index=True).to_csv(RESULTS_DIR / f"val_comparison_all_with_nn{suffix}.csv", index=False)
    return nn_df, all_df, seeds_df, cfg


def main():
    parser = argparse.ArgumentParser(description="Phase 4 neural network (MLP with embeddings) on validation.")
    parser.add_argument("--scheme", default="main", choices=["main", "all_years", "random"])
    parser.add_argument("--balanced", action="store_true", help="Class-weighted loss (changes the probabilities).")
    parser.add_argument("--tune", action="store_true", help="Try 5 small configs on validation first.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds to train.")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--quick", action="store_true", help="3 epochs, one seed, no ablation: a fast check.")
    args = parser.parse_args()

    if args.quick:
        nn_df, all_df, seeds_df, cfg = run_neural(args.scheme, args.balanced, False, 1, False, max_epochs=3)
    else:
        nn_df, all_df, seeds_df, cfg = run_neural(args.scheme, args.balanced, args.tune, args.seeds, not args.skip_ablation)

    print(f"\n=== Neural network, validation ({args.scheme}, balanced={args.balanced}, config '{cfg['name']}') ===")
    cols = ["model", "pr_auc", "roc_auc", "brier", "recall_at_p20", "recall_at_p30", "recall_at_p50",
            "f1_at_05", "f1_best", "train_seconds"]
    print(all_df[cols].round(4).to_string(index=False))
    if len(seeds_df) > 1:
        print(f"\nPR-AUC across {len(seeds_df)} seeds: " + ", ".join(f"{v:.4f}" for v in seeds_df["pr_auc"])
              + f" | mean {seeds_df['pr_auc'].mean():.4f}, std {seeds_df['pr_auc'].std():.4f}")
    print("\n=== Rows for the checkpoint results table ===")
    print(markdown_rows(nn_df, "Arshan"))
    print(f"\nSaved results to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()