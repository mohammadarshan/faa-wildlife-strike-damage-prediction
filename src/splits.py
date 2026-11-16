"""Time-aware train/validation/test splits, plus an all-years and a random split for comparison.

Run from the repo root to print the split sizes and baselines:
    python -m src.splits

In modeling code:
    from src.splits import load_split_data
    data = load_split_data("main")                  # {"train": (X, y), "validation": (X, y), "test": (X, y)}
    X_train, y_train = data["train"]

Rules: fit every preprocessing step and model on train only; tune and calibrate on validation;
touch test once, at the end.
"""
import numpy as np
import pandas as pd

from src.data_loading import ROOT, load_raw, make_target, split_features

REPORTS = ROOT / "reports"
NAMES = ["train", "validation", "test"]

# Year ranges (inclusive). 2026 is a partial year and is excluded from every scheme.
YEAR_SCHEMES = {
    # Main scheme: the stable reporting era only (damage rate ~4-5% in train, ~3.7% in validation and test)
    "main": {"train": (2014, 2022), "validation": (2023, 2024), "test": (2025, 2025)},
    # Same validation and test, but train also includes the 1990-2013 years (different reporting regime)
    "all_years": {"train": (1990, 2022), "validation": (2023, 2024), "test": (2025, 2025)},
}
RANDOM_SHARES = {"train": 0.70, "validation": 0.15, "test": 0.15}
RANDOM_YEARS = (1990, 2025)


def assign_split(df: pd.DataFrame, scheme: str = "main", seed: int = 42) -> pd.Series:
    """Label every row 'train', 'validation', 'test' or 'excluded'.
    scheme: 'main', 'all_years', or 'random' (stratified by damage; needs a 'target' column)."""
    year = pd.to_numeric(df["INCIDENT_YEAR"], errors="coerce")
    split = pd.Series("excluded", index=df.index, dtype=object)

    if scheme in YEAR_SCHEMES:
        for name, (lo, hi) in YEAR_SCHEMES[scheme].items():
            split[(year >= lo) & (year <= hi)] = name
        return split

    if scheme == "random":
        rng = np.random.default_rng(seed)
        eligible = (year >= RANDOM_YEARS[0]) & (year <= RANDOM_YEARS[1])
        for cls in (0, 1):  # split damaging and non-damaging strikes separately so the rate stays equal
            idx = np.array(df.index[eligible & (df["target"] == cls)])  # writable copy
            rng.shuffle(idx)
            n_train = int(len(idx) * RANDOM_SHARES["train"])
            n_val = int(len(idx) * RANDOM_SHARES["validation"])
            split[idx[:n_train]] = "train"
            split[idx[n_train:n_train + n_val]] = "validation"
            split[idx[n_train + n_val:]] = "test"
        return split

    raise ValueError(f"Unknown scheme {scheme!r}. Use one of {list(YEAR_SCHEMES) + ['random']}.")


def summarize_splits(df: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Rows, damaging strikes, damage rate, and the baselines any model has to beat."""
    g = df.assign(split=split).groupby("split")["target"].agg(n="size", damaged="sum", rate="mean")
    g["pr_auc_baseline"] = g["rate"]          # PR-AUC of a model with no skill equals the damage rate
    g["majority_accuracy"] = 1 - g["rate"]    # accuracy of always predicting 'no damage'
    order = [n for n in NAMES + ["excluded"] if n in g.index]
    return g.loc[order]


def load_split_data(scheme: str = "main", seed: int = 42) -> dict:
    """Return {'train': (X, y), 'validation': (X, y), 'test': (X, y)} using only the audited feature columns."""
    df = load_raw()
    df["target"] = make_target(df)
    split = assign_split(df, scheme, seed)
    X, y, _ = split_features(df)
    return {name: (X[split == name], y[split == name]) for name in NAMES}


def main():
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "eda_tables").mkdir(exist_ok=True)
    df = load_raw()
    df["target"] = make_target(df)
    out = []
    for scheme in ["main", "all_years", "random"]:
        table = summarize_splits(df, assign_split(df, scheme))
        print(f"\n=== {scheme} ===")
        print(table.round(4).to_string())
        out.append(table.reset_index().assign(scheme=scheme))
    pd.concat(out).to_csv(REPORTS / "eda_tables" / "split_summary.csv", index=False)
    print(f"\nSaved {REPORTS / 'eda_tables' / 'split_summary.csv'}")


if __name__ == "__main__":
    main()