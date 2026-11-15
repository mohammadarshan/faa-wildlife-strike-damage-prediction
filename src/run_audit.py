"""Run the leakage audit and write the results to reports/.

Run from the repo root:
    python -m src.run_audit
"""
from src.data_loading import ROOT, load_raw, make_target, split_features

REPORTS = ROOT / "reports"
MISSING_CHECK_COLS = ["HEIGHT", "SPEED", "SKY", "PHASE_OF_FLIGHT"]


def main():
    REPORTS.mkdir(exist_ok=True)
    df = load_raw()
    print("Shape:", df.shape)

    # Confirm how the damage columns are coded before trusting make_target
    for col in ["INDICATED_DAMAGE", "DAMAGE_LEVEL"]:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False))

    df["target"] = make_target(df)
    X, y, audit = split_features(df)
    print(f"\nDamage rate (target=1): {y.mean():.3%} ({int(y.sum())} of {len(y)})  |  features kept: {X.shape[1]}")

    audit.to_csv(REPORTS / "leakage_audit.csv", index=False)

    # Missingness of kept features, worst first
    miss = X.isna().mean().sort_values(ascending=False).rename("pct_missing")
    miss.to_csv(REPORTS / "missingness_kept_features.csv")
    print("\nMost-missing kept features:")
    print(miss.head(15))

    # Is missingness informative? Damage rate when the field is blank vs present
    print("\nDamage rate when a field is blank vs present:")
    for col in MISSING_CHECK_COLS:
        blank = X[col].isna()
        print(f"{col}: blank {y[blank].mean():.1%} of n={int(blank.sum())} | "
              f"present {y[~blank].mean():.1%} of n={int((~blank).sum())}")

    print(f"\nSaved to {REPORTS}")


if __name__ == "__main__":
    main()