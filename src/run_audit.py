"""Run the first-pass leakage audit and write the results to reports/.

Run from the repo root:
    python -m src.run_audit
"""
from pathlib import Path
from src.data_loading import load_raw, make_target, split_features

REPORTS = Path("reports")


def main():
    REPORTS.mkdir(exist_ok=True)
    df = load_raw()
    print("Shape:", df.shape)

    # Confirm how the damage columns are coded BEFORE trusting make_target
    for col in ["INDICATED_DAMAGE", "DAMAGE_LEVEL"]:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False))

    df["target"] = make_target(df)
    X, y, audit = split_features(df)
    print(f"\nDamage rate (target=1): {y.mean():.3%}  |  features kept: {X.shape[1]}")

    audit.to_csv(REPORTS / "leakage_audit.csv", index=False)

    # Missingness of kept features, worst first
    miss = X.isna().mean().sort_values(ascending=False).rename("pct_missing")
    miss.to_csv(REPORTS / "missingness_kept_features.csv")
    print("\nMost-missing kept features:")
    print(miss.head(15))
    print(f"\nSaved: {REPORTS/'leakage_audit.csv'}, {REPORTS/'missingness_kept_features.csv'}")


if __name__ == "__main__":
    main()