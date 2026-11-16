"""Exploratory data analysis. Saves tables to reports/eda_tables/ and figures to reports/figures/.

Run from the repo root:
    python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loading import ROOT, load_raw, make_target

REPORTS = ROOT / "reports"
CATEGORICAL = ["PHASE_OF_FLIGHT", "SIZE", "TIME_OF_DAY", "SKY", "PRECIPITATION",
               "WARNED", "AC_CLASS", "TYPE_ENG", "NUM_ENGS"]
MISSING_COLS = ["HEIGHT", "SPEED", "SKY", "PHASE_OF_FLIGHT", "TIME_OF_DAY", "DISTANCE", "AC_MASS"]
HEIGHT_BINS = [-1, 0, 50, 100, 500, 1000, 2000, 5000, 10000, np.inf]   # feet above ground
SPEED_BINS = [-1, 0, 50, 100, 150, 200, 250, 300, np.inf]              # knots


def rate_table(df, by):
    """Strikes, damaging strikes and damage rate for each value of `by`."""
    g = df.groupby(by, dropna=False, observed=True)["target"].agg(n="size", damaged="sum", rate="mean")
    return g.reset_index()


def bar_rate(table, col, title, path, base_rate, min_n=200, order_by_rate=True):
    """Bar chart of damage rate per group, labelled with the group size. Small groups are skipped."""
    t = table[table["n"] >= min_n].copy()
    if t.empty:
        return
    t[col] = t[col].map(lambda v: "(blank)" if pd.isna(v) else str(v))  # blank group gets a label
    if order_by_rate:
        t = t.sort_values("rate")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(t) + 1.5)))
    ax.barh(t[col], t["rate"] * 100, color="#1F3A5F")
    ax.axvline(base_rate * 100, color="red", linestyle="--", label=f"overall {base_rate:.1%}")
    for i, (r, n) in enumerate(zip(t["rate"], t["n"])):
        ax.text(r * 100, i, f" {r:.1%} (n={n:,})", va="center", fontsize=8)
    ax.set_xlabel("Damage rate (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, t["rate"].max() * 100 * 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def suggest_cuts(by_year, train_share=0.70, val_share=0.15):
    """Pick cut years so train/validation/test hold roughly 70/15/15 of the strikes."""
    t = by_year.sort_values("INCIDENT_YEAR").copy()
    t["cum_share"] = t["n"].cumsum() / t["n"].sum()
    train_end = int(t.loc[t["cum_share"] >= train_share, "INCIDENT_YEAR"].iloc[0])
    val_end = int(t.loc[t["cum_share"] >= train_share + val_share, "INCIDENT_YEAR"].iloc[0])
    return train_end, val_end


def main():
    tables, figs = REPORTS / "eda_tables", REPORTS / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df["target"] = make_target(df)
    for c in ["HEIGHT", "SPEED", "INCIDENT_YEAR", "INCIDENT_MONTH"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    base = df["target"].mean()
    print(f"Rows: {len(df):,} | damaging: {int(df['target'].sum()):,} | overall damage rate: {base:.2%}")

    # ---- 1. Damage rate and reporting volume by year (drives the time-aware split) ----
    by_year = rate_table(df, "INCIDENT_YEAR").dropna(subset=["INCIDENT_YEAR"])
    by_year["INCIDENT_YEAR"] = by_year["INCIDENT_YEAR"].astype(int)
    by_year.to_csv(tables / "damage_by_year.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(by_year["INCIDENT_YEAR"], by_year["n"], color="#B8C6DB")
    ax.set_ylabel("Reported strikes")
    ax2 = ax.twinx()
    ax2.plot(by_year["INCIDENT_YEAR"], by_year["rate"] * 100, color="#1F3A5F", marker="o")
    ax2.set_ylabel("Damage rate (%)")
    ax.set_title("Reported strikes (bars) and damage rate (line) by year")
    fig.tight_layout()
    fig.savefig(figs / "damage_by_year.png", dpi=150)
    plt.close(fig)

    # ---- 2. Suggested time-aware split ----
    train_end, val_end = suggest_cuts(by_year)
    yr = df["INCIDENT_YEAR"]
    split = pd.Series("test", index=df.index)
    split[yr <= val_end] = "validation"
    split[yr <= train_end] = "train"
    split[yr.isna()] = "no year"
    split_tbl = df.assign(split=split).groupby("split")["target"].agg(n="size", damaged="sum", rate="mean")
    split_tbl.to_csv(tables / "suggested_split.csv")
    print(f"\nSuggested split (about 70/15/15): train <= {train_end}, validation {train_end + 1}-{val_end}, test > {val_end}")
    print(split_tbl.round(4))

    # ---- 3. Damage rate by categorical fields ----
    for col in CATEGORICAL:
        t = rate_table(df, col)
        t.to_csv(tables / f"damage_by_{col.lower()}.csv", index=False)
        bar_rate(t, col, f"Damage rate by {col}", figs / f"damage_by_{col.lower()}.png", base)

    month = rate_table(df, "INCIDENT_MONTH").dropna(subset=["INCIDENT_MONTH"]).sort_values("INCIDENT_MONTH")
    month.to_csv(tables / "damage_by_month.csv", index=False)
    bar_rate(month, "INCIDENT_MONTH", "Damage rate by month", figs / "damage_by_month.png", base, order_by_rate=False)

    # ---- 4. Height and speed: distributions and damage rate by bin ----
    for col, bins, unit in [("HEIGHT", HEIGHT_BINS, "ft"), ("SPEED", SPEED_BINS, "kt")]:
        vals = df[col].dropna()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(vals[vals <= vals.quantile(0.99)], bins=50, color="#1F3A5F")
        ax.set_title(f"{col} ({unit}), values up to the 99th percentile; {df[col].isna().mean():.0%} missing")
        fig.tight_layout()
        fig.savefig(figs / f"hist_{col.lower()}.png", dpi=150)
        plt.close(fig)
        df[f"{col}_bin"] = pd.cut(df[col], bins=bins)
        t = rate_table(df.dropna(subset=[col]), f"{col}_bin")
        t.to_csv(tables / f"damage_by_{col.lower()}_bin.csv", index=False)
        bar_rate(t, f"{col}_bin", f"Damage rate by {col} ({unit}) bin", figs / f"damage_by_{col.lower()}_bin.png",
                 base, min_n=100, order_by_rate=False)

    # ---- 5. Species: unknown labels and the most common species ----
    unknown = df["SPECIES"].astype(str).str.startswith("Unknown")
    df["species_unknown"] = np.where(unknown, "unknown species", "identified species")
    t = rate_table(df, "species_unknown")
    t.to_csv(tables / "damage_by_species_unknown.csv", index=False)
    print("\nDamage rate by unknown vs identified species:")
    print(t.round(4).to_string(index=False))
    top = rate_table(df, "SPECIES").sort_values("n", ascending=False).head(25)
    top.to_csv(tables / "top25_species.csv", index=False)
    bar_rate(top, "SPECIES", "Damage rate, 25 most reported species", figs / "damage_top_species.png", base, min_n=1)

    # ---- 6. Is blank data informative? Damage rate when blank vs present ----
    rows = []
    for col in MISSING_COLS:
        blank = df[col].isna()
        rows.append({"column": col, "pct_blank": blank.mean(),
                     "rate_if_blank": df.loc[blank, "target"].mean(),
                     "rate_if_present": df.loc[~blank, "target"].mean(), "n_blank": int(blank.sum())})
    miss = pd.DataFrame(rows)
    miss.to_csv(tables / "missingness_vs_damage.csv", index=False)
    print("\nDamage rate when blank vs present:")
    print(miss.round(4).to_string(index=False))

    # ---- 7. Did reporting change over the years? Percent blank per year ----
    yearly = df.dropna(subset=["INCIDENT_YEAR"]).copy()
    yearly["unknown_species_share"] = unknown[yearly.index].astype(float)
    cols = ["HEIGHT", "SPEED", "SKY", "PHASE_OF_FLIGHT"]
    pct = pd.DataFrame({c: yearly[c].isna().groupby(yearly["INCIDENT_YEAR"]).mean() for c in cols})
    pct["unknown_species"] = yearly.groupby("INCIDENT_YEAR")["unknown_species_share"].mean()
    pct.to_csv(tables / "blank_share_by_year.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in pct.columns:
        ax.plot(pct.index, pct[c] * 100, label=c)
    ax.set_ylabel("% blank / unknown")
    ax.set_title("How complete are reports, by year?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "blank_share_by_year.png", dpi=150)
    plt.close(fig)

    print(f"\nTables saved to {tables}\nFigures saved to {figs}")


if __name__ == "__main__":
    main()