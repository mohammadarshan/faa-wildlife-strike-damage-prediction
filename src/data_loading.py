"""Load the FAA wildlife strike data and apply the first-pass leakage audit.

Usage (from a notebook in notebooks/):
    import sys; sys.path.append("..")
    from src.data_loading import load_raw, make_target, split_features
    df = load_raw()                      # reads parquet if present, else converts the Excel file once
    df["target"] = make_target(df)       # check value_counts first (see make_target)
    X, y, audit = split_features(df)
"""
from pathlib import Path
import pandas as pd

RAW_XLSX = Path("data/raw/Public.xlsx")
PARQUET = Path("data/processed/strikes.parquet")

# ---- First-pass audit (draft: confirm as a team, edit the lists, keep the reasons) ----
LEAKAGE = {
    "COST_REPAIRS": "repair cost, known only after the strike",
    "COST_OTHER": "other cost, known only after the strike",
    "COST_REPAIRS_INFL_ADJ": "repair cost, known only after the strike",
    "COST_OTHER_INFL_ADJ": "other cost, known only after the strike",
    "AOS": "aircraft out-of-service time, after the strike",
    "EFFECT": "effect on flight (aborted takeoff etc.), after the strike",
    "EFFECT_OTHER": "effect on flight, after the strike",
    "INGESTED_OTHER": "ingestion flag, after the strike",
    "NR_INJURIES": "outcome of the strike",
    "NR_FATALITIES": "outcome of the strike",
    "REMAINS_COLLECTED": "post-strike reporting field",
    "REMAINS_SENT": "post-strike reporting field",
    "OTHER_SPECIFY": "describes damaged/struck parts, after the strike",
}
LEAKAGE_PREFIXES = {"DAM_": "damaged-component flag, after the strike",
                    "ING_": "engine ingestion flag, after the strike"}
# Target source columns: used to build the label, never as features
TARGET_SOURCES = ["INDICATED_DAMAGE", "DAMAGE_LEVEL"]
# Identifiers, free text and admin fields
ID_TEXT = ["INDEX_NR", "REG", "FLT", "BIRD_BAND_NUMBER", "REMARKS", "COMMENTS", "IMAGE",
           "REPORTED_NAME", "REPORTED_TITLE", "PERSON", "SOURCE", "LUPDATE", "TRANSFER",
           "OUT_OF_RANGE_SPECIES"]
# Borderline: dropped by default, test adding back as a sensitivity check
BORDERLINE_PREFIXES = ["STR_"]
BORDERLINE = ["NUM_STRUCK", "NUM_SEEN"]
# Kept only to build the time-aware split, not used as model features
SPLIT_ONLY = ["INCIDENT_DATE", "INCIDENT_YEAR"]


def _fix_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet needs one type per column. Excel often gives text columns that mix ints and strings
    (e.g. AIRPORT_ID). Convert those columns to strings, keeping missing values as missing."""
    df = df.copy()
    fixed = []
    for col in df.columns[df.dtypes == object]:
        kinds = set()
        for v in df[col].dropna().unique():
            if isinstance(v, bool):
                kinds.add("bool")
            elif isinstance(v, (int, float)) or hasattr(v, "dtype"):
                kinds.add("number")
            elif isinstance(v, str):
                kinds.add("str")
            else:
                kinds.add(type(v).__name__)
        if len(kinds) > 1:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
            fixed.append(col)
    if fixed:
        print(f"Converted mixed-type columns to text for parquet: {fixed}")
    return df


def load_raw() -> pd.DataFrame:
    """Load from parquet if it exists; otherwise read the Excel file once and cache it."""
    if PARQUET.exists():
        return pd.read_parquet(PARQUET)
    if not RAW_XLSX.exists():
        raise FileNotFoundError(f"Download the FAA data and place it at {RAW_XLSX}")
    df = pd.read_excel(RAW_XLSX, engine="calamine")  # pip install python-calamine
    df = _fix_mixed_types(df)
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET)
    return df


def make_target(df: pd.DataFrame, col: str = "INDICATED_DAMAGE") -> pd.Series:
    """Binary damage label. RUN df[col].value_counts(dropna=False) FIRST and confirm the coding.

    This handles 0/1, True/False and Y/N style values; anything else raises, so you notice.
    Decide with your teammate whether to use INDICATED_DAMAGE or DAMAGE_LEVEL (not implemented here).
    """
    s = df[col]
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    m = s.astype(str).str.strip().str.upper()
    yes, no = {"1", "Y", "YES", "TRUE"}, {"0", "N", "NO", "FALSE", "NAN", ""}
    unknown = set(m.unique()) - yes - no
    if unknown:
        raise ValueError(f"Unexpected values in {col}: {unknown}. Check value_counts and update make_target.")
    return m.isin(yes).astype(int)


def audit_table(columns) -> pd.DataFrame:
    """One row per column: keep/drop and the reason. Save this to reports/ for the Checkpoint."""
    rows = []
    for c in columns:
        if c in LEAKAGE:
            rows.append((c, "drop", "leakage: " + LEAKAGE[c]))
        elif any(c.startswith(p) for p in LEAKAGE_PREFIXES):
            p = next(p for p in LEAKAGE_PREFIXES if c.startswith(p))
            rows.append((c, "drop", "leakage: " + LEAKAGE_PREFIXES[p]))
        elif c in TARGET_SOURCES:
            rows.append((c, "drop", "target source"))
        elif c in ID_TEXT:
            rows.append((c, "drop", "identifier / free text / admin"))
        elif c in BORDERLINE or any(c.startswith(p) for p in BORDERLINE_PREFIXES):
            rows.append((c, "drop (borderline)", "known at/after impact; test as sensitivity check"))
        elif c in SPLIT_ONLY:
            rows.append((c, "split only", "used to build the time-aware split"))
        else:
            rows.append((c, "keep", "assumed known at or before impact (verify in EDA)"))
    return pd.DataFrame(rows, columns=["column", "decision", "reason"])


def split_features(df: pd.DataFrame, target_col: str = "target"):
    """Return (X, y, audit). X holds only 'keep' columns; split-only columns stay on df for splitting."""
    audit = audit_table([c for c in df.columns if c != target_col])
    keep = audit.loc[audit["decision"] == "keep", "column"].tolist()
    return df[keep].copy(), df[target_col].copy(), audit