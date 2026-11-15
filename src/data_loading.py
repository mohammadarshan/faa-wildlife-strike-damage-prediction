"""Load the FAA wildlife strike data and apply the leakage audit.

Run scripts from the repo root, e.g.:  python -m src.run_audit
Paths below are anchored to the repo root, so imports also work from other folders.

    from src.data_loading import load_raw, make_target, split_features
    df = load_raw()                      # reads the parquet cache, or converts the Excel file once
    df["target"] = make_target(df)       # raises if the damage column has missing values
    X, y, audit = split_features(df)     # raises if any column has not been classified
"""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data/raw/Public.xlsx"
CACHE_VERSION = 2  # bump when the cleaning below changes, so an old cache is not reused
PARQUET = ROOT / f"data/processed/strikes_v{CACHE_VERSION}.parquet"

# ---------------------------------------------------------------------------
# Leakage audit. EVERY column of the FAA file must land in exactly one group below.
# audit_table() raises on any column that is not listed (fail closed), and warns
# about listed names that are missing from the data (typo or renamed FAA column).
# ---------------------------------------------------------------------------

# Recorded after the strike -> never a feature
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
LEAKAGE_PREFIXES = {
    "DAM_": "damaged-component flag, after the strike",
    "ING_": "engine ingestion flag, after the strike",
}

# Used to build the label, never as features
TARGET_SOURCES = ["INDICATED_DAMAGE", "DAMAGE_LEVEL"]

# Identifiers, free text and admin fields
ID_TEXT = [
    "INDEX_NR", "REG", "FLT", "BIRD_BAND_NUMBER", "REMARKS", "COMMENTS", "IMAGE",
    "REPORTED_NAME", "REPORTED_TITLE", "PERSON", "SOURCE", "LUPDATE", "TRANSFER",
    "OUT_OF_RANGE_SPECIES",
]

# Another column already carries the same information, or the column is structurally empty
REDUNDANT = {
    # Most aircraft have 1-2 engines, so these are empty by design. NUM_ENGS covers it.
    "ENG_3_POS": "structurally empty (most aircraft have <3 engines); NUM_ENGS covers it",
    "ENG_4_POS": "structurally empty (most aircraft have <4 engines); NUM_ENGS covers it",
    # Only filled in for en-route strikes; STATE and PHASE_OF_FLIGHT cover the same ground
    "ENROUTE_STATE": "almost always blank; overlaps STATE and PHASE_OF_FLIGHT",
    "LOCATION": "free text, mostly blank",
    # Name/ID pairs: keep the ID, drop the name
    "AIRPORT": "duplicate of AIRPORT_ID (name variants for a few IDs, e.g. ARPT vs AIRPORT)",
    "OPERATOR": "duplicate of OPID (one-to-one)",
    # Species: keep the readable name, drop the code (effectively one-to-one).
    # SPECIES itself is mostly "Unknown bird..." labels, so add a species_unknown flag later.
    "SPECIES_ID": "duplicate of SPECIES (one ID has a name variant)",
}

# Known at or just after impact, but tied closely to the outcome: dropped by default,
# test adding them back as a sensitivity check
BORDERLINE = ["NUM_STRUCK", "NUM_SEEN"]
BORDERLINE_PREFIXES = ["STR_"]

# Kept only to build the time-aware split, never used as features
SPLIT_ONLY = ["INCIDENT_DATE", "INCIDENT_YEAR"]

# Features we chose to keep (known at or before impact)
KEEP = [
    "INCIDENT_MONTH", "TIME", "TIME_OF_DAY",
    "AIRPORT_ID", "LATITUDE", "LONGITUDE", "RUNWAY", "STATE", "FAAREGION",
    "OPID", "AIRCRAFT", "AMA", "AMO", "EMA", "EMO",
    "AC_CLASS", "AC_MASS", "TYPE_ENG", "NUM_ENGS", "ENG_1_POS", "ENG_2_POS",
    "PHASE_OF_FLIGHT", "HEIGHT", "SPEED", "DISTANCE", "SKY", "PRECIPITATION",
    "SPECIES", "SIZE", "WARNED",
]

# Columns that should be numbers but arrive as text because of stray entries
NUMERIC_COLS = ["LATITUDE", "LONGITUDE"]


def _clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Make the columns parquet-safe without damaging real numeric columns."""
    df = df.copy()

    # 1. Numeric columns with stray text -> numbers (bad entries become missing, and we say how many)
    for col in NUMERIC_COLS:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            lost = df[col].isna().sum() - before
            if lost:
                print(f"{col}: {lost} non-numeric entries set to missing")

    # 2. Other columns that mix numbers and text -> text (parquet needs one type per column)
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
    """Load the parquet cache; rebuild it from the Excel file if it is missing or older than the Excel file."""
    stale = RAW_XLSX.exists() and PARQUET.exists() and RAW_XLSX.stat().st_mtime > PARQUET.stat().st_mtime
    if PARQUET.exists() and not stale:
        return pd.read_parquet(PARQUET)
    if not RAW_XLSX.exists():
        raise FileNotFoundError(f"Download the FAA data and place it at {RAW_XLSX}")
    df = pd.read_excel(RAW_XLSX, engine="calamine")  # pip install python-calamine
    df = _clean_types(df)
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tmp = PARQUET.with_name(PARQUET.name + ".tmp")
    df.to_parquet(tmp)           # write to a temp file first, so a crash cannot leave a broken cache
    os.replace(tmp, PARQUET)
    return df


def make_target(df: pd.DataFrame, col: str = "INDICATED_DAMAGE") -> pd.Series:
    """Binary damage label (1 = damaged). Raises if the column has missing or unexpected values,
    so a missing value can never silently become 'no damage'."""
    s = df[col]
    n_missing = int(s.isna().sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} of {len(s)} rows have missing {col}. Decide whether to drop them or keep "
            "them as 'unknown' (and note the decision in the report) before building the target."
        )
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (s > 0).astype(int)
    m = s.astype(str).str.strip().str.upper()
    yes, no = {"1", "Y", "YES", "TRUE"}, {"0", "N", "NO", "FALSE"}
    unknown = set(m.unique()) - yes - no
    if unknown:
        raise ValueError(f"Unexpected values in {col}: {unknown}. Check value_counts and update make_target.")
    return m.isin(yes).astype(int)


def audit_table(columns, extra_ok=()) -> pd.DataFrame:
    """One row per column: decision and reason. Raises on any column that has not been classified.
    extra_ok: names of derived columns that are allowed to exist but are never features."""
    columns = list(columns)
    listed = (set(LEAKAGE) | set(TARGET_SOURCES) | set(ID_TEXT) | set(REDUNDANT)
              | set(BORDERLINE) | set(SPLIT_ONLY) | set(KEEP))
    missing = sorted(listed - set(columns))
    if missing:
        print(f"WARNING: listed in the audit but not in the data (typo or renamed FAA column?): {missing}")

    rows, unclassified = [], []
    for c in columns:
        if c in extra_ok:
            rows.append((c, "drop", "derived column, not a feature"))
        elif c in LEAKAGE:
            rows.append((c, "drop", "leakage: " + LEAKAGE[c]))
        elif any(c.startswith(p) for p in LEAKAGE_PREFIXES):
            p = next(p for p in LEAKAGE_PREFIXES if c.startswith(p))
            rows.append((c, "drop", "leakage: " + LEAKAGE_PREFIXES[p]))
        elif c in TARGET_SOURCES:
            rows.append((c, "drop", "target source"))
        elif c in ID_TEXT:
            rows.append((c, "drop", "identifier / free text / admin"))
        elif c in REDUNDANT:
            rows.append((c, "drop", "redundant: " + REDUNDANT[c]))
        elif c in BORDERLINE or any(c.startswith(p) for p in BORDERLINE_PREFIXES):
            rows.append((c, "drop (borderline)", "tied closely to the outcome; test as sensitivity check"))
        elif c in SPLIT_ONLY:
            rows.append((c, "split only", "used to build the time-aware split"))
        elif c in KEEP:
            rows.append((c, "keep", "known at or before impact"))
        else:
            unclassified.append(c)
    if unclassified:
        raise ValueError(
            f"Unclassified columns: {unclassified}. Add each one to KEEP or to a drop list in data_loading.py."
        )
    return pd.DataFrame(rows, columns=["column", "decision", "reason"])


def split_features(df: pd.DataFrame, target_col: str = "target"):
    """Return (X, y, audit). X holds only the columns in KEEP."""
    audit = audit_table([c for c in df.columns if c != target_col])
    keep = audit.loc[audit["decision"] == "keep", "column"].tolist()
    return df[keep].copy(), df[target_col].copy(), audit