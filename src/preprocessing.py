"""Preprocessing pipeline for the 30 audited features.

Run from the repo root to check it on the real data:
    python -m src.preprocessing

In modeling code:
    from src.preprocessing import make_matrices
    m = make_matrices("main", add_missing_indicators=True, scale=True)   # scale=True for LR / neural net
    X_train, y_train = m["train"]
    X_val, y_val = m["validation"]
    pre = m["preprocessor"]          # already fitted on train only

Design choices
- Everything that learns from data (medians, category frequencies, scaling) is fitted on TRAIN only.
- add_missing_indicators=True: blanks stay visible to the model (indicator columns for numeric fields,
  '(blank)' as its own category for text fields). False: blanks are imputed (median / most common value)
  and carry no signal. Run models both ways: blank fields are strongly tied to damage in this data.
- High-cardinality ID columns (airport, operator, aircraft, species...) are frequency-encoded, not one-hot
  encoded. CatBoost can use them natively later.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.splits import NAMES, assign_split
from src.data_loading import load_raw, make_target, split_features

NUMERIC = ["INCIDENT_MONTH", "TIME_HOUR", "LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE",
           "AC_MASS", "NUM_ENGS"]
LOW_CARD = ["PHASE_OF_FLIGHT", "SKY", "PRECIPITATION", "WARNED", "SIZE", "TIME_OF_DAY", "AC_CLASS",
            "TYPE_ENG", "ENG_1_POS", "ENG_2_POS", "FAAREGION", "STATE"]
HIGH_CARD = ["AIRPORT_ID", "OPID", "AIRCRAFT", "AMA", "AMO", "EMA", "EMO", "SPECIES", "RUNWAY"]
BINARY = ["species_unknown"]
NUMERIC_RAW = ["INCIDENT_MONTH", "LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE", "AC_MASS", "NUM_ENGS"]


def _stringify(s: pd.Series) -> pd.Series:
    """Every non-blank value becomes text; blanks stay NaN. Stops numeric codes (e.g. 1.0) from mixing with text."""
    s = pd.Series(s)
    return s.astype(object).where(s.isna(), s.astype(str))


def _parse_hour(t: pd.Series) -> pd.Series:
    """Hour of day (0-23) from TIME. Accepts 'HH:MM', 'HH:MM:SS', and 'HHMM' / 'HMM'; anything else becomes blank."""
    t = t.astype("string").str.strip()
    hour = t.str.extract(r"^(\d{1,2}):")[0]
    digits = t.str.extract(r"^(\d{3,4})(?:\.0)?$")[0].str.zfill(4).str[:2]
    hour = pd.to_numeric(hour.fillna(digits), errors="coerce")
    return hour.where(hour.between(0, 23))


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature building (nothing is learned from data, so it is safe to run before splitting).
    - numeric columns are coerced to numbers
    - TIME ("HH:MM") becomes TIME_HOUR
    - species_unknown = 1 when the species name starts with 'Unknown'
    """
    X = X.copy()
    for c in NUMERIC_RAW:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X["TIME_HOUR"] = _parse_hour(X["TIME"])
    X["species_unknown"] = X["SPECIES"].astype("string").str.startswith("Unknown").fillna(False).astype(int)
    for c in LOW_CARD + HIGH_CARD:                      # text columns: all values text, blanks as NaN
        X[c] = _stringify(X[c])
    return X.drop(columns=["TIME"])


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by how common it is in the training data (0 for unseen categories).
    blank='keep': blanks form their own category. blank='mode': blanks become the most common category."""

    def __init__(self, blank="keep"):
        self.blank = blank

    def _clean(self, s):
        s = s.astype(object)
        return s.where(s.notna(), "__blank__").astype(str)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.maps_, self.modes_ = {}, {}
        for c in self.columns_:
            counts = X[c].dropna().astype(str).value_counts()
            self.modes_[c] = counts.index[0] if len(counts) else "__blank__"
            s = self._clean(X[c])
            if self.blank == "mode":
                s = s.where(s != "__blank__", self.modes_[c])
            self.maps_[c] = s.value_counts(normalize=True)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = {}
        for c in self.columns_:
            s = self._clean(X[c])
            if self.blank == "mode":
                s = s.where(s != "__blank__", self.modes_[c])
            out[c + "_freq"] = s.map(self.maps_[c]).fillna(0.0).astype(float).to_numpy()
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([c + "_freq" for c in self.columns_])


def _to_object(X):
    X = pd.DataFrame(X)
    return X.apply(_stringify)


def build_preprocessor(add_missing_indicators=True, scale=False) -> ColumnTransformer:
    """Unfitted preprocessor. scale=True standardizes numeric and frequency columns (for LR / neural net)."""
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=add_missing_indicators))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    if add_missing_indicators:
        cat_imputer = SimpleImputer(strategy="constant", fill_value="(blank)")
    else:
        cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_steps = [
        ("to_object", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
        ("impute", cat_imputer),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200,
                                 sparse_output=False, dtype=np.float32)),
    ]
    freq_steps = [("freq", FrequencyEncoder(blank="keep" if add_missing_indicators else "mode"))]
    if scale:
        freq_steps.append(("scale", StandardScaler()))

    pre = ColumnTransformer(
        [("num", Pipeline(num_steps), NUMERIC),
         ("cat", Pipeline(cat_steps), LOW_CARD),
         ("freq", Pipeline(freq_steps), HIGH_CARD),
         ("bin", "passthrough", BINARY)],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


def make_matrices(scheme="main", add_missing_indicators=True, scale=False, seed=42) -> dict:
    """Split with src.splits, build features, fit the preprocessor on TRAIN only, transform all three splits.
    Returns {'train': (X, y), 'validation': (X, y), 'test': (X, y), 'preprocessor': fitted}."""
    df = load_raw()
    df["target"] = make_target(df)
    split = assign_split(df, scheme, seed)
    X, y, _ = split_features(df)
    X = add_features(X)
    pre = build_preprocessor(add_missing_indicators, scale)
    pre.fit(X[split == "train"])
    out = {n: (pre.transform(X[split == n]), y[split == n]) for n in NAMES}
    out["preprocessor"] = pre
    return out


def main():
    # Sanity check: TIME_HOUR should be blank about as often as TIME (about 34% in the real data).
    # If it is blank much more often, the TIME format is not "HH:MM" and add_features needs a new pattern.
    raw = load_raw()
    print(f"TIME blank: {raw['TIME'].isna().mean():.1%} | TIME_HOUR blank after parsing: "
          f"{add_features(split_features(raw.assign(target=0))[0])['TIME_HOUR'].isna().mean():.1%}")
    raw_time = raw["TIME"].astype("string")
    unparsed = raw_time[raw_time.notna() & _parse_hour(raw["TIME"]).isna()]
    if len(unparsed):
        print(f"TIME values that could not be parsed ({len(unparsed):,}); most common: "
              f"{unparsed.value_counts().head(10).to_dict()}")
    for indicators in (True, False):
        m = make_matrices("main", add_missing_indicators=indicators, scale=True)
        Xtr, ytr = m["train"]
        print(f"\n=== missing indicators: {indicators} ===")
        for n in NAMES:
            X, y = m[n]
            print(f"{n:<11} rows={len(X):>7,}  features={X.shape[1]}  NaN left={int(X.isna().sum().sum())}  damage rate={y.mean():.2%}")
        extra = [c for c in Xtr.columns if c.startswith("missingindicator_")]
        print(f"missing-indicator columns: {len(extra)}  | first features: {list(Xtr.columns[:6])}")


if __name__ == "__main__":
    main()