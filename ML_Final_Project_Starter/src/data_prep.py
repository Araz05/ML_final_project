"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Fill in the TODOs. Keep ALL data wrangling here so the rest of the code just
calls these functions. Fix your random seed everywhere (see SEED).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/bina_az_sale.csv"   # change if your filename differs

# Columns that LEAK the target (derived from price). Drop these before training.
# TODO: list the actual leakage columns you find (e.g. a unit/per-m2 price).
LEAKAGE_COLUMNS: list[str] = [
    # "unit_price",
    # "total_price",
]

RENAME_MAP = {
    "sahe": "area",
    "otaq_sayi": "rooms",
    "mertebe": "floor",
    "torpaq_sahesi": "land_area",
    "binanin_novu": "building_type",
    "kateqoriya": "category",
    "temir": "repair",
    "ipoteka": "mortgage",
    "cixaris": "bill_of_sale",
}
#Columns that must be numeric after cleaning (names AFTER renaming).
#`floor`("7 / 9") is split into `floor` and `floor_total` by coerce_numeric.
NUMERIC_COLS = ["price", "area", "rooms", "land_area", "lat", "lng"]


CURRENCY_COL_CANDIDATES = ("currency_x", "currency_y", "currency", "valyuta")

# Everything is converted to AZN. 1 USD = 1.70 AZN is the official fixed rate;
FX_TO_AZN = {"AZN": 1.0, "USD": 1.70, "EUR": 1.85}

SOT_TO_M2 = 100.0

_AZ_FOLD = str.maketrans({
    "ə": "e", "Ə": "e", "ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s",
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ö": "o", "Ö": "o",
    "ü": "u", "Ü": "u",
})

def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def _fold(name: str) -> str:
    """'Otaq sayı' -> 'otaq_sayi' (ASCII, lowercase, underscores)."""
    name = str(name).translate(_AZ_FOLD).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")
 
 
def standardize_columns(df):
    """
    Give every column a stable English snake_case name, with NO duplicates.
    Azerbaijani names are translated via RENAME_MAP."""
    df = df.copy()
    folded = [_fold(c) for c in df.columns]
    mapped_targets = {RENAME_MAP[f] for f in folded if f in RENAME_MAP}
 
    new_names = []
    for f in folded:
        if f in RENAME_MAP:
            new_names.append(RENAME_MAP[f])
        elif f in mapped_targets:
            new_names.append(f + "_listing")
        else:
            new_names.append(f)
    df.columns = new_names
    return df
 
 
def _to_number(s):
    """'120 000 AZN' / '85,5 m²' -> float. Unparseable values become NaN
    (handled later as missing)."""
    cleaned = (
        s.astype("string")
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)   # drop spaces, currency, units
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")
 
 
def _area_to_m2(s):
    """'145 m²' -> 145.0 ; '3 sot' -> 300.0 (1 sot = 100 m²)."""
    num = _to_number(s)
    is_sot = s.astype("string").str.contains("sot", case=False).fillna(False).astype(bool)
    return num.where(~is_sot, num * SOT_TO_M2)
 
 
def _split_floor(s):
    """'7 / 9' -> (floor=7, floor_total=9). Missing/odd -> (NaN, NaN)."""
    parts = s.astype("string").str.extract(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
    return (pd.to_numeric(parts[0], errors="coerce").astype("float64"),
            pd.to_numeric(parts[1], errors="coerce").astype("float64"))
 
 
def coerce_numeric(df, cols=NUMERIC_COLS):
    """
    Cast the numeric columns to float.
      - area and land_area are converted to m² (sot x 100),
      - floor "7 / 9" is split into `floor` and `floor_total`,
      - columns that are already numeric are left alone.
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns or pd.api.types.is_numeric_dtype(df[c]):
            continue
        df[c] = _area_to_m2(df[c]) if c in ("area", "land_area") else _to_number(df[c])
    # area / land_area that arrived numeric are assumed to be m² already.
 
    if "floor" in df.columns and not pd.api.types.is_numeric_dtype(df["floor"]):
        df["floor"], df["floor_total"] = _split_floor(df["floor"])
    return df
 
 
def _detect_currency(text) -> str:
    """Find the currency in a string; AZN if none is written."""
    t = str(text).upper()
    if "USD" in t or "$" in t:
        return "USD"
    if "EUR" in t or "\u20ac" in t:
        return "EUR"
    return "AZN"  
 
 
def normalize_currency(df, price_col: str = "price"):
    """Convert prices to AZN. Run it BEFORE coerce_numeric (that function would
    strip the currency text)"""
    df = df.copy()
    if price_col not in df.columns:
        return df
 
    cur_col = next((c for c in CURRENCY_COL_CANDIDATES if c in df.columns), None)
    source = df[cur_col] if cur_col else df[price_col]
    cur = source.map(_detect_currency)
    num = df[price_col] if pd.api.types.is_numeric_dtype(df[price_col]) \
        else _to_number(df[price_col])
    df[price_col] = num * cur.map(FX_TO_AZN).astype(float)
    df["price_currency_orig"] = cur
    return df


def clean(df):
    """
    Clean the raw frame:
      - fix dtypes (prices/areas to numeric; strip currency/text),
      - handle missing values,
      - drop duplicates,
      - handle outliers (document your rule),
      - drop LEAKAGE_COLUMNS.
    Return the cleaned DataFrame.
    """
    raise NotImplementedError("TODO: clean the data")


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    raise NotImplementedError("TODO: build X, y")


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    raise NotImplementedError("TODO: derive the price-tier label")


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    raise NotImplementedError("TODO: split the data")


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    raise NotImplementedError("TODO: standardize using train stats")