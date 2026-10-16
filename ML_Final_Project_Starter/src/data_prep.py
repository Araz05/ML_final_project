# """
# data_prep.py — loading, cleaning, splitting, and the price-tier label.

# Fill in the TODOs. Keep ALL data wrangling here so the rest of the code just
# calls these functions. Fix your random seed everywhere (see SEED).
# """

# from __future__ import annotations

# import numpy as np

# # --- Config -----------------------------------------------------------------
# SEED = 42
# DATA_PATH = "data/bina_az_sale.csv"   # change if your filename differs

# # Columns that LEAK the target (derived from price). Drop these before training.
# # TODO: list the actual leakage columns you find (e.g. a unit/per-m2 price).
# LEAKAGE_COLUMNS: list[str] = [
#     # "unit_price",
#     # "total_price",
# ]


# def load_raw(path: str = DATA_PATH):
#     """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
#     raise NotImplementedError("TODO: load the raw CSV")


# def clean(df):
#     """
#     Clean the raw frame:
#       - fix dtypes (prices/areas to numeric; strip currency/text),
#       - handle missing values,
#       - drop duplicates,
#       - handle outliers (document your rule),
#       - drop LEAKAGE_COLUMNS.
#     Return the cleaned DataFrame.
#     """
#     df = df.copy()
    
#     # 1. Drop leakage columns immediately
#     df = df.drop(columns=[col for col in LEAKAGE_COLUMNS if col in df.columns], errors='ignore')
    
#     # 2. Drop duplicates
#     df = df.drop_duplicates()
    
#     # 3. Clean numeric columns (stripping 'AZN', 'm2', spaces, etc.)
#     # Update these column names based on the exact CSV headers.
#     numeric_cols = ['price', 'Sahə (area)', 'Otaq sayı (rooms)', 'Mərtəbə (floor)', 'lat', 'lng']
#     for col in numeric_cols:
#         if col in df.columns:
#             # Strip everything except digits and decimals, then convert to float
#             df[col] = df[col].astype(str).replace(r'[^\d.]', '', regex=True)
#             df[col] = pd.to_numeric(df[col], errors='coerce')
            
#     # 4. Handle missing values (e.g., drop rows where target 'price' is missing)
#     df = df.dropna(subset=['price'])
    
#     # Optional: Fill missing numericals with median, categoricals with mode
#     # For now, a strict dropna on critical features is safest.
#     df = df.dropna()
    
#     # 5. Outlier removal (e.g., keep prices within the 1st and 99th percentile)
#     q_low = df["price"].quantile(0.01)
#     q_hi  = df["price"].quantile(0.99)
#     df = df[(df["price"] > q_low) & (df["price"] < q_hi)]
    
#     return df

#     raise NotImplementedError("TODO: clean the data")


# def make_features(df):
#     """
#     Turn the cleaned frame into a numeric design matrix.
#       - encode categoricals (one-hot / ordinal — your choice, justify it),
#       - assemble X (features) and y (target `price`).
#     Return (X, y, feature_names).
#     """
#     # Define categorical columns to one-hot encode
#     cat_cols = ['location', 'city', 'Binanın növü', 'Kateqoriya', 'Təmir (repair)', 'İpoteka (mortgage)', 'Çıxarış']
#     existing_cats = [c for c in cat_cols if c in df.columns]
    
#     # One-hot encode categoricals
#     df_encoded = pd.get_dummies(df, columns=existing_cats, drop_first=True)
    
#     # Drop any leftover text columns (like description)
#     text_cols = ['description', 'attributes']
#     df_encoded = df_encoded.drop(columns=[c for c in text_cols if c in df_encoded.columns], errors='ignore')
    
#     # Separate X and y
#     y = df_encoded['price'].values
#     X_df = df_encoded.drop(columns=['price'])
#     X = X_df.values.astype(float)
    
#     return X, y, X_df.columns.tolist()
#     raise NotImplementedError("TODO: build X, y")


# def make_tier_label(y_price, threshold=None):
#     """
#     Derived classification target: price TIER (premium vs standard).
#     Use the MEDIAN of the TRAINING price as the split threshold.
#       - premium = 1 (price >  threshold), standard = 0.
#     Return (y_tier, threshold).  The threshold must come from TRAIN only
#     (no peeking at val/test) to avoid leakage.
#     """
#     if threshold is None:
#         threshold = np.median(y_price)
    
#     # 1 for premium (above threshold), 0 for standard
#     y_tier = (y_price > threshold).astype(int)
#     return y_tier, threshold
#     raise NotImplementedError("TODO: derive the price-tier label")


# def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
#     """
#     Deterministic split into train / validation / test.
#     Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
#     """
#     np.random.seed(seed)
#     indices = np.random.permutation(len(X))
    
#     test_idx = int(len(X) * test_size)
#     val_idx = int(len(X) * val_size)
    
#     test_indices = indices[:test_idx]
#     val_indices = indices[test_idx:test_idx + val_idx]
#     train_indices = indices[test_idx + val_idx:]
    
#     return (X[train_indices], y[train_indices],
#             X[val_indices], y[val_indices],
#             X[test_indices], y[test_indices])


#     raise NotImplementedError("TODO: split the data")


# def standardize(X_tr, *others):
#     """
#     Standardize features using TRAIN statistics only (mean/std from X_tr),
#     then apply the same transform to the other splits.
#     Return the scaled arrays in the same order as given.
#     (Needed for the SVM; trees don't require it.)
#     """
#     mean = np.mean(X_tr, axis=0)
#     std = np.std(X_tr, axis=0) + 1e-8 
        
#     X_tr_scaled = (X_tr - mean) / std
    
#     scaled_others = []
#     for X_other in others:
#         scaled_others.append((X_other - mean) / std)
        
#     return tuple([X_tr_scaled] + scaled_others)
#     raise NotImplementedError("TODO: standardize using train stats")





























"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Problem 1 of the project: turn the raw bina.az dump into a leakage-free
modelling dataset. All wrangling lives here; the rest of the code only calls
these functions. Randomness is always seeded (see SEED).

Pipeline order (this order matters for avoiding leakage):
    load_raw -> clean -> make_features -> train_val_test_split
             -> impute_median (train stats) -> standardize (train stats)
             -> make_tier_label (train median)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/bina_az_sale.csv"   # change if your filename differs

# Columns that LEAK the target (arithmetic functions of price).
LEAKAGE_COLUMNS: list[str] = ["unit_price", "total_price"]

# Identifiers / free text we do not model on (they identify a listing or its
# owner rather than describe the property). description/attributes are free
# text: left out for now (optional feature engineering later).
IDENTIFIER_COLUMNS: list[str] = [
    "owner_name", "shop_name", "address", "description", "attributes",
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

# --- Unit / currency quirks --------------------------------------------------
UNIT_FACTORS = {"sot": 100.0, "ha": 10_000.0}      # land area -> m^2
FX_TO_AZN = {"usd": 1.70, "$": 1.70}               # official AZN peg: 1.70/USD


# ============================================================================
# Parsing helpers
# ============================================================================
def _normalise_token(token: str) -> float:
    """Turn '125,000' / '125.000' / '85,5' / '1.250,75' into a float."""
    has_dot, has_comma = "." in token, "," in token
    if has_dot and has_comma:
        # the separator that appears LAST is the decimal mark
        dec = "." if token.rfind(".") > token.rfind(",") else ","
        thou = "," if dec == "." else "."
        token = token.replace(thou, "").replace(dec, ".")
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        parts = token.split(sep)
        # exactly one separator followed by 1-2 digits -> decimal mark;
        # otherwise (e.g. '125,000' or '1.250.000') -> thousands separator.
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            token = parts[0] + "." + parts[1]
        else:
            token = "".join(parts)
    return float(token)


def parse_number(value) -> float:
    """Extract the first number from messy text ('85.5 m²', '6 sot', 7)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).lower().replace("\xa0", " ").strip()
    factor = 1.0
    for unit, f in UNIT_FACTORS.items():
        if re.search(rf"\b{unit}\b", text):
            factor = f
    text = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", text)   # '125 000' -> '125000'
    match = re.search(r"\d+(?:[.,]\d+)*", text)
    if match is None:
        return np.nan
    return _normalise_number_safe(match.group()) * factor


def _normalise_number_safe(token: str) -> float:
    try:
        return _normalise_token(token)
    except ValueError:
        return np.nan


def parse_price(value) -> float:
    """Like parse_number, but converts USD-tagged prices to AZN."""
    number = parse_number(value)
    if isinstance(value, str):
        low = value.lower()
        for tag, rate in FX_TO_AZN.items():
            if tag in low:
                return number * rate
    return number


# ============================================================================
# Pipeline steps
# ============================================================================
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

def clean(df, verbose: bool = True):
    """Clean the raw frame; every step prints how many rows it changed."""
    df = df.copy()

    def log(step):
        if verbose:
            print(f"  [clean] {step:<42} -> {len(df):>7,} rows")

    log("start")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=RENAME)

    # 1. Drop leakage + identifier columns BEFORE deduplicating, otherwise a
    #    unique id/url column makes every row look different.
    drop = [c for c in LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS if c in df.columns]
    df = df.drop(columns=drop)
    log(f"dropped {len(drop)} leakage/identifier columns")

    # 2. Type coercion: text -> numbers.
    df["price"] = df["price"].map(parse_price)
    if "floor" in df.columns and not pd.api.types.is_numeric_dtype(df["floor"]):
        # bina.az shows floors as '5 / 9' = floor 5 of a 9-storey building.
        df["floor_total"] = pd.to_numeric(
            df["floor"].astype("string").str.extract(r"/\s*(\d+)")[0],
            errors="coerce",
        )
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(parse_number)

    # 3. Duplicates (reposted listings). Left in, a repost can land in train
    #    AND test, which silently inflates the test score.
    df = df.drop_duplicates()
    log("dropped exact duplicates")

    # 4. Rows we cannot use at all: missing target or area.
    df = df.dropna(subset=["price", "area_m2"])
    log("dropped rows with missing price/area")

    # 5. Outliers via fixed domain bounds (documented in the report).
    keep = df["price"].between(MIN_PRICE, MAX_PRICE) \
         & df["area_m2"].between(MIN_AREA, MAX_AREA)
    if "rooms" in df.columns:
        keep &= df["rooms"].isna() | df["rooms"].between(1, MAX_ROOMS)
    df = df[keep].copy()      # .copy(): avoids pandas SettingWithCopyWarning below
    log("dropped price/area/rooms outliers")

    # 6. Impossible coordinates become missing (row is kept).
    for col, (lo, hi) in (("lat", LAT_RANGE), ("lng", LNG_RANGE)):
        if col in df.columns:
            df[col] = df[col].where(df[col].between(lo, hi))

    return df.reset_index(drop=True)


def make_features(df):
    """Cleaned frame -> (X, y, feature_names). NaNs are LEFT in X on purpose:
    imputing needs train-only statistics, so it happens after the split."""
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    num = df[num_cols].astype(float).copy()
    if "land_area_m2" in num.columns:
        # NaN land area means "no land" (flats), not "unknown" -> 0 is correct
        # and needs no statistic from the data.
        num["land_area_m2"] = num["land_area_m2"].fillna(0.0)

    cats = df[cat_cols].astype("string").fillna("missing")
    for col in cat_cols:
        counts = cats[col].value_counts()
        rare = counts[counts < MIN_CAT_COUNT].index
        cats[col] = cats[col].where(~cats[col].isin(rare), "other")
    dummies = pd.get_dummies(cats, dtype=float)

    X_df = pd.concat([num, dummies], axis=1)
    return X_df.to_numpy(dtype=float), df["price"].to_numpy(dtype=float), \
        list(X_df.columns)


def make_tier_label(y_price, threshold=None):
    """premium=1 (price > threshold), standard=0. If no threshold is given it
    is the MEDIAN of y_price, so call it on TRAIN prices first and reuse the
    returned threshold for validation/test."""
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), threshold


def price_strata(y_price, n_bins: int = 4):
    """Quartile bin of each price: used ONLY to balance the split, never to
    define the tier threshold (that comes from the train median)."""
    edges = np.quantile(y_price, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(y_price, edges)


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED,
                         stratify=None):
    """Deterministic (optionally stratified) split.
    Returns (X_tr, y_tr, X_val, y_val, X_te, y_te)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    rng = np.random.default_rng(seed)
    groups = np.zeros(len(y), dtype=int) if stratify is None \
        else np.asarray(stratify)

    tr, va, te = [], [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        rng.shuffle(idx)
        n_te = int(round(len(idx) * test_size))
        n_va = int(round(len(idx) * val_size))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    tr, va, te = (rng.permutation(np.concatenate(p)) for p in (tr, va, te))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def impute_median(X_tr, *others):
    """Fill NaNs with the per-column median of the TRAIN set only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns
        med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def fill(A):
        A = np.array(A, dtype=float, copy=True)
        rows, cols = np.where(np.isnan(A))
        A[rows, cols] = med[cols]
        return A

    out = tuple(fill(A) for A in (X_tr,) + others)
    return out[0] if len(out) == 1 else out


def standardize(X_tr, *others):
    """z-score using TRAIN mean/std, applied to every split."""
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)       # constant column: avoid divide by 0
    out = tuple((np.asarray(A, dtype=float) - mu) / sd
                for A in (X_tr,) + others)
    return out[0] if len(out) == 1 else out


# ============================================================================
# `python -m src.data_prep` -> inspect + run the whole Problem-1 pipeline
# ============================================================================
if __name__ == "__main__":
    raw = load_raw()
    print(f"RAW: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print(pd.DataFrame({"dtype": raw.dtypes.astype(str),
                        "missing": raw.isna().sum(),
                        "example": raw.iloc[0]}).to_string())

    print("\nCleaning:")
    df = clean(raw)
    X, y, names = make_features(df)
    X_tr, y_tr, X_va, y_va, X_te, y_te = train_val_test_split(
        X, y, stratify=price_strata(y))
    X_tr, X_va, X_te = impute_median(X_tr, X_va, X_te)
    tier_tr, thr = make_tier_label(y_tr)
    tier_va, _ = make_tier_label(y_va, thr)
    tier_te, _ = make_tier_label(y_te, thr)

    print(f"\nFeatures: {X.shape[1]}  | split sizes  train/val/test = "
          f"{len(y_tr):,} / {len(y_va):,} / {len(y_te):,}")
    print(f"Tier threshold (TRAIN median): {thr:,.0f} AZN")
    for name, t in (("train", tier_tr), ("val", tier_va), ("test", tier_te)):
        print(f"  premium share in {name:<5}: {t.mean():.3f}")
