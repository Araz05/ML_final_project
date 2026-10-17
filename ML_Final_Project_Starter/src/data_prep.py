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
import warnings

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
    "owner_name", "shop_name", "shop_title", "address", "description",
    "attributes", "extra_info", "img_url", "rel_url",
    "id_x", "id_y", "estate_id", "estate_details_id_x", "estate_details_id_y",
    "estate_rel_url", "estate_rel_url_x", "estate_rel_url_y",
    "datetime_scrape_x", "datetime_scrape_y",
]

# Azerbaijani names, folded to ASCII (see _fold), -> English names.
# NOTE: the raw dump has BOTH a listing-level column and a detail-level column
# for repair / mortgage / bill_of_sale (e.g. `repair` = "Təmirli"/NaN and
# `Təmir` = "var"/"yoxdur"/NaN). The detail-level Azerbaijani column wins the
# clean English name; the listing-level twin is kept as `<name>_listing`.
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

# Columns that must be numeric after cleaning (names AFTER renaming).
# `floor` ("7 / 9") is split into `floor` and `floor_total` by coerce_numeric.
NUMERIC_COLS = ["price", "area", "rooms", "land_area", "lat", "lng"]

CURRENCY_COL_CANDIDATES = ("currency_x", "currency_y", "currency", "valyuta")

# Everything is converted to AZN. 1 USD = 1.70 AZN is the official fixed rate.
FX_TO_AZN = {"AZN": 1.0, "USD": 1.70, "EUR": 1.85}

# Land plots are listed in "sot" (1 sot = 100 m^2). ALL areas are stored in m^2.
SOT_TO_M2 = 100.0

_AZ_FOLD = str.maketrans({
    "ə": "e", "Ə": "e", "ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s",
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ö": "o", "Ö": "o",
    "ü": "u", "Ü": "u",
})

# --- Outlier rules (fixed domain bounds, justify them in the report) --------
MIN_PRICE, MAX_PRICE = 5_000.0, 10_000_000.0   # AZN  (~0.1% / 99.9% quantiles)
MIN_AREA, MAX_AREA = 10.0, 100_000.0           # m^2  (100,000 m^2 = 10 ha plot)
MAX_ROOMS = 15
LAT_RANGE = (38.3, 41.95)                      # Azerbaijan bounding box
LNG_RANGE = (44.7, 50.9)

# --- Features ---------------------------------------------------------------
NUMERIC_FEATURES = ["area", "rooms", "floor", "floor_total", "land_area", "lat", "lng"]
CATEGORICAL_FEATURES = ["category", "repair", "mortgage", "bill_of_sale",
                        "city", "location", "owner_title"]
MIN_CAT_COUNT = 100        # rarer category levels are merged into "other"


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
    Azerbaijani names are translated via RENAME_MAP. If a translated name
    collides with an existing English column (repair, mortgage, bill_of_sale),
    the pre-existing listing-level column becomes `<name>_listing`.
    """
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
    return "AZN"   # AZN, manat, \u20bc, or nothing written


def normalize_currency(df, price_col: str = "price"):
    """
    Convert prices to AZN. Run it BEFORE coerce_numeric (that function would
    strip the currency text). The currency is read from the first column in
    CURRENCY_COL_CANDIDATES that exists, otherwise from the price string.
    The original currency is kept in `price_currency_orig` for the report.
    (In the bina.az dump every row is AZN, so prices come out unchanged.)
    """
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


def drop_reposts(df):
    """
    The dump contains the same listing scraped several times (same
    `estate_rel_url_x`, ~36k extra rows, price changed in ~16% of them). Keep
    only the MOST RECENT scrape of each listing. Left in, a repost can land in
    train AND test and silently inflate the test score.
    """
    if "estate_rel_url_x" not in df.columns:
        return df
    if "datetime_scrape_x" in df.columns:
        order = pd.to_datetime(df["datetime_scrape_x"], errors="coerce")
        df = df.assign(_t=order).sort_values("_t", kind="stable").drop(columns="_t")
    return df.drop_duplicates(subset="estate_rel_url_x", keep="last")


def clean(df, verbose: bool = True):
    """Clean the raw frame; every step prints how many rows it changed."""
    df = df.copy()

    def log(step):
        if verbose:
            print(f"  [clean] {step:<42} -> {len(df):>7,} rows")

    log("start")
    df = standardize_columns(df)

    # 1. Reposts first (needs the url + scrape-time columns, dropped later).
    df = drop_reposts(df)
    log("kept latest scrape per listing (reposts)")

    # 2. Drop leakage + identifier columns.
    drop = [c for c in LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS if c in df.columns]
    df = df.drop(columns=drop)
    log(f"dropped {len(drop)} leakage/identifier columns")

    # 3. Types: currency -> AZN (before numeric parsing), text -> numbers.
    df = normalize_currency(df)
    df = coerce_numeric(df)

    # 4. Exact duplicates that remain.
    df = df.drop_duplicates()
    log("dropped exact duplicates")

    # 5. Rows we cannot use at all: missing target or area.
    df = df.dropna(subset=["price", "area"])
    log("dropped rows with missing price/area")

    # 6. Outliers via fixed domain bounds (documented in the report).
    keep = df["price"].between(MIN_PRICE, MAX_PRICE) \
         & df["area"].between(MIN_AREA, MAX_AREA)
    if "rooms" in df.columns:
        keep &= df["rooms"].isna() | df["rooms"].between(1, MAX_ROOMS)
    df = df[keep].copy()
    log("dropped price/area/rooms outliers")

    # 7. Impossible coordinates become missing (row is kept).
    for col, (lo, hi) in (("lat", LAT_RANGE), ("lng", LNG_RANGE)):
        if col in df.columns:
            df[col] = df[col].where(df[col].between(lo, hi))

    return df.reset_index(drop=True)


def fit_category_levels(df):
    """Which category levels are frequent enough to keep? Returns
    {column: set(levels)}. Call it on the TRAIN rows only: rare/unseen levels
    in any split are later merged into "other"."""
    levels = {}
    for col in (c for c in CATEGORICAL_FEATURES if c in df.columns):
        cats = df[col].astype("string").fillna("missing")
        counts = cats.value_counts()
        levels[col] = set(counts[counts >= MIN_CAT_COUNT].index)
    return levels


def make_features(df, levels=None, columns=None):
    """
    Cleaned frame -> (X, y, feature_names).

    NaNs are LEFT in X on purpose: imputing needs train-only statistics, so it
    happens after the split. To avoid using test-set statistics here too:
      * `levels`  = output of fit_category_levels(train_df). If None, the
        levels are computed from `df` itself (fine for exploration / one frame).
      * `columns` = the feature names of the TRAIN design matrix; val/test
        matrices are reindexed to exactly these columns (unseen -> all zeros).
    """
    levels = fit_category_levels(df) if levels is None else levels
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    num = df[num_cols].astype(float).copy()
    if "land_area" in num.columns:
        # NaN land area means "no land" (flats), not "unknown" -> 0 is correct
        # and needs no statistic from the data.
        num["land_area"] = num["land_area"].fillna(0.0)

    cats = df[cat_cols].astype("string").fillna("missing")
    for col in cat_cols:
        keep = levels.get(col, set())
        cats[col] = cats[col].where(cats[col].isin(keep), "other")
    dummies = pd.get_dummies(cats, dtype=float)

    X_df = pd.concat([num, dummies], axis=1)
    if columns is not None:
        X_df = X_df.reindex(columns=columns, fill_value=0.0)
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
    X_tr = np.asarray(X_tr, dtype=float)
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)       # constant column: avoid divide by 0
    out = tuple((np.asarray(A, dtype=float) - mu) / sd
                for A in (X_tr,) + others)
    return out[0] if len(out) == 1 else out


def split_and_featurize(df, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Leakage-safe path from a CLEANED frame to design matrices:
      1. split the ROWS (stratified on price quartiles),
      2. learn the category levels from the TRAIN rows only,
      3. build X for each split with those levels / columns.
    NaNs are left in X (use impute_median next). Returns a dict with
    Xtr, ytr, Xva, yva, Xte, yte, names, levels, df_tr, df_va, df_te.
    """
    y_all = df["price"].to_numpy(dtype=float)
    ids = np.arange(len(df), dtype=float)
    i_tr, _, i_va, _, i_te, _ = train_val_test_split(
        ids, y_all, val_size, test_size, seed, stratify=price_strata(y_all))
    df_tr, df_va, df_te = (df.iloc[a.astype(int)] for a in (i_tr, i_va, i_te))
    levels = fit_category_levels(df_tr)
    Xtr, ytr, names = make_features(df_tr, levels)
    Xva, yva, _ = make_features(df_va, levels, names)
    Xte, yte, _ = make_features(df_te, levels, names)
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte, names=names,
                levels=levels, df_tr=df_tr, df_va=df_va, df_te=df_te)


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
    S = split_and_featurize(df)
    y_tr, y_va, y_te = S["ytr"], S["yva"], S["yte"]
    X_tr, X_va, X_te = impute_median(S["Xtr"], S["Xva"], S["Xte"])
    tier_tr, thr = make_tier_label(y_tr)
    tier_va, _ = make_tier_label(y_va, thr)
    tier_te, _ = make_tier_label(y_te, thr)

    print(f"\nFeatures: {X_tr.shape[1]}  | split sizes  train/val/test = "
          f"{len(y_tr):,} / {len(y_va):,} / {len(y_te):,}")
    print(f"Tier threshold (TRAIN median): {thr:,.0f} AZN")
    for name, t in (("train", tier_tr), ("val", tier_va), ("test", tier_te)):
        print(f"  premium share in {name:<5}: {t.mean():.3f}")