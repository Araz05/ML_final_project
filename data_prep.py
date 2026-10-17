"""Data loading, cleaning and category-specific preparation for house-sale ML.

Preprocessing that learns from data (imputation/encoding/scaling) must be fitted
on training data only. The helpers below preserve that separation.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = "data/house_sale.csv"
LEAKAGE_COLUMNS: list[str] = ["unit_price", "total_price", "log_price"]

CATEGORY_FEATURES = {
    "Yeni tikili": ["area_m2", "rooms", "floor", "repair", "lat", "lng"],
    "Köhnə tikili": ["area_m2", "rooms", "floor", "repair", "lat", "lng"],
    "Həyət evi/Bağ evi": ["area_m2", "land_area_m2", "rooms", "repair", "lat", "lng"],
    "Torpaq": ["area_m2", "lat", "lng"],
    "Obyekt": ["area_m2", "repair", "lat", "lng"],
    "Ofis": ["area_m2", "rooms", "repair", "lat", "lng"],
    "Qaraj": ["area_m2", "lat", "lng"],
}


def convert_area_to_m2(value):
    """Convert strings like '12 sot', '145 m²' and '50,3 sot' to m²."""
    if pd.isna(value):
        return np.nan
    value = str(value).lower().strip().replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", value)
    if match is None:
        return np.nan
    number = float(match.group())
    if "sot" in value:
        return number * 100
    if "m²" in value or "m2" in value or "m^2" in value or "kv.m" in value:
        return number
    return np.nan


def load_raw(path: str = DATA_PATH):
    """Read the CSV without changing its contents."""
    return pd.read_csv(path, encoding="utf-8")


def clean(df):
    """Deterministic row-level cleaning, without train-fitted transformations.

    Rule: remove invalid/nonpositive price or area, and duplicate estate_id.
    Do not automatically remove IQR outliers: expensive properties can be real.
    """
    data = df.copy()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["area_m2"] = data["Sahə"].apply(convert_area_to_m2)
    data["land_area_m2"] = data["Torpaq sahəsi"].apply(convert_area_to_m2)
    data = data.loc[
        data["price"].notna() & (data["price"] > 0)
        & data["area_m2"].notna() & (data["area_m2"] > 0)
    ].copy()
    if "estate_id" in data.columns:
        data = data.drop_duplicates(subset=["estate_id"], keep="first")
    data = data.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
    return data


def load_and_clean_data(file_path=DATA_PATH):
    """Compatibility helper used in the interactive notebook/terminal steps."""
    raw = load_raw(file_path)
    print("Original dataset:", raw.shape)
    data = clean(raw)
    # Remove the explicitly excluded 600 and 60-million-AZN listing after price cleaning.
    before = len(data)

    excluded_prices = [600_000_000, 60_000_000]
    data = data[~data["price"].isin(excluded_prices)].copy()

    print("Removed exceptional listings:", before - len(data))
    print("Cleaned dataset:", data.shape)
    data["log_price"] = np.log(data["price"])
    print("\nLog-price statistics:")
    print(data["log_price"].describe())
    return data


def split_dataset(df, random_state=SEED):
    """Group-aware 70/15/15 split; identical listing URLs stay together.

    Uses item-page URLs only. Missing/non-item URLs get independent groups.
    This is conservative: URL reuse may group distinct properties, but cannot
    create cross-split overlap for the same item URL.
    """
    from sklearn.model_selection import GroupShuffleSplit

    url_col = next((c for c in ("estate_rel_url", "estate_rel_url_x", "estate_rel_url_y")
                    if c in df.columns), None)
    if url_col is None:
        raise ValueError("No listing URL column available for group-aware splitting")

    urls = df[url_col].astype("string").str.strip().str.lower()
    valid = urls.str.contains(r"(?:^|/)items/\d+(?:[/?#]|$)", regex=True, na=False)
    # Use the listing ID from the URL, excluding query parameters.
    item_ids = urls.str.extract(r"(?:^|/)items/(\d+)", expand=False)
    # Unique fallback prevents all missing URLs from becoming one giant group.
    groups = [f"item:{item_ids.iloc[i]}" if bool(valid.iloc[i]) else f"row:{i}"
              for i in range(len(df))]

    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    train_idx, temp_idx = next(first.split(df, groups=groups))
    temp_groups = [groups[i] for i in temp_idx]
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    val_rel, test_rel = next(second.split(df.iloc[temp_idx], groups=temp_groups))
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[temp_idx[val_rel]].copy()
    test_df = df.iloc[temp_idx[test_rel]].copy()

    def group_set(indices):
        return {groups[i] for i in indices}
    train_g = group_set(train_idx)
    val_g = group_set(temp_idx[val_rel])
    test_g = group_set(temp_idx[test_rel])
    assert train_g.isdisjoint(val_g) and train_g.isdisjoint(test_g) and val_g.isdisjoint(test_g)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    print(f"Train: {train_df.shape} | Validation: {val_df.shape} | Test: {test_df.shape}")
    print(f"Grouped by listing URL: {sum(valid)} valid URLs, "
          f"{len(set(groups))} groups, zero group overlap")
    return train_df, val_df, test_df


def split_by_category(df):
    """Return a separate DataFrame for each property category."""
    groups = {}
    for category, group in df.groupby("Kateqoriya"):
        groups[category] = group.copy()
        print(f"{category}: {len(group)} elan")
    return groups


def analyze_categories(category_datasets):
    columns = ["Sahə", "Torpaq sahəsi", "Otaq sayı", "Mərtəbə", "Təmir", "lat", "lng"]
    for category, frame in category_datasets.items():
        print("\n" + "=" * 50)
        print("Kateqoriya:", category)
        print("Elan sayı:", len(frame))
        print("=" * 50)
        print("Boş məlumatlar:")
        print(frame[columns].isna().sum().to_string())
        print("\nQiymət medianı:", round(frame["price"].median(), 2))


def prepare_category_features(df, category):
    """Return raw features and log-price target for one property category.

    Missing values and categorical encoding are handled *after splitting*.
    """
    if category not in CATEGORY_FEATURES:
        raise ValueError(f"Naməlum əmlak kateqoriyası: {category}")
    data = df.copy()
    data["rooms"] = pd.to_numeric(data["Otaq sayı"], errors="coerce")
    data["floor"] = pd.to_numeric(
        data["Mərtəbə"].astype(str).str.extract(r"^\s*(\d+)")[0], errors="coerce"
    )
    data["repair"] = data["Təmir"].fillna("Unknown").astype(str)
    features = CATEGORY_FEATURES[category]
    X = data[features].copy()
    y = data["log_price"].copy() if "log_price" in data else np.log(data["price"])
    return X, y


def select_features(df):
    """Legacy single-model feature selection (not recommended for category models)."""
    data = df.copy()
    data["rooms"] = pd.to_numeric(data["Otaq sayı"], errors="coerce")
    data["floor"] = pd.to_numeric(
        data["Mərtəbə"].astype(str).str.extract(r"^\s*(\d+)")[0], errors="coerce"
    )
    data["repair"] = data["Təmir"].fillna("Unknown").astype(str)
    data["category"] = data["Kateqoriya"].astype(str)
    data["location"] = data["location"].fillna("Unknown").astype(str)
    features = ["area_m2", "rooms", "floor", "lat", "lng", "category", "repair", "location"]
    y = data["log_price"].copy() if "log_price" in data else np.log(data["price"])
    return data[features].copy(), y


def make_features(df):
    """Starter API: numeric X, raw AZN y, feature names.

    For strict leakage-free experiments, split raw rows first and use
    fit_feature_encoder on training X; apply_feature_encoder on validation/test.
    """
    X_raw, _ = select_features(df)
    numeric = ["area_m2", "rooms", "floor", "lat", "lng"]
    X = X_raw[numeric].copy()
    for col in numeric:
        X[col] = X[col].fillna(X[col].median())
    cats = pd.get_dummies(X_raw[["category", "repair", "location"]], dtype=float)
    X = pd.concat([X, cats], axis=1)
    return X.to_numpy(dtype=float), df["price"].to_numpy(dtype=float), list(X.columns)


def make_tier_label(y_price, threshold=None):
    """Fit threshold from training y only; reuse supplied threshold elsewhere."""
    y = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y))
    return (y > threshold).astype(int), float(threshold)


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """Return X_tr,y_tr,X_val,y_val,X_te,y_te (70/15/15 by default)."""
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to < 1")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=seed
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=test_size / (val_size + test_size), random_state=seed
    )
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def standardize(X_tr, *others):
    """Scale using training means and standard deviations only."""
    train = np.asarray(X_tr, dtype=float)
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    result = [(train - means) / stds]
    result.extend((np.asarray(X, dtype=float) - means) / stds for X in others)
    return tuple(result)


def fit_feature_encoder(X_train):
    """Learn numeric fill values and categorical columns on training data only."""
    numeric = list(X_train.select_dtypes(include="number").columns)
    categorical = [col for col in X_train if col not in numeric]
    medians = X_train[numeric].median().fillna(0).to_dict()
    cat_columns = list(pd.get_dummies(
        X_train[categorical].fillna("Unknown").astype(str), dtype=float
    ).columns) if categorical else []
    return {"numeric": numeric, "categorical": categorical,
            "medians": medians, "cat_columns": cat_columns}


def apply_feature_encoder(X, encoder):
    """Apply a train-fitted encoder and return a numeric NumPy array."""
    numeric = encoder["numeric"]
    categorical = encoder["categorical"]
    nums = X[numeric].copy().fillna(encoder["medians"]).astype(float)
    if categorical:
        cats = pd.get_dummies(X[categorical].fillna("Unknown").astype(str), dtype=float)
        cats = cats.reindex(columns=encoder["cat_columns"], fill_value=0)
        nums = pd.concat([nums, cats], axis=1)
    return nums.to_numpy(dtype=float)
def split_all_categories(category_datasets):
    result = {}

    for category, df in category_datasets.items():
        train_df, val_df, test_df = split_dataset(df)

        result[category] = {
            "train": train_df,
            "validation": val_df,
            "test": test_df,
        }

        print(
            f"{category}: "
            f"Train={len(train_df)}, "
            f"Validation={len(val_df)}, "
            f"Test={len(test_df)}"
        )

    return result
def encode_category_features(X_train, X_val, X_test):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Mətn sütunlarını yalnız train məlumatlarından müəyyən edirik
    categorical_columns = X_train.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    # Hər üç hissəni eyni qaydada kodlaşdırırıq
    X_train = pd.get_dummies(
        X_train, columns=categorical_columns, dtype=int
    )
    X_val = pd.get_dummies(
        X_val, columns=categorical_columns, dtype=int
    )
    X_test = pd.get_dummies(
        X_test, columns=categorical_columns, dtype=int
    )

    # Validation və test sütunlarını train ilə uyğunlaşdırırıq
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_val, X_test
def fill_missing_numeric(X_train, X_val, X_test):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    numeric_columns = X_train.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        median_value = X_train[column].median()

        # Train sütunu tamamilə boşdursa, ehtiyat qiymət
        if pd.isna(median_value):
            median_value = 0.0

        X_train[column] = X_train[column].fillna(median_value)
        X_val[column] = X_val[column].fillna(median_value)
        X_test[column] = X_test[column].fillna(median_value)

    return X_train, X_val, X_test