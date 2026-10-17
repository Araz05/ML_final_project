import os
 
import numpy as np
import pandas as pd
import pytest
 
from src import data_prep as dp
 
 
# ---------------------------------------------------------------- fixtures --
def _raw(n=400, seed=0):
    """Synthetic frame with the same quirks as the real bina.az dump."""
    rng = np.random.default_rng(seed)
    area = rng.uniform(30, 250, n).round(0)
    price = (area * rng.uniform(1500, 3000, n)).round(0)
    df = pd.DataFrame({
        "estate_rel_url_x": [f"/items/{i}" for i in range(n)],
        "datetime_scrape_x": ["2024-01-01"] * n,
        "price": price,
        "currency_x": "AZN",
        "unit_price": [f"{p / a:.0f} AZN/m²" for p, a in zip(price, area)],
        "total_price": price,
        "owner_name": "someone",
        "city": rng.choice(["bakı", "xırdalan"], n),
        "location": rng.choice(["A m.", "B m.", "C q."], n),
        "lat": rng.uniform(40.3, 40.5, n),
        "lng": rng.uniform(49.7, 50.0, n),
        "repair": "Təmirli",                                 # listing-level twin
        "Kateqoriya": rng.choice(["Yeni tikili", "Köhnə tikili"], n),
        "Mərtəbə": [f"{rng.integers(1, 9)} / 9" for _ in range(n)],
        "Otaq sayı": rng.integers(1, 6, n).astype(float),
        "Sahə": [f"{a:.0f} m²" for a in area],
        "Torpaq sahəsi": [np.nan] * n,
        "Təmir": rng.choice(["var", "yoxdur", None], n),
        "İpoteka": rng.choice(["var", None], n),
        "Çıxarış": rng.choice(["var", "yoxdur"], n),
        "owner_title": rng.choice(["mülkiyyətçi", "vasitəçi (agent)"], n),
    })
    return df
 
 
def _clean(df=None):
    return dp.clean(_raw() if df is None else df, verbose=False)
 
 
# ------------------------------------------------------------------- clean --
def test_clean_drops_leakage_and_identifiers():
    out = _clean()
    for c in dp.LEAKAGE_COLUMNS + ["owner_name", "estate_rel_url_x"]:
        assert c not in out.columns
 
 
def test_clean_keeps_latest_scrape_of_reposts():
    df = _raw(50)
    dup = df.iloc[:10].copy()
    dup["datetime_scrape_x"] = "2025-06-01"
    dup["price"] = dup["price"] * 2                 # the price changed later
    out = dp.clean(pd.concat([df, dup], ignore_index=True), verbose=False)
    assert len(out) == 50                            # one row per listing
    stale = df.iloc[:10]["price"].to_numpy()
    assert np.isin(out["price"], stale * 2).sum() == 10
 
 
def test_clean_outliers_and_missing_rows_removed():
    df = _raw(60)
    df.loc[0, "price"] = 11.0                        # absurdly cheap
    df.loc[1, "price"] = 6e8                         # absurdly expensive
    df.loc[2, "price"] = np.nan                      # missing target
    df.loc[3, "Sahə"] = None                         # missing area
    df.loc[4, "Otaq sayı"] = 99.0                    # impossible rooms
    out = dp.clean(df, verbose=False)
    assert len(out) == 55
    assert out["price"].between(dp.MIN_PRICE, dp.MAX_PRICE).all()
 
 
def test_clean_bad_coordinates_become_nan_not_dropped():
    df = _raw(60)
    df.loc[0, "lat"], df.loc[0, "lng"] = 32.7, 12.6  # not in Azerbaijan
    out = dp.clean(df, verbose=False)
    assert len(out) == 60
    assert out["lat"].isna().sum() == 1 and out["lng"].isna().sum() == 1
 
 
# ----------------------------------------------------------- make_features --
def test_make_features_shapes_and_no_target_leak():
    out = _clean()
    X, y, names = dp.make_features(out)
    assert X.shape == (len(out), len(names)) and y.shape == (len(out),)
    assert "price" not in names
    assert not any(n.startswith(("unit_price", "total_price")) for n in names)
    assert X.dtype == float
    # land_area NaN means "no land" -> 0, never NaN
    assert not np.isnan(X[:, names.index("land_area")]).any()
 
 
def test_rare_categories_merged_into_other():
    df = _clean(_raw(400))
    df.loc[df.index[:3], "location"] = "Rare place"
    _, _, names = dp.make_features(df)
    assert "location_Rare place" not in names
    assert "location_other" in names

# -------------------------------------------------------------------- split --
def test_split_is_disjoint_complete_and_deterministic():
    X = np.arange(1000, dtype=float).reshape(-1, 1)
    y = np.arange(1000)
    a = dp.train_val_test_split(X, y, seed=1)
    b = dp.train_val_test_split(X, y, seed=1)
    c = dp.train_val_test_split(X, y, seed=2)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert not np.array_equal(a[0], c[0])
    tr, va, te = a[1], a[3], a[5]
    assert len(set(tr) & set(va)) == len(set(tr) & set(te)) == len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == 1000
    assert abs(len(va) - 150) <= 2 and abs(len(te) - 150) <= 2

def test_stratified_split_balances_groups():
    rng = np.random.default_rng(0)
    y = rng.lognormal(12, 1, 4000)
    X = np.arange(len(y), dtype=float).reshape(-1, 1)
    strata = dp.price_strata(y)
    _, ytr, _, yva, _, yte = dp.train_val_test_split(X, y, stratify=strata)
    med = np.median(y)
    shares = [(s > med).mean() for s in (ytr, yva, yte)]
    assert max(shares) - min(shares) < 0.02
 
 
# ------------------------------------------------- tier label / no leakage --
def test_tier_threshold_comes_from_train_only():
    y_tr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tier_tr, thr = dp.make_tier_label(y_tr)
    assert thr == 3.0 and tier_tr.tolist() == [0, 0, 0, 1, 1]
    # val/test reuse the TRAIN threshold, they must not recompute their own
    tier_te, thr2 = dp.make_tier_label(np.array([10.0, 20.0, 30.0]), thr)
    assert thr2 == thr and tier_te.tolist() == [1, 1, 1]
 
 
def test_impute_uses_train_median_only():
    X_tr = np.array([[1.0], [2.0], [np.nan], [3.0]])       # median 2
    X_te = np.array([[np.nan], [100.0], [200.0]])           # test median would be 150
    a, b = dp.impute_median(X_tr, X_te)
    assert a[2, 0] == 2.0 and b[0, 0] == 2.0
    assert not np.isnan(a).any() and not np.isnan(b).any()
 
 
def test_standardize_uses_train_statistics():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(5, 3, size=(500, 4))
    X_te = rng.normal(50, 3, size=(100, 4))
    a, b = dp.standardize(X_tr, X_te)
    assert np.allclose(a.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(a.std(axis=0), 1, atol=1e-9)
    assert b.mean() > 10                      # test is NOT re-centred on itself
 
 
def test_standardize_constant_column_does_not_divide_by_zero():
    X = np.ones((10, 2))
    assert np.isfinite(dp.standardize(X)).all()
 
 
# ------------------------------------------------------ real data (if any) --
@pytest.mark.skipif(not os.path.exists(dp.DATA_PATH), reason="real CSV not present")
def test_real_csv_full_pipeline():
    df = dp.clean(dp.load_raw(), verbose=False)
    assert 50_000 < len(df) < 100_000
    assert df["price"].between(dp.MIN_PRICE, dp.MAX_PRICE).all()
    assert not df.columns.duplicated().any()
 
    X, y, names = dp.make_features(df)
    Xtr, ytr, Xva, yva, Xte, yte = dp.train_val_test_split(
        X, y, stratify=dp.price_strata(y))
    Xtr, Xva, Xte = dp.impute_median(Xtr, Xva, Xte)
    assert not np.isnan(Xtr).any() and not np.isnan(Xte).any()
 
    _, thr = dp.make_tier_label(ytr)
    shares = [dp.make_tier_label(s, thr)[0].mean() for s in (ytr, yva, yte)]
    assert all(0.45 < s < 0.55 for s in shares)
 
    Ztr, Zva = dp.standardize(Xtr, Xva)
    assert np.allclose(Ztr.mean(axis=0), 0, atol=1e-6)