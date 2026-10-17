"""NumPy evaluation metrics and fair model comparison helpers.

Run the regression comparison from the project root:
    python -m src.evaluate
"""
from __future__ import annotations

import numpy as np


def _paired(y_true, y_pred):
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if a.size == 0 or a.shape != b.shape:
        raise ValueError('Expected equally sized, nonempty arrays')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Metric inputs must be finite')
    return a, b


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    a, b = _paired(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true, y_pred) -> float:
    a, b = _paired(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def r2(y_true, y_pred) -> float:
    a, b = _paired(y_true, y_pred)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def median_absolute_error(y_true, y_pred) -> float:
    a, b = _paired(y_true, y_pred)
    return float(np.median(np.abs(a - b)))


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return [[TN, FP], [FN, TP]] for binary labels 0/1."""
    a, b = _paired(y_true, y_pred)
    if not (np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()):
        raise ValueError('Binary labels must be 0 or 1')
    tn = int(np.sum((a == 0) & (b == 0)))
    fp = int(np.sum((a == 0) & (b == 1)))
    fn = int(np.sum((a == 1) & (b == 0)))
    tp = int(np.sum((a == 1) & (b == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def precision_recall_f1(y_true, y_pred):
    """Return precision, recall, F1 for positive class 1 (premium)."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def roc_auc(y_true, scores) -> float:
    """ROC AUC using average ranks to handle tied decision scores."""
    labels, values = _paired(y_true, scores)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Binary labels must be 0 or 1')
    n_pos = int(np.sum(labels == 1))
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float('nan')
    order = np.argsort(values, kind='mergesort')
    sorted_scores = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((np.sum(ranks[labels == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """Print a compact table of models evaluated on the same split."""
    if not name_to_metrics:
        print('No models to compare.')
        return
    metric_names = list(dict.fromkeys(k for metrics in name_to_metrics.values() for k in metrics))
    columns = ['Model', *metric_names]
    rows = []
    for name, metrics in name_to_metrics.items():
        rows.append([str(name), *[f'{metrics[k]:,.2f}' if k in metrics else '-' for k in metric_names]])
    widths = [max(len(str(c)), *(len(r[i]) for r in rows)) for i, c in enumerate(columns)]
    print(' | '.join(str(c).ljust(widths[i]) for i, c in enumerate(columns)))
    print('-+-'.join('-' * w for w in widths))
    for row in rows:
        print(' | '.join(value.ljust(widths[i]) for i, value in enumerate(row)))


def run_decision_tree_comparison():
    """Compare the NumPy regression tree with sklearn on Yeni tikili validation data."""
    from sklearn.tree import DecisionTreeRegressor
    from src.data_prep import (
        load_and_clean_data, split_by_category, split_dataset,
        prepare_category_features, encode_category_features, fill_missing_numeric,
    )
    from src.decision_tree import DecisionTree

    df = load_and_clean_data('data/house_sale.csv')
    category = 'Yeni tikili'
    train, val, test = split_dataset(split_by_category(df)[category])
    xtr, ytr = prepare_category_features(train, category)
    xva, yva = prepare_category_features(val, category)
    xte, _ = prepare_category_features(test, category)
    xtr, xva, xte = encode_category_features(xtr, xva, xte)
    xtr, xva, xte = fill_missing_numeric(xtr, xva, xte)
    X_train = xtr.to_numpy(dtype=float)
    X_val = xva.to_numpy(dtype=float)
    y_train = ytr.to_numpy(dtype=float)
    actual_price = np.exp(yva.to_numpy(dtype=float))

    own = DecisionTree(task='regression', criterion='mse', max_depth=20,
                       min_samples_split=20, min_samples_leaf=2, random_state=42)
    baseline = DecisionTreeRegressor(criterion='squared_error', max_depth=20,
                                     min_samples_split=20, min_samples_leaf=2,
                                     random_state=42)
    own.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    predictions = {
        'Our NumPy Decision Tree': np.exp(own.predict(X_val)),
        'Sklearn Decision Tree': np.exp(baseline.predict(X_val)),
    }
    results = {
        name: {
            'MAE (AZN)': mae(actual_price, pred),
            'RMSE (AZN)': rmse(actual_price, pred),
            'Median AE (AZN)': median_absolute_error(actual_price, pred),
            'R2 (AZN)': r2(actual_price, pred),
        }
        for name, pred in predictions.items()
    }
    print('\nDECISION TREE REGRESSION COMPARISON — Yeni tikili (validation)')
    compare(results)
    return results




# --- Final held-out evaluation (Step 36) ------------------------------------
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep
from .decision_tree import DecisionTree
from .svm import PegasosSVM
TREE_PARAMS = dict(max_depth=20, min_samples_split=20, min_samples_leaf=2, random_state=42)
PEGASOS_PARAMS = dict(lambda_=0.001, n_iters=300_000, random_state=42)
SVC_PARAMS = dict(C=0.1, loss="squared_hinge", dual=False, tol=1e-4, max_iter=50_000, random_state=42)

OUT = Path(__file__).resolve().parent.parent / 'report'


def verify_group_isolation(train, val, test):
    """Independently verify that canonical /items/ IDs cannot cross splits."""
    col = next((c for c in ('estate_rel_url', 'estate_rel_url_x', 'estate_rel_url_y')
                if c in train.columns), None)
    if col is None:
        raise RuntimeError('Missing listing URL: cannot verify group-aware split')

    def ids(frame):
        s = frame[col].astype('string').str.strip().str.lower()
        return set(s.str.extract(r'(?:^|/)items/(\d+)(?:[/?#]|$)', expand=False).dropna())

    a, b, c = ids(train), ids(val), ids(test)
    overlap = (len(a & b), len(a & c), len(b & c))
    if any(overlap):
        raise RuntimeError(f'LEAKAGE CHECK FAILED: item ID overlaps={overlap}. '
                           'Ensure src/data_prep.py is the group-aware version.')
    # Also verify that no exact DataFrame row is reused between partitions.
    if not (set(train.index).isdisjoint(val.index)
            and set(train.index).isdisjoint(test.index)
            and set(val.index).isdisjoint(test.index)):
        raise RuntimeError('LEAKAGE CHECK FAILED: overlapping source row indices')
    return {'item_url_groups_train': len(a), 'item_url_groups_validation': len(b),
            'item_url_groups_test': len(c), 'cross_split_item_id_overlaps': 0}


def final_regression_metrics(y_true, pred):
    return {'MAE_AZN': mae(y_true, pred),
            'RMSE_AZN': rmse(y_true, pred),
            'MedianAE_AZN': median_absolute_error(y_true, pred),
            'R2': r2(y_true, pred)}


def final_classification_metrics(y_true, pred, scores):
    precision, recall, f1 = precision_recall_f1(y_true, pred)
    return {'Accuracy': float(np.mean(y_true == pred)),
            'Precision': float(precision), 'Recall': float(recall),
            'F1': float(f1), 'ROC_AUC': float(roc_auc(y_true, scores)),
            'Confusion_Matrix': confusion_matrix(y_true, pred).tolist()}


def score_category(category, frame):
    train, val, test = data_prep.split_dataset(frame)
    group_check = verify_group_isolation(train, val, test)
    # Keep the original held-out test partition; refit on train + validation.
    fit_rows = pd.concat([train, val], axis=0)
    x_fit_raw, log_fit = data_prep.prepare_category_features(fit_rows, category)
    x_test_raw, log_test = data_prep.prepare_category_features(test, category)
    x_fit, x_test, _ = data_prep.encode_category_features(
        x_fit_raw, x_test_raw, x_fit_raw.iloc[:1].copy())
    x_fit, x_test, _ = data_prep.fill_missing_numeric(
        x_fit, x_test, x_fit.iloc[:1].copy())
    x_fit = x_fit.to_numpy(dtype=float)
    x_test = x_test.to_numpy(dtype=float)
    if not np.isfinite(x_fit).all() or not np.isfinite(x_test).all():
        raise RuntimeError(f'Non-finite features for {category}')
    log_fit = log_fit.to_numpy(dtype=float)
    actual = np.exp(log_test.to_numpy(dtype=float))
    fit_prices = np.exp(log_fit)
    threshold = float(np.median(fit_prices))
    labels_fit = (fit_prices > threshold).astype(int)
    labels_test = (actual > threshold).astype(int)
    common = {'category': category, 'n_total': len(frame),
              'n_train': len(train), 'n_validation': len(val), 'n_test': len(test),
              'premium_threshold_AZN': threshold, **group_check}
    rows = []

    regression_models = [
        ('NumPy Decision Tree', DecisionTree(task='regression', criterion='mse', **TREE_PARAMS)),
        ('Sklearn Decision Tree', DecisionTreeRegressor(criterion='squared_error', **TREE_PARAMS)),
    ]
    for name, model in regression_models:
        model.fit(x_fit, log_fit)
        pred = np.exp(model.predict(x_test))
        rows.append({**common, 'task': 'regression', 'model': name,
                     **final_regression_metrics(actual, pred)})

    if np.unique(labels_fit).size < 2:
        print(f'  {category}: classification skipped (only one training class)')
        return rows

    classification_models = [
        ('NumPy Decision Tree', DecisionTree(task='classification', criterion='gini', **TREE_PARAMS)),
        ('Sklearn Decision Tree', DecisionTreeClassifier(criterion='gini', **TREE_PARAMS)),
    ]
    for name, model in classification_models:
        model.fit(x_fit, labels_fit)
        scores = model.predict_proba(x_test)[:, 1]
        pred = model.predict(x_test)
        rows.append({**common, 'task': 'classification', 'model': name,
                     **final_classification_metrics(labels_test, pred, scores)})

    svm_fit, svm_test = data_prep.standardize(x_fit, x_test)
    svm_models = [
        ('NumPy Pegasos SVM', PegasosSVM(**PEGASOS_PARAMS)),
        ('Sklearn LinearSVC', LinearSVC(**SVC_PARAMS)),
    ]
    for name, model in svm_models:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(svm_fit, labels_fit)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            print(f'  WARNING: {name} convergence warning in {category}')
        scores = model.decision_function(svm_test)
        pred = model.predict(svm_test)
        rows.append({**common, 'task': 'classification', 'model': name,
                     **final_classification_metrics(labels_test, pred, scores)})
    return rows


def run_final_evaluation():
    print('STEP 36 — FINAL HOLD-OUT TEST (run once; no parameter tuning)')
    df = data_prep.load_and_clean_data('data/house_sale.csv')
    groups = data_prep.split_by_category(df)
    all_rows = []
    for category, frame in groups.items():
        if category not in data_prep.CATEGORY_FEATURES:
            continue
        print(f'\n{category} | n={len(frame)}')
        new_rows = score_category(category, frame)
        all_rows.extend(new_rows)
        tree_reg = next((r for r in new_rows if r['task'] == 'regression'
                         and r['model'] == 'NumPy Decision Tree'), None)
        tree_clf = next((r for r in new_rows if r['task'] == 'classification'
                         and r['model'] == 'NumPy Decision Tree'), None)
        if tree_reg:
            print(f"  NumPy Tree test MAE={tree_reg['MAE_AZN']:,.2f} AZN, R2={tree_reg['R2']:.4f}")
        if tree_clf:
            print(f"  NumPy Tree test F1={tree_clf['F1']:.4f}, AUC={tree_clf['ROC_AUC']:.4f}")
        if len(frame) < 500:
            print('  CAUTION: small sample; high metric uncertainty')

    OUT.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(OUT / 'test_metrics.csv', index=False, encoding='utf-8-sig')
    def json_safe(obj):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(v) for v in obj]
        return obj

    with (OUT / 'test_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(json_safe(all_rows), f, ensure_ascii=False, indent=2, allow_nan=False)
    print('\n' + '=' * 78)
    print('FINAL TEST SUMMARY — NUMPY DECISION TREE (preselected primary model)')
    print('=' * 78)
    print(f"{'Category':24} {'MAE (AZN)':>15} {'R2':>9} {'F1':>9} {'ROC AUC':>9}")
    for category in groups:
        rs = [r for r in all_rows if r['category'] == category and r['model'] == 'NumPy Decision Tree']
        reg = next((r for r in rs if r['task'] == 'regression'), None)
        clf = next((r for r in rs if r['task'] == 'classification'), None)
        if reg:
            print(f"{category:24} {reg['MAE_AZN']:15,.2f} {reg['R2']:9.4f} "
                  f"{clf['F1'] if clf else float('nan'):9.4f} "
                  f"{clf['ROC_AUC'] if clf else float('nan'):9.4f}")
    print(f'\nSaved {OUT / "test_metrics.csv"} and {OUT / "test_metrics.json"}')
    print('IMPORTANT: Do not retune using these test results.')
    return all_rows


if __name__ == '__main__':
    run_final_evaluation()
