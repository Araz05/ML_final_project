"""All property categories: validation-only tree regression/classification and linear SVM.
Run from project root: python -m src.run_all
No test-set evaluation or hyperparameter selection is performed here.
"""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

from . import data_prep, evaluate
from .decision_tree import DecisionTree
from .svm import PegasosSVM

TREE_PARAMS = dict(max_depth=20, min_samples_split=20, min_samples_leaf=2, random_state=42)
PEGASOS_PARAMS = dict(lambda_=0.001, n_iters=300_000, random_state=42)
SVC_PARAMS = dict(C=0.1, loss="squared_hinge", dual=False, tol=1e-4,
                  max_iter=50_000, random_state=42)


def classification_metrics(actual, predicted, scores):
    precision, recall, f1 = evaluate.precision_recall_f1(actual, predicted)
    return {"Accuracy": float(np.mean(actual == predicted)), "Precision": precision,
            "Recall": recall, "F1": f1, "ROC AUC": evaluate.roc_auc(actual, scores)}


def regression_metrics(actual, predicted):
    return {"MAE (AZN)": evaluate.mae(actual, predicted),
            "RMSE (AZN)": evaluate.rmse(actual, predicted),
            "Median AE (AZN)": evaluate.median_absolute_error(actual, predicted),
            "R2": evaluate.r2(actual, predicted)}


def run_category(category, frame):
    print("\n" + "=" * 72)
    print(f"CATEGORY: {category} | listings={len(frame)}")
    print("=" * 72)
    train, validation, _test = data_prep.split_dataset(frame)
    raw_train, log_train = data_prep.prepare_category_features(train, category)
    raw_val, log_val = data_prep.prepare_category_features(validation, category)

    # Third argument is a train-only placeholder, not the held-out test set.
    # The existing preprocessing API requires exactly three arguments.
    x_train, x_val, _ = data_prep.encode_category_features(
        raw_train, raw_val, raw_train.iloc[:1].copy())
    x_train, x_val, _ = data_prep.fill_missing_numeric(
        x_train, x_val, x_train.iloc[:1].copy())
    x_train = x_train.to_numpy(dtype=float)
    x_val = x_val.to_numpy(dtype=float)
    y_train = log_train.to_numpy(dtype=float)
    actual = np.exp(log_val.to_numpy(dtype=float))

    own_reg = DecisionTree(task="regression", criterion="mse", **TREE_PARAMS)
    base_reg = DecisionTreeRegressor(criterion="squared_error", **TREE_PARAMS)
    own_reg.fit(x_train, y_train)
    base_reg.fit(x_train, y_train)
    reg_results = {
        "Our NumPy Decision Tree": regression_metrics(actual, np.exp(own_reg.predict(x_val))),
        "Sklearn Decision Tree": regression_metrics(actual, np.exp(base_reg.predict(x_val))),
    }
    print("\nREGRESSION — validation")
    evaluate.compare(reg_results)

    threshold = float(np.median(np.exp(y_train)))
    labels_train = (np.exp(y_train) > threshold).astype(int)
    labels_val = (actual > threshold).astype(int)
    print(f"\nCLASSIFICATION — validation | TRAIN median={threshold:,.2f} AZN")
    print(f"Train classes: {np.bincount(labels_train, minlength=2).tolist()} | "
          f"Validation classes: {np.bincount(labels_val, minlength=2).tolist()}")
    if len(np.unique(labels_train)) < 2 or len(np.unique(labels_val)) < 2:
        print("SKIPPED: two classes are required in both train and validation.")
        return {"regression": reg_results, "classification": {}, "svm": {}}

    own_clf = DecisionTree(task="classification", criterion="gini", **TREE_PARAMS)
    base_clf = DecisionTreeClassifier(criterion="gini", **TREE_PARAMS)
    own_clf.fit(x_train, labels_train)
    base_clf.fit(x_train, labels_train)
    class_results = {
        "Our NumPy Decision Tree": classification_metrics(
            labels_val, own_clf.predict(x_val), own_clf.predict_proba(x_val)[:, 1]),
        "Sklearn Decision Tree": classification_metrics(
            labels_val, base_clf.predict(x_val), base_clf.predict_proba(x_val)[:, 1]),
    }
    evaluate.compare(class_results)

    svm_train, svm_val = data_prep.standardize(x_train, x_val)
    own_svm = PegasosSVM(**PEGASOS_PARAMS)
    base_svm = LinearSVC(**SVC_PARAMS)
    own_svm.fit(svm_train, labels_train)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        base_svm.fit(svm_train, labels_train)
    svm_results = {
        "Our NumPy Pegasos SVM": classification_metrics(
            labels_val, own_svm.predict(svm_val), own_svm.decision_function(svm_val)),
        "Sklearn LinearSVC": classification_metrics(
            labels_val, base_svm.predict(svm_val), base_svm.decision_function(svm_val)),
    }
    print("\nSVM — validation (train-only standardization)")
    evaluate.compare(svm_results)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        print("WARNING: LinearSVC did not converge for this category.")
    if len(frame) < 500:
        print("CAUTION: small category; validation metrics have high uncertainty.")
    return {"regression": reg_results, "classification": class_results, "svm": svm_results}


def main():
    df = data_prep.load_and_clean_data("data/house_sale.csv")
    groups = data_prep.split_by_category(df)
    results = {}
    for category, frame in groups.items():
        if category not in data_prep.CATEGORY_FEATURES:
            print(f"Skipping unsupported category: {category}")
            continue
        results[category] = run_category(category, frame)
    print("\n" + "=" * 72)
    print("SUMMARY — validation only")
    print("=" * 72)
    print(f"{'Category':24} {'Tree MAE':>14} {'Tree F1':>10} {'Pegasos F1':>12} {'LinearSVC F1':>13}")
    for category, r in results.items():
        reg = r["regression"]["Our NumPy Decision Tree"]["MAE (AZN)"]
        tree = r["classification"].get("Our NumPy Decision Tree", {}).get("F1", float('nan'))
        pegasos = r["svm"].get("Our NumPy Pegasos SVM", {}).get("F1", float('nan'))
        linear = r["svm"].get("Sklearn LinearSVC", {}).get("F1", float('nan'))
        print(f"{category:24} {reg:14,.2f} {tree:10.4f} {pegasos:12.4f} {linear:13.4f}")
    print("\nNOTE: Test sets were not evaluated. Parameters are exploratory outside Yeni tikili.")
    return results


if __name__ == "__main__":
    main()
