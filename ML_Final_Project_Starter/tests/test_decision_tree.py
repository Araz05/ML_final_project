"""
Tests for src/decision_tree.py  (Problem 2c: correctness).

Run from the repo root:   python -m pytest tests -v
scikit-learn is used HERE ONLY, as a reference to compare against.
"""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree


# ---------- helpers ---------------------------------------------------------
def my_structure(node):
    """Pre-order list of ('leaf') or (feature, threshold) for MY tree."""
    if node.is_leaf():
        return [("leaf",)]
    return ([(node.feature, node.threshold)]
            + my_structure(node.left) + my_structure(node.right))


def sk_structure(t, i=0):
    """Same pre-order list for a fitted scikit-learn tree."""
    if t.children_left[i] == -1:
        return [("leaf",)]
    return ([(int(t.feature[i]), float(t.threshold[i]))]
            + sk_structure(t, t.children_left[i])
            + sk_structure(t, t.children_right[i]))


def same_structure(a, b, tol=1e-4):
    if len(a) != len(b):
        return False
    for p, q in zip(a, b):
        if p[0] == "leaf" or q[0] == "leaf":
            if p != q:
                return False
        elif p[0] != q[0] or not np.isclose(p[1], q[1], rtol=tol, atol=tol):
            return False
    return True


def _canon(n, imp, kids):
    """Canonical form of a subtree: (size, impurity, children sorted). Sorting
    the children makes it ignore WHICH side a group ended up on."""
    return (n, round(imp, 9), tuple(sorted(kids)))


def my_canon(node):
    kids = [] if node.is_leaf() else [my_canon(node.left), my_canon(node.right)]
    return _canon(node.n_samples, node.impurity, kids)


def sk_canon(t, i=0):
    kids = [] if t.children_left[i] == -1 else [
        sk_canon(t, t.children_left[i]), sk_canon(t, t.children_right[i])]
    return _canon(int(t.n_node_samples[i]), float(t.impurity[i]), kids)


def random_data(n=300, d=5, seed=0, task="classification"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))                   # continuous -> no ties
    if task == "classification":
        y = (X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * rng.normal(size=n) > 0.5)
        return X, y.astype(int)
    return X, 3 * X[:, 0] - X[:, 2] ** 2 + 0.2 * rng.normal(size=n)


# ---------- 1. the tiny, fully-determined example (required by the PDF) -----
def test_tiny_example_classification_matches_sklearn():
    # Feature 0 separates the classes perfectly at 3.5; feature 1 is noise.
    X = np.array([[1, 10], [2, 20], [3, 10], [4, 20], [5, 10], [6, 20]], float)
    y = np.array([0, 0, 0, 1, 1, 1])
    mine = DecisionTree("classification", "gini").fit(X, y)
    ref = DecisionTreeClassifier(criterion="gini", random_state=0).fit(X, y)
    assert my_structure(mine.root) == [(0, 3.5), ("leaf",), ("leaf",)]
    assert same_structure(my_structure(mine.root), sk_structure(ref.tree_))
    assert np.array_equal(mine.predict(X), ref.predict(X))
    assert np.allclose(mine.feature_importances_, ref.feature_importances_)


def test_tiny_example_regression_matches_sklearn():
    X = np.array([[1], [2], [3], [10], [11], [12]], float)
    y = np.array([1.0, 1.2, 0.8, 5.0, 5.2, 4.8])
    mine = DecisionTree("regression", "mse", max_depth=1).fit(X, y)
    ref = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)
    assert my_structure(mine.root)[0] == (0, 6.5)       # midpoint of 3 and 10
    assert same_structure(my_structure(mine.root), sk_structure(ref.tree_))
    assert np.allclose(mine.predict(X), ref.predict(X))  # leaf means 1.0 / 5.0


# ---------- 2. larger random data --------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_regression_matches_sklearn_exactly_on_random_data(seed):
    """Regression on a continuous target has no ties, so the exact feature,
    threshold, predictions and importances must all agree."""
    X, y = random_data(seed=seed, task="regression")
    mine = DecisionTree("regression", "mse", max_depth=6,
                        min_samples_leaf=10).fit(X, y)
    ref = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10,
                                random_state=0).fit(X, y)
    assert same_structure(my_structure(mine.root), sk_structure(ref.tree_))
    assert np.allclose(mine.predict(X), ref.predict(X), atol=1e-6)
    assert np.allclose(mine.feature_importances_, ref.feature_importances_,
                       atol=1e-6)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classification_root_gain_matches_sklearn(criterion, seed):
    """Classification gain depends only on class COUNTS, so different features
    often TIE and trees need not be identical (tie-breaking differs from
    sklearn's random feature order). What must always agree is the BEST gain
    the root can achieve, since both see exactly the same data there."""
    X, y = random_data(seed=seed, task="classification")
    mine = DecisionTree("classification", criterion, max_depth=1).fit(X, y)
    ref = DecisionTreeClassifier(criterion=criterion, max_depth=1,
                                 random_state=0).fit(X, y)
    t = ref.tree_
    l, r = t.children_left[0], t.children_right[0]
    sk_gain = (t.impurity[0] * t.n_node_samples[0]
               - t.impurity[l] * t.n_node_samples[l]
               - t.impurity[r] * t.n_node_samples[r]) / t.n_node_samples[0]
    assert np.isclose(mine.root.gain, sk_gain, atol=1e-9)


def brute_force_best_decrease(tree, X, y, msl):
    """Independent, slow reference: loop over every feature and every midpoint,
    compute child impurities directly from the definition."""
    best = -np.inf
    parent = tree._impurity(y)
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for thr in (vals[:-1] + vals[1:]) / 2:
            left = X[:, j] <= thr
            nl, nr = left.sum(), (~left).sum()
            if nl < msl or nr < msl:
                continue
            w = (nl * tree._impurity(y[left]) + nr * tree._impurity(y[~left])) / len(y)
            best = max(best, parent - w)
    return best


@pytest.mark.parametrize("task,criterion", [("classification", "gini"),
                                            ("classification", "entropy"),
                                            ("regression", "mse")])
@pytest.mark.parametrize("msl", [1, 4])
@pytest.mark.parametrize("seed", range(4))
def test_best_split_is_truly_optimal(task, criterion, msl, seed):
    """The vectorised cumulative-sum search must find the same maximum gain as
    a naive brute-force search over every feature and threshold."""
    X, y = random_data(n=40, d=3, seed=seed, task=task)
    tree = DecisionTree(task, criterion, min_samples_leaf=msl).fit(X, y)
    y_enc = np.searchsorted(tree.classes_, y) if task == "classification" else y
    feature, threshold, decrease = tree._best_split(X, y_enc)
    assert np.isclose(decrease, brute_force_best_decrease(tree, X, y_enc, msl))
    # and the reported (feature, threshold) really achieves that decrease
    left = X[:, feature] <= threshold
    w = (left.sum() * tree._impurity(y_enc[left])
         + (~left).sum() * tree._impurity(y_enc[~left])) / len(y_enc)
    assert np.isclose(tree._impurity(y_enc) - w, decrease)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classification_accuracy_comparable_to_sklearn(criterion):
    """Sanity check: even with tie differences, accuracy should be close."""
    for seed in range(5):
        X, y = random_data(seed=seed, task="classification")
        mine = DecisionTree("classification", criterion, max_depth=5,
                            min_samples_leaf=15).fit(X, y)
        ref = DecisionTreeClassifier(criterion=criterion, max_depth=5,
                                     min_samples_leaf=15, random_state=0).fit(X, y)
        assert abs((mine.predict(X) == y).mean() - (ref.predict(X) == y).mean()) <= 0.02


# ---------- 3. each hyper-parameter must actually change the tree -----------
def test_every_hyperparameter_changes_the_tree():
    X, y = random_data(n=400, task="classification")
    base = DecisionTree("classification").fit(X, y)

    deep = base.get_depth()
    assert DecisionTree("classification", max_depth=2).fit(X, y).get_depth() == 2
    assert deep > 2

    leaves = base.get_n_leaves()
    assert DecisionTree("classification", min_samples_split=100).fit(X, y).get_n_leaves() < leaves
    assert DecisionTree("classification", min_samples_leaf=25).fit(X, y).get_n_leaves() < leaves
    assert DecisionTree("classification", min_impurity_decrease=0.01).fit(X, y).get_n_leaves() < leaves

    # min_samples_leaf must be honoured by EVERY leaf
    t = DecisionTree("classification", min_samples_leaf=25).fit(X, y)
    assert all(n.n_samples >= 25 for n in t._nodes() if n.is_leaf())


# ---------- 4. API behaviour -------------------------------------------------
def test_predict_proba_rows_sum_to_one_and_labels_roundtrip():
    X, y = random_data(task="classification")
    labels = np.where(y == 1, "premium", "standard")      # string labels
    t = DecisionTree("classification", max_depth=4).fit(X, labels)
    assert np.allclose(t.predict_proba(X).sum(axis=1), 1.0)
    assert set(t.predict(X)) <= {"premium", "standard"}


def test_pure_node_and_constant_feature_do_not_crash():
    X = np.ones((10, 2))                                  # nothing to split on
    t = DecisionTree("classification").fit(X, np.array([0, 1] * 5))
    assert t.root.is_leaf() and t.get_depth() == 0
    t = DecisionTree("regression", "mse").fit(X, np.arange(10.0))
    assert t.predict(X[:2]).tolist() == [4.5, 4.5]


def test_wrong_criterion_for_task_raises():
    with pytest.raises(ValueError):
        DecisionTree("regression", "gini")
    with pytest.raises(ValueError):
        DecisionTree("classification", "mse")
