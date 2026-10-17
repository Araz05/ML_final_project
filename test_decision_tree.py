"""Small correctness checks for the from-scratch DecisionTree.

Run: python -m src.test_decision_tree
Does not load the house-sale CSV or train on the full dataset.
"""
import numpy as np
from .decision_tree import DecisionTree


def test_regression():
    x = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.where(x[:, 0] < 6, 10.0, 20.0)
    model = DecisionTree(task="regression", criterion="mse",
                         max_depth=3, min_samples_split=2,
                         min_samples_leaf=1, random_state=42)
    model.fit(x, y)
    predicted = np.asarray(model.predict(x), dtype=float)
    assert predicted.shape == y.shape
    assert np.allclose(predicted, y), (predicted, y)


def test_classification():
    x = np.arange(12, dtype=float).reshape(-1, 1)
    y = (x[:, 0] >= 6).astype(int)
    model = DecisionTree(task="classification", criterion="gini",
                         max_depth=3, min_samples_split=2,
                         min_samples_leaf=1, random_state=42)
    model.fit(x, y)
    predicted = np.asarray(model.predict(x))
    assert np.array_equal(predicted, y), (predicted, y)


if __name__ == "__main__":
    test_regression()
    test_classification()
    print("Decision Tree regression and classification checks passed.")
