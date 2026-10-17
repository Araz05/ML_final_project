"""Decision tree classification and regression implemented with NumPy only."""
from __future__ import annotations
import numpy as np


class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.proba = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(self, task: str = "classification", criterion: str = "gini",
                 max_depth: int | None = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, min_impurity_decrease: float = 0.0,
                 random_state: int = 42):
        if task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        allowed = ("gini", "entropy") if task == "classification" else ("mse",)
        if criterion not in allowed:
            raise ValueError(f"criterion for {task} must be one of {allowed}")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be nonnegative")
        if min_samples_split < 2 or min_samples_leaf < 1 or min_impurity_decrease < 0:
            raise ValueError("Invalid stopping parameters")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None
        self.classes_ = None
        self.n_features_in_ = None

    def _impurity(self, y) -> float:
        y = np.asarray(y)
        if y.size == 0:
            return 0.0
        if self.task == "regression":
            return float(np.var(y.astype(float)))
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        if self.criterion == "gini":
            return float(1.0 - np.sum(p ** 2))
        return float(-np.sum(p * np.log2(p)))

    def _best_split(self, X, y):
        n, p = X.shape
        minimum = self.min_samples_leaf
        if n < max(self.min_samples_split, 2 * minimum):
            return None
        parent_impurity = self._impurity(y)
        if parent_impurity <= 1e-15:
            return None
        best_gain = max(0.0, self.min_impurity_decrease)
        best = None
        indices = np.arange(minimum, n - minimum + 1)
        if indices.size == 0:
            return None
        for feature in range(p):
            order = np.argsort(X[:, feature], kind="mergesort")
            values = X[order, feature]
            valid = indices[values[indices - 1] < values[indices]]
            if valid.size == 0:
                continue
            left_n = valid.astype(float)
            right_n = n - left_n
            if self.task == "regression":
                sorted_y = y[order].astype(float)
                sums = np.cumsum(sorted_y)
                sums2 = np.cumsum(sorted_y ** 2)
                left_sum, left_sq = sums[valid - 1], sums2[valid - 1]
                right_sum = sums[-1] - left_sum
                right_sq = sums2[-1] - left_sq
                # Weighted child variance = within-group squared error / n.
                child = (sums2[-1] - left_sum**2 / left_n
                         - right_sum**2 / right_n) / n
            else:
                labels = np.searchsorted(self.classes_, y[order])
                k = len(self.classes_)
                counts = np.zeros((n + 1, k), dtype=np.int64)
                counts[np.arange(1, n + 1), labels] = 1
                counts = np.cumsum(counts, axis=0)
                left = counts[valid].astype(float)
                right = counts[-1] - left
                lp = left / left_n[:, None]
                rp = right / right_n[:, None]
                if self.criterion == "gini":
                    li = 1 - np.sum(lp**2, axis=1)
                    ri = 1 - np.sum(rp**2, axis=1)
                else:
                    li = -np.sum(np.where(lp > 0, lp * np.log2(np.maximum(lp, 1e-300)), 0), axis=1)
                    ri = -np.sum(np.where(rp > 0, rp * np.log2(np.maximum(rp, 1e-300)), 0), axis=1)
                child = (left_n * li + right_n * ri) / n
            gains = parent_impurity - child
            index = int(np.argmax(gains))
            gain = float(gains[index])
            if gain > best_gain + 1e-12:
                pos = int(valid[index])
                # Numerically safe midpoint.
                a, b = float(values[pos - 1]), float(values[pos])
                threshold = a + (b - a) / 2
                best_gain = gain
                best = (feature, threshold, gain)
        return best

    def _grow(self, X, y, depth: int) -> Node:
        node = Node()
        if self.task == "regression":
            node.value = float(np.mean(y))
        else:
            counts = np.array([np.count_nonzero(y == c) for c in self.classes_])
            node.proba = counts / len(y)
            node.value = self.classes_[int(np.argmax(counts))]
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, _ = split
        mask = X[:, feature] <= threshold
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return node
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be 2D, y must be 1D, with equal nonzero rows")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity; preprocess first")
        if self.task == "regression":
            y = y.astype(float)
            if not np.isfinite(y).all():
                raise ValueError("y contains NaN or infinity")
            self.classes_ = None
        else:
            self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def _validate_predict(self, X):
        if self.root is None:
            raise RuntimeError("Fit the tree before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Invalid X shape or non-finite values")
        return X

    def predict(self, X):
        X = self._validate_predict(X)
        return np.asarray([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is only for classification")
        X = self._validate_predict(X)
        result = []
        for x in X:
            node = self.root
            while not node.is_leaf():
                node = node.left if x[node.feature] <= node.threshold else node.right
            result.append(node.proba)
        return np.asarray(result, dtype=float)
