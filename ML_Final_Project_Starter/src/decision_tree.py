# """
# decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

# Must support BOTH tasks:
#   - classification (criterion = "gini" or "entropy")  -> price tier
#   - regression     (criterion = "mse")                -> price

# scikit-learn is NOT allowed here. This is your own code.

# Suggested design:
#   - a small Node (feature index, threshold, left, right, value/leaf_value),
#   - recursive _grow with a best-split search over features & thresholds,
#   - impurity functions (gini / entropy / mse) and the impurity-decrease
#     objective used to pick splits,
#   - hyperparameters that control overfitting (below).
# """

# from __future__ import annotations

# import numpy as np


# class Node:
#     """One node of the tree. Fill in the fields you need."""
#     def __init__(self):
#         self.feature = None      # int: feature index to split on (internal)
#         self.threshold = None    # float: split threshold (internal)
#         self.left = None         # Node
#         self.right = None        # Node
#         self.value = None        # leaf prediction (class label/proba or mean)

#     def is_leaf(self) -> bool:
#         return self.left is None and self.right is None


# class DecisionTree:
#     def __init__(
#         self,
#         task: str = "classification",   # "classification" | "regression"
#         criterion: str = "gini",        # "gini" | "entropy" | "mse"
#         max_depth: int | None = None,
#         min_samples_split: int = 2,
#         min_samples_leaf: int = 1,
#         min_impurity_decrease: float = 0.0,
#         random_state: int = 42,
#     ):
#         self.task = task
#         self.criterion = criterion
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_impurity_decrease = min_impurity_decrease
#         self.random_state = random_state
#         self.root: Node | None = None

#     # --- impurity -----------------------------------------------------------
#     def _impurity(self, y) -> float:
#         """Gini / entropy (classification) or MSE/variance (regression)."""
#         raise NotImplementedError("TODO: impurity")

#     def _best_split(self, X, y):
#         """
#         Search features & candidate thresholds; return the split that
#         maximizes impurity decrease (feature, threshold, decrease), or
#         None if no valid split improves things.
#         """
#         raise NotImplementedError("TODO: best split search")

#     # --- grow / predict -----------------------------------------------------
#     def _grow(self, X, y, depth: int) -> Node:
#         """Recursively build the tree, respecting the stopping rules."""
#         raise NotImplementedError("TODO: recursive grow")

#     def fit(self, X, y):
#         X = np.asarray(X, dtype=float)
#         y = np.asarray(y)
#         self.root = self._grow(X, y, depth=0)
#         return self

#     def _predict_one(self, x, node: Node):
#         raise NotImplementedError("TODO: traverse to a leaf")

#     def predict(self, X):
#         raise NotImplementedError("TODO: predict")

#     def predict_proba(self, X):
#         """Classification only: return class probabilities from leaf counts."""
#         raise NotImplementedError("TODO: predict_proba (classification)")





















"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Supports BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT used here. (It is only used in tests/ as a reference.)

Maths implemented (all "impurity" I(.) are per node):
    Gini     G = 1 - sum_k p_k^2
    Entropy  H = - sum_k p_k log2 p_k
    MSE      I = (1/N) sum_i (y_i - mean(y))^2        (= variance)
    Information gain / impurity decrease of a split:
        IG = I(parent) - (N_left/N) I(left) - (N_right/N) I(right)
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node: samples with x[feature] <= threshold
        self.right = None        # Node: samples with x[feature] >  threshold
        self.value = None        # leaf prediction: class-prob vector or mean
        self.n_samples = 0       # how many training rows reached this node
        self.impurity = 0.0      # impurity of this node
        self.gain = 0.0          # N_t/N * impurity decrease (for importances)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,         # unused for now; reserved for the
                                        # random-forest bonus (feature sampling)
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        valid = ("gini", "entropy") if task == "classification" else ("mse",)
        if criterion not in valid:
            raise ValueError(f"criterion for {task} must be one of {valid}, "
                             f"got {criterion!r}")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # ------------------------------------------------------------------
    # impurity
    # ------------------------------------------------------------------
    @staticmethod
    def _impurity_from_counts(counts, criterion):
        """Gini/entropy for class-count rows. counts: (..., K). Works on a
        single node (1-D) or on MANY candidate splits at once (2-D)."""
        n = counts.sum(axis=-1, keepdims=True)
        p = counts / np.maximum(n, 1)              # avoid 0/0
        if criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=-1)
        # entropy: define 0*log2(0) = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(p > 0, p * np.log2(p), 0.0)
        return -np.sum(plogp, axis=-1)

    def _impurity(self, y) -> float:
        """Impurity of one node. For classification y holds class INDICES."""
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            counts = np.bincount(y, minlength=self.n_classes_).astype(float)
            return float(self._impurity_from_counts(counts, self.criterion))
        return float(np.var(y))                    # MSE around the mean

    # ------------------------------------------------------------------
    # split search
    # ------------------------------------------------------------------
    def _best_split(self, X, y):
        """
        Try every feature and every candidate threshold; return
        (feature, threshold, decrease) for the split with the LARGEST impurity
        decrease, or None if no valid split exists.

        Trick: sort each feature once, then use running (cumulative) sums so
        the impurity of ALL possible left/right partitions is computed at once
        instead of re-scanning the data for every threshold.
        """
        n, d = X.shape
        msl = self.min_samples_leaf
        parent_imp = self._impurity(y)

        if self.task == "classification":
            onehot = np.eye(self.n_classes_)[y]            # (n, K)
            total = onehot.sum(axis=0)
        else:
            yc = y - y.mean()          # centring leaves variance unchanged but
            total_sum, total_sq = yc.sum(), (yc ** 2).sum()   # avoids float
                                       # cancellation with big prices

        n_left = np.arange(1, n)       # left size if we cut after position i
        n_right = n - n_left

        best = None                    # (decrease, feature, threshold)
        for j in range(d):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]

            # A cut is only possible BETWEEN two different values, and must
            # leave at least min_samples_leaf rows on each side.
            valid = (xs[:-1] != xs[1:]) & (n_left >= msl) & (n_right >= msl)
            if not valid.any():
                continue

            if self.task == "classification":
                left = np.cumsum(onehot[order], axis=0)[:-1]   # (n-1, K)
                right = total - left
                imp_l = self._impurity_from_counts(left, self.criterion)
                imp_r = self._impurity_from_counts(right, self.criterion)
            else:
                cs = np.cumsum(yc[order])[:-1]
                cq = np.cumsum((yc[order]) ** 2)[:-1]
                imp_l = np.maximum(cq / n_left - (cs / n_left) ** 2, 0.0)
                rs, rq = total_sum - cs, total_sq - cq
                imp_r = np.maximum(rq / n_right - (rs / n_right) ** 2, 0.0)

            weighted = (n_left * imp_l + n_right * imp_r) / n
            decrease = np.where(valid, parent_imp - weighted, -np.inf)
            i = int(np.argmax(decrease))       # first best position
            if best is None or decrease[i] > best[0]:   # strict > : on ties the
                lo, hi = xs[i], xs[i + 1]                # lowest feature wins
                thr = (lo + hi) / 2.0          # midpoint, like scikit-learn
                if thr >= hi:                  # float rounding guard
                    thr = lo
                best = (float(decrease[i]), j, float(thr))

        if best is None:
            return None
        return best[1], best[2], best[0]

    # ------------------------------------------------------------------
    # grow / predict
    # ------------------------------------------------------------------
    def _leaf_value(self, y):
        if self.task == "classification":
            counts = np.bincount(y, minlength=self.n_classes_).astype(float)
            return counts / counts.sum()          # class probabilities
        return float(np.mean(y))

    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        node = Node()
        node.n_samples = len(y)
        node.impurity = self._impurity(y)
        node.value = self._leaf_value(y)

        # --- stopping rules: any of these makes this node a leaf ---------
        if (self.max_depth is not None and depth >= self.max_depth) \
                or len(y) < self.min_samples_split \
                or len(y) < 2 * self.min_samples_leaf \
                or node.impurity <= 1e-12:                    # already pure
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, decrease = split

        # min_impurity_decrease uses the same scale as scikit-learn:
        # (N_t / N_total) * (impurity decrease).
        weighted_gain = decrease * len(y) / self.n_train_
        if weighted_gain < self.min_impurity_decrease:
            return node

        mask = X[:, feature] <= threshold
        node.feature, node.threshold, node.gain = feature, threshold, weighted_gain
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification":
            # map arbitrary labels to 0..K-1 so we can use bincount/one-hot
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
        else:
            y = y.astype(float)
        self.n_features_ = X.shape[1]
        self.n_train_ = len(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        """Walk from the root to a leaf and return that leaf's value."""
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def _leaf_values(self, X):
        X = np.asarray(X, dtype=float)
        return [self._predict_one(x, self.root) for x in X]

    def predict(self, X):
        values = self._leaf_values(X)
        if self.task == "classification":
            idx = np.argmax(np.vstack(values), axis=1)   # majority class
            return self.classes_[idx]
        return np.array(values)                          # leaf mean

    def predict_proba(self, X):
        """Classification only: class probabilities from leaf frequencies."""
        if self.task != "classification":
            raise ValueError("predict_proba is only defined for classification")
        return np.vstack(self._leaf_values(X))

    # ------------------------------------------------------------------
    # diagnostics (used later for Problem 4 and 6)
    # ------------------------------------------------------------------
    def _nodes(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if not node.is_leaf():
                stack += [node.left, node.right]

    @property
    def feature_importances_(self):
        """Impurity-decrease importance: total weighted gain of every split
        made on a feature, normalised to sum to 1."""
        imp = np.zeros(self.n_features_)
        for node in self._nodes():
            if not node.is_leaf():
                imp[node.feature] += node.gain
        total = imp.sum()
        return imp / total if total > 0 else imp

    def get_depth(self) -> int:
        def depth(node):
            return 0 if node.is_leaf() else 1 + max(depth(node.left),
                                                    depth(node.right))
        return depth(self.root)

    def get_n_leaves(self) -> int:
        return sum(node.is_leaf() for node in self._nodes())

    def export_text(self, feature_names=None) -> str:
        """Human-readable tree (handy for the demo and for debugging)."""
        names = feature_names or [f"x{j}" for j in range(self.n_features_)]
        lines = []

        def rec(node, indent):
            pad = "|   " * indent
            if node.is_leaf():
                v = node.value
                out = (self.classes_[int(np.argmax(v))]
                       if self.task == "classification" else round(v, 3))
                lines.append(f"{pad}|--- value: {out}  (n={node.n_samples})")
                return
            lines.append(f"{pad}|--- {names[node.feature]} <= {node.threshold:.4g}")
            rec(node.left, indent + 1)
            lines.append(f"{pad}|--- {names[node.feature]} >  {node.threshold:.4g}")
            rec(node.right, indent + 1)

        rec(self.root, 0)
        return "\n".join(lines)