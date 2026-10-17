"""From-scratch linear soft-margin SVM trained with Pegasos SGD."""
from __future__ import annotations
import numpy as np


class PegasosSVM:
    def __init__(self, lambda_: float = 1e-4, n_iters: int = 20_000,
                 random_state: int = 42):
        if lambda_ <= 0 or n_iters < 1:
            raise ValueError('lambda_ must be positive and n_iters >= 1')
        self.lambda_ = float(lambda_)
        self.n_iters = int(n_iters)
        self.random_state = random_state
        self.w = None
        self.b = 0.0
        self.classes_ = None

    @staticmethod
    def _to_pm1(y):
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError('Exactly two classes are required')
        return np.where(y == classes[0], -1.0, 1.0), classes

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or not y.size:
            raise ValueError('Invalid X/y dimensions')
        if not np.isfinite(X).all():
            raise ValueError('X contains non-finite values')
        y_pm, self.classes_ = self._to_pm1(y)
        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(X.shape[1], dtype=float)
        self.b = 0.0
        for t in range(1, self.n_iters + 1):
            i = rng.integers(X.shape[0])
            xi, yi = X[i], y_pm[i]
            eta = 1.0 / (self.lambda_ * t)
            margin = yi * (np.dot(self.w, xi) + self.b)
            self.w *= (1.0 - 1.0 / t)
            if margin < 1.0:
                self.w += eta * yi * xi
                self.b += eta * yi
            # Project weights to the Pegasos L2 ball (bias is not penalized).
            norm = np.linalg.norm(self.w)
            bound = 1.0 / np.sqrt(self.lambda_)
            if norm > bound:
                self.w *= bound / norm
        return self

    def decision_function(self, X):
        if self.w is None:
            raise RuntimeError('Fit the model before prediction')
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.w.size or not np.isfinite(X).all():
            raise ValueError('Invalid prediction features')
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])
