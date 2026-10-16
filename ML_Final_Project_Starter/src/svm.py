"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.

Design choices made in this implementation
------------------------------------------
* Bias: FOLDED into the data as one extra constant feature (value
  `bias_scale`, default 1). So b = w_aug[-1] * bias_scale and the update rule
  above is used unchanged. Consequence: the bias is (very slightly)
  regularised too; the objective we track includes that tiny term.
* Mini-batches: each step samples `batch_size` rows (with replacement) and
  uses  w <- (1 - eta*lambda) w + (eta/k) * sum_{violators} y_i x_i.
  batch_size = 1 is the classic single-example Pegasos.
* Projection: after each step w is projected onto the ball
  ||w|| <= 1/sqrt(lambda) (Shalev-Shwartz et al., 2011), which the optimum is
  known to satisfy. This tames the huge first steps (eta_1 = 1/lambda).
* Iterate averaging: the returned model is the average of the iterates from
  the second half of training (`average=True`), which removes most of the
  noise of the last SGD step.
* Non-linearity: RANDOM FOURIER FEATURES (Rahimi & Recht, 2007) +
  linear Pegasos -> `RBFPegasosSVM`. Chosen over kernelised Pegasos because
  training stays O(n_iters * batch * D) with no n x n kernel matrix, which
  matters with ~10^5 listings.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
        batch_size: int = 64,      # examples per step (1 = classic Pegasos)
        average: bool = True,      # average the 2nd half of the iterates
        bias_scale: float = 1.0,   # value of the constant bias feature
        track_every: int = 500,    # record the objective every k steps (0 = off)
        sv_tol: float = 1e-8,      # tolerance for the support-vector test
    ):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be > 0")
        if n_iters < 1 or batch_size < 1:
            raise ValueError("n_iters and batch_size must be >= 1")
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.batch_size = batch_size
        self.average = average
        self.bias_scale = bias_scale
        self.track_every = track_every
        self.sv_tol = sv_tol
        self.w = None              # weight vector
        self.b = 0.0               # bias

        # filled by fit():
        self.classes_ = None            # the two original labels (neg, pos)
        self.objective_history_ = []    # [(step, J), ...] for the report plot
        self.support_ = None            # indices of support vectors in X_train
        self.n_support_ = 0

    # ------------------------------------------------------------------ labels
    @staticmethod
    def _to_pm1(y):
        """Map {0,1} (or any 2-class labels) to {-1, +1}.
        Returns (y_pm1, classes) where classes[0] -> -1 and classes[1] -> +1
        (classes are sorted, so {0,1} -> 0:-1, 1:+1). fit() stores `classes`."""
        y = np.asarray(y).ravel()
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"need exactly 2 classes, got {classes.size}: {classes}")
        y_pm1 = np.where(y == classes[1], 1.0, -1.0)
        return y_pm1, classes

    # -------------------------------------------------------------- internals
    def _augment(self, X):
        """Append the constant bias feature."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        col = np.full((X.shape[0], 1), self.bias_scale)
        return np.hstack([X, col])

    def _objective(self, Xa, y, w):
        """J = lambda/2 * ||w_aug||^2 + mean hinge loss (bias penalty included)."""
        margins = y * (Xa @ w)
        hinge = np.maximum(0.0, 1.0 - margins).mean()
        return float(0.5 * self.lambda_ * (w @ w) + hinge)

    # -------------------------------------------------------------------- fit
    def fit(self, X, y):
        """
        Run Pegasos. Sample one (or a mini-batch of) example(s) per step,
        use eta_t = 1/(lambda*t), and apply the update above.
        Return self.
        """
        y_pm1, self.classes_ = self._to_pm1(y)
        Xa = self._augment(X)
        n, d = Xa.shape
        if y_pm1.shape[0] != n:
            raise ValueError("X and y have different numbers of rows")

        rng = np.random.default_rng(self.random_state)
        lam = self.lambda_
        radius = 1.0 / np.sqrt(lam)
        k = min(self.batch_size, n)

        w = np.zeros(d)
        w_sum = np.zeros(d)
        n_avg = 0
        avg_start = self.n_iters // 2
        self.objective_history_ = []

        for t in range(1, self.n_iters + 1):
            idx = rng.integers(0, n, size=k)
            Xb, yb = Xa[idx], y_pm1[idx]
            eta = 1.0 / (lam * t)

            viol = yb * (Xb @ w) < 1.0                 # hinge-active examples
            w *= (1.0 - eta * lam)                      # regularisation shrink
            if viol.any():
                w += (eta / k) * (yb[viol] @ Xb[viol])  # sum_i y_i x_i / k

            norm = np.linalg.norm(w)                    # projection step
            if norm > radius:
                w *= radius / norm

            if self.average and t > avg_start:
                w_sum += w
                n_avg += 1

            if self.track_every and (t == 1 or t % self.track_every == 0
                                     or t == self.n_iters):
                self.objective_history_.append((t, self._objective(Xa, y_pm1, w)))

        w_final = (w_sum / n_avg) if (self.average and n_avg > 0) else w
        self.w = w_final[:-1].copy()
        self.b = float(w_final[-1] * self.bias_scale)

        # support vectors: points on or inside the margin (y f(x) <= 1)
        margins = y_pm1 * (Xa @ w_final)
        self.support_ = np.flatnonzero(margins <= 1.0 + self.sv_tol)
        self.n_support_ = int(self.support_.size)
        return self

    # -------------------------------------------------------------- inference
    def _check_fitted(self):
        if self.w is None:
            raise RuntimeError("model is not fitted yet; call fit(X, y) first")

    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]

    # ------------------------------------------------------ margin / SV report
    def margin_width(self) -> float:
        """Geometric margin width 2/||w|| (bias excluded)."""
        self._check_fitted()
        norm = np.linalg.norm(self.w)
        return float("inf") if norm == 0 else float(2.0 / norm)

    def weight_norm(self) -> float:
        """||w|| of the learned hyperplane."""
        self._check_fitted()
        return float(np.linalg.norm(self.w))

    def objective(self, X, y) -> float:
        """Primal objective J(w, b) on (X, y) with the documented formula
        (bias NOT penalised)."""
        self._check_fitted()
        y_pm1 = np.where(np.asarray(y).ravel() == self.classes_[1], 1.0, -1.0)
        hinge = np.maximum(0.0, 1.0 - y_pm1 * self.decision_function(X)).mean()
        return float(0.5 * self.lambda_ * (self.w @ self.w) + hinge)


