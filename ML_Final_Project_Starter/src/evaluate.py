"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH, NumPy only).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers (see tests/) or for the baseline models you compare
against.

Conventions
-----------
* Regression (Task A): y is the price in AZN.
* Classification (Task B): the POSITIVE class is 1 = premium. Labels in
  {-1, +1} (as used by the SVM) are accepted and mapped to {0, 1}.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# input checking
# =============================================================================
def _check_pair(y_true, y_pred):
    """Flatten both arrays to 1D float and make sure the shapes match."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        raise ValueError("empty input")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def _check_binary_pair(y_true, y_pred):
    """Flatten to 1D int arrays of 0/1. Labels in {-1, +1} are mapped to {0, 1}."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        raise ValueError("empty input")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    out = []
    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        y = np.where(y == -1, 0, y).astype(int)
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f"{name} must contain only 0/1 (or -1/+1) labels")
        out.append(y)
    return out[0], out[1]


def _check_scores(y_true, scores):
    """Binary labels + real-valued scores, both flattened."""
    y_true = np.asarray(y_true).ravel()
    y_true = np.where(y_true == -1, 0, y_true).astype(int)
    scores = np.asarray(scores, dtype=float).ravel()
    if y_true.shape != scores.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {scores.shape}")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0/1 (or -1/+1) labels")
    if y_true.min() == y_true.max():
        raise ValueError("y_true contains a single class; the metric is undefined")
    return y_true, scores


# =============================================================================
# Regression (Task A: price)
# =============================================================================
def rmse(y_true, y_pred) -> float:
    """Root mean squared error: sqrt(mean((y - y_hat)^2))."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error: mean(|y - y_hat|)."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:                      # constant target (same as sklearn)
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# =============================================================================
# Classification (Task B: price tier)
# =============================================================================
def accuracy(y_true, y_pred) -> float:
    """Fraction of correct predictions."""
    y_true, y_pred = _check_binary_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]] (rows = true, cols = predicted;
    positive class = 1 = premium)."""
    y_true, y_pred = _check_binary_pair(y_true, y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class.
    precision = TP/(TP+FP), recall = TP/(TP+FN), f1 = harmonic mean.
    A zero denominator gives 0.0 (same as sklearn's zero_division=0)."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def _average_ranks(x):
    """Ranks 1..n of x, ties receive the average of their ranks."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)               # rank of the last member of each tie group
    first = last - counts + 1              # rank of the first member
    return ((first + last) / 2.0)[inverse]


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney U): AUC = (R_pos - n_pos(n_pos+1)/2) / (n_pos*n_neg),
    where R_pos is the sum of the (tie-averaged) ranks of the positives.
    """
    y_true, scores = _check_scores(y_true, scores)
    ranks = _average_ranks(scores)
    n_pos = int(y_true.sum())
    n_neg = y_true.size - n_pos
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(y_true, scores):
    """Return (fpr, tpr, thresholds) with one point per distinct score,
    starting at (0, 0). Used for the ROC figure."""
    y_true, scores = _check_scores(y_true, scores)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], y_true[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]   # last index of each tie
    tps = np.cumsum(y)[distinct]
    fps = (1 + distinct) - tps
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    thresholds = np.r_[np.inf, s[distinct]]
    return fpr, tpr, thresholds


def average_precision(y_true, scores) -> float:
    """PR-AUC as average precision: sum_k (R_k - R_{k-1}) * P_k over the
    distinct score thresholds (same definition as sklearn)."""
    y_true, scores = _check_scores(y_true, scores)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], y_true[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]
    tps = np.cumsum(y)[distinct]
    precision = tps / (1 + distinct)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


# =============================================================================
# Comparison helper
# =============================================================================
def _fmt(value) -> str:
    """float -> '0.8123'; (mean, std) -> '0.8123 ± 0.0040'; None/NaN -> 'n/a'."""
    if value is None:
        return "n/a"
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{value[0]:.4f} ± {value[1]:.4f}"
    if isinstance(value, str):
        return value
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(value)
    mag = abs(float(value))
    return f"{value:,.0f}" if mag >= 1000 else f"{value:.4f}"


def format_table(name_to_metrics: dict) -> str:
    """Render {model_name: {metric: value}} as an aligned text table.
    A value may be a float or a (mean, std) pair; a missing metric prints n/a."""
    models = list(name_to_metrics)
    metrics = []
    for m in models:
        for k in name_to_metrics[m]:
            if k not in metrics:
                metrics.append(k)
    rows = [["Model"] + metrics]
    for m in models:
        rows.append([m] + [_fmt(name_to_metrics[m].get(k)) for k in metrics])
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    lines = []
    for i, r in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[0]) if c == 0 else cell.rjust(widths[c])
                               for c, cell in enumerate(r)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.
    """
    print(format_table(name_to_metrics))
    