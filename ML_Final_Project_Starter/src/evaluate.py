"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import numpy as np

def _check_pair(y_true, y_pred):
    """Flatten both arrays to 1D float and make sure the shapes match."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        raise ValueError("empty input")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred

# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    raise NotImplementedError("TODO: sqrt(mean((y_true - y_pred)^2))")


def mae(y_true, y_pred) -> float:
    raise NotImplementedError("TODO: mean(|y_true - y_pred|)")


def r2(y_true, y_pred) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:                      # constant target (same as sklearn)
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# --- Classification (Task B: price tier) -----------------------------------
def _check_binary_pair(y_true, y_pred):
    """Flatten to 1D int arrays of 0/1. Labels in {-1, +1} (as used by the SVM)
    are mapped to {0, 1}, so both conventions work."""
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
    """Return (precision, recall, f1) for the positive (premium) class."""
    raise NotImplementedError("TODO: precision/recall/f1")


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.
    """
    raise NotImplementedError("TODO: ROC AUC")


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.
    """
    raise NotImplementedError("TODO: print comparison table")
