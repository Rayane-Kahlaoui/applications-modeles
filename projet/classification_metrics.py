"""Image-level binary classification metrics for one fixed threshold."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_for_recall(labels, scores, target_recall):
    """Choose the highest threshold attaining a recall target on tuning data."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or not len(labels) or labels.shape != scores.shape
            or not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all()
            or not np.isfinite(target_recall) or not 0 < target_recall <= 1):
        raise ValueError("Expected binary labels, finite scores and a recall target in (0, 1]")
    positive_scores = np.sort(scores[labels == 1])
    if not len(positive_scores):
        raise ValueError("Recall threshold requires at least one positive example")
    required = int(np.ceil(target_recall * len(positive_scores)))
    return float(positive_scores[len(positive_scores) - required])


def binary_metrics(labels, scores, threshold):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or not len(labels) or labels.shape != scores.shape
            or not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all()
            or not np.isfinite(threshold)):
        raise ValueError("Expected nonempty binary labels, finite scores and a finite threshold")
    positive = labels.astype(bool)
    predicted = scores >= threshold
    tp = int(np.count_nonzero(positive & predicted))
    fp = int(np.count_nonzero(~positive & predicted))
    fn = int(np.count_nonzero(positive & ~predicted))
    tn = int(np.count_nonzero(~positive & ~predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "specificity": specificity, "false_positive_rate": 1 - specificity if specificity is not None else None,
            "average_precision": float(average_precision_score(labels, scores)) if positive.any() else None,
            "roc_auc": float(roc_auc_score(labels, scores)) if positive.any() and (~positive).any() else None}
