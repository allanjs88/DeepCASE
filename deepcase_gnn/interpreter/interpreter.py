"""Validation-calibrated surprise threshold, not DeepCASE attention clustering."""

import numpy as np
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             precision_score, recall_score)


class Interpreter:
    def __init__(self, threshold=None):
        self.threshold = threshold

    def calibrate(self, scores, quantile=0.99):
        scores = np.asarray(scores, dtype=float)
        if not 0 < quantile < 1 or scores.size == 0 or not np.isfinite(scores).all():
            raise ValueError("Supply finite validation scores and a quantile between 0 and 1.")
        self.threshold = float(np.quantile(scores, quantile, method="higher"))
        return self.threshold

    def predict(self, scores):
        if self.threshold is None:
            raise RuntimeError("Calibrate the threshold on validation data first.")
        return (np.asarray(scores) > self.threshold).astype(int)


def detection_metrics(labels, scores, predictions):
    labels, scores, predictions = map(np.asarray, (labels, scores, predictions))
    if not (labels.ndim == scores.ndim == predictions.ndim == 1
            and len(labels) == len(scores) == len(predictions) and len(labels)):
        raise ValueError("Metrics require aligned, nonempty one-dimensional arrays.")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Labels and predictions must be binary.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    pr_auc = ap = None
    if positive:
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = float(auc(recall, precision))
        ap = float(average_precision_score(labels, scores))
    return {"events": len(labels), "positives": positive,
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)) if positive else None,
            "auc_pr": pr_auc, "average_precision": ap,
            "false_positive_rate": float(((labels == 0) & (predictions == 1)).sum() / negative)
            if negative else None}
