"""Binary classification metrics implemented without scikit-learn."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    # Average ranks preserve the standard tie handling used by Mann-Whitney AUC.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def evaluate_task_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int | None]:
    """Compute metrics for explicit binary labels and unnormalized two-class logits."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Expected logits shaped [N,2].")
    values = labels.detach().cpu().numpy().astype(int)
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Evaluation labels must be binary 0/1 values.")
    probabilities = torch.softmax(logits.detach().cpu(), dim=1)[:, 1].numpy()
    predicted = (probabilities >= 0.5).astype(int)
    true_positive = int(np.logical_and(predicted == 1, values == 1).sum())
    true_negative = int(np.logical_and(predicted == 0, values == 0).sum())
    false_positive = int(np.logical_and(predicted == 1, values == 0).sum())
    false_negative = int(np.logical_and(predicted == 0, values == 1).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"samples": len(values), "accuracy": (true_positive + true_negative) / len(values) if len(values) else None,
            "precision": precision, "recall": recall, "f1": f1, "roc_auc": _roc_auc(values, probabilities),
            "true_positive": true_positive, "true_negative": true_negative, "false_positive": false_positive, "false_negative": false_negative}


def summarize_predictions(predictions: dict[str, list[tuple[torch.Tensor, torch.Tensor]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task, entries in predictions.items():
        if entries:
            summary[task] = evaluate_task_logits(torch.cat([item[0] for item in entries]), torch.cat([item[1] for item in entries]))
        else:
            summary[task] = {"samples": 0, "accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None}
    return summary
