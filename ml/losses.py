"""Mathematically correct loss aggregation for incomplete task labels."""
from __future__ import annotations
try:
    import torch
    import torch.nn.functional as F
except ImportError: torch = F = None


def masked_multitask_loss(logits: dict, labels: dict, masks: dict):
    if torch is None: raise RuntimeError("PyTorch is required for training.")
    losses = []
    for task in ("spoof", "forgery", "document"):
        valid = masks[task].bool()
        if logits.get(task) is not None and valid.any():
            losses.append(F.cross_entropy(logits[task][valid], labels[task][valid]))
    if not losses: raise ValueError("Batch contains no active task labels.")
    return torch.stack(losses).mean()
