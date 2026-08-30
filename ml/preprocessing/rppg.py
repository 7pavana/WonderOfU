"""Lightweight CHROM rPPG extraction from a skin ROI RGB sequence.

Input is [T, H, W, 3] RGB. A face detector/alignment stage must supply skin-rich
face crops; this function does not pretend whole-frame RGB is physiology.
"""
from __future__ import annotations
import numpy as np


def extract_chrom_signal(face_rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if face_rgb.ndim != 4 or face_rgb.shape[-1] != 3 or face_rgb.shape[0] < 2:
        raise ValueError("Expected at least two RGB face ROI frames shaped [T,H,W,3].")
    # Central ROI avoids hair/background; robust mean is a defensible lightweight skin proxy.
    _, height, width, _ = face_rgb.shape
    roi = face_rgb[:, height // 4:3 * height // 4, width // 4:3 * width // 4].astype(np.float32)
    rgb = roi.mean(axis=(1, 2))
    normalized = rgb / (rgb.mean(axis=0, keepdims=True) + eps) - 1.0
    x = 3.0 * normalized[:, 0] - 2.0 * normalized[:, 1]
    y = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]
    alpha = np.std(x) / (np.std(y) + eps)
    signal = x - alpha * y
    return (signal - signal.mean()) / (signal.std() + eps)
