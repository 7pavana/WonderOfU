"""NumPy RGB arrays to normalized PyTorch tensors without torchvision."""
from __future__ import annotations

import numpy as np
import torch


def rgb_to_tensor(rgb_image: np.ndarray, config: dict) -> torch.Tensor:
    if rgb_image.ndim != 3 or rgb_image.shape[-1] != 3:
        raise ValueError("Expected RGB image shaped [H,W,3].")
    image = torch.from_numpy(np.ascontiguousarray(rgb_image.transpose(2, 0, 1))).float().div(255.0)
    mean = torch.tensor(config["normalization_mean"], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(config["normalization_std"], dtype=image.dtype).view(3, 1, 1)
    if torch.any(std <= 0):
        raise ValueError("normalization_std values must be positive.")
    return (image - mean) / std
