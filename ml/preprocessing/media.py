"""Dependency-light, deterministic media decoding used before model preprocessing."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class MediaDecodeError(RuntimeError):
    """Raised when a real manifest sample cannot be decoded."""


def decode_image(path: str | Path) -> np.ndarray:
    """Decode a document image to an RGB uint8 array shaped ``[H, W, 3]``."""
    source = Path(path)
    if not source.is_file():
        raise MediaDecodeError(f"Image file does not exist: {source}")
    try:
        with Image.open(source) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        raise MediaDecodeError(f"Could not decode image {source}: {exc}") from exc


def decode_video_frames(path: str | Path, frames_per_video: int) -> np.ndarray:
    """Uniformly sample decoded RGB frames as ``[T, H, W, 3]``.

    The function does not generate frames if decoding fails. A short video returns
    all decodable frames and is rejected if it has fewer than two, because CHROM
    rPPG requires a temporal sequence.
    """
    source = Path(path)
    if not source.is_file():
        raise MediaDecodeError(f"Video file does not exist: {source}")
    if frames_per_video < 2:
        raise ValueError("frames_per_video must be at least 2 for rPPG extraction.")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise MediaDecodeError(f"Could not open video: {source}")
    try:
        declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if declared_count < 2:
            raise MediaDecodeError(f"Video has fewer than two frames: {source}")
        positions = np.unique(np.linspace(0, declared_count - 1, min(frames_per_video, declared_count), dtype=int))
        frames: list[np.ndarray] = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise MediaDecodeError(f"Could not decode frame {position} from {source}")
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) < 2:
        raise MediaDecodeError(f"Video yielded fewer than two frames: {source}")
    return np.stack(frames, axis=0)
