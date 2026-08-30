"""Modular detection and crop-based alignment for lightweight face-video models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


class FaceNotDetectedError(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceDetector(Protocol):
    def detect(self, rgb_image: np.ndarray) -> list[FaceBox]: ...


class OpenCVHaarFaceDetector:
    """CPU-friendly frontal-face detector shipped with OpenCV.

    It is a practical initial detector for an RTX 2050-era development machine,
    but does not provide landmarks; crop-based alignment is documented as such.
    """
    def __init__(self, min_size: int = 40):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError(f"OpenCV Haar cascade is unavailable: {cascade_path}")
        self.min_size = min_size

    def detect(self, rgb_image: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        boxes = self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_size, self.min_size))
        return [FaceBox(int(x), int(y), int(width), int(height)) for x, y, width, height in boxes]


class NoOpFaceDetector:
    """Test-only/configurable detector that deliberately reports no face."""
    def detect(self, rgb_image: np.ndarray) -> list[FaceBox]:
        return []


def create_face_detector(config: dict) -> FaceDetector:
    name = config["face_detector"]
    if name == "opencv_haar":
        return OpenCVHaarFaceDetector(int(config["face_min_size"]))
    if name == "none":
        return NoOpFaceDetector()
    raise ValueError(f"Unsupported face_detector {name!r}; supported values: 'opencv_haar', 'none'.")


def crop_and_align_face(rgb_image: np.ndarray, boxes: list[FaceBox], image_size: int, padding: float, allow_center_crop_fallback: bool) -> np.ndarray:
    """Choose the largest detection, pad its crop, and resize it deterministically.

    This is geometric crop normalization, not landmark alignment. If no face is
    detected, production configuration raises instead of silently treating a
    non-face as facial evidence. The centered fallback exists only for controlled
    software tests or explicitly approved low-assurance development runs.
    """
    if not 0 <= padding < 1:
        raise ValueError("face_crop_padding must be in [0, 1).")
    height, width = rgb_image.shape[:2]
    if boxes:
        box = max(boxes, key=lambda item: item.area)
        pad_x, pad_y = int(box.width * padding), int(box.height * padding)
        left, top = max(0, box.x - pad_x), max(0, box.y - pad_y)
        right, bottom = min(width, box.x + box.width + pad_x), min(height, box.y + box.height + pad_y)
    elif allow_center_crop_fallback:
        side = min(height, width)
        left, top = (width - side) // 2, (height - side) // 2
        right, bottom = left + side, top + side
    else:
        raise FaceNotDetectedError("No face detected. No center-crop fallback is enabled.")
    crop = rgb_image[top:bottom, left:right]
    if crop.size == 0:
        raise FaceNotDetectedError("Face crop is empty after bounds handling.")
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)


def preprocess_face_sequence(frames_rgb: np.ndarray, detector: FaceDetector, config: dict) -> np.ndarray:
    crops = [crop_and_align_face(frame, detector.detect(frame), int(config["image_size"]), float(config["face_crop_padding"]), bool(config["allow_center_crop_fallback"])) for frame in frames_rgb]
    return np.stack(crops, axis=0)
