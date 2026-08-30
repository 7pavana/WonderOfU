"""Inference service intentionally returns unavailable without a compatible trained checkpoint."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class InferenceService:
    checkpoint: Path | None = None
    def analyze(self, face_path: Path | None, document_path: Path | None) -> dict:
        if not self.checkpoint or not self.checkpoint.is_file():
            unavailable = {"status": "unavailable", "reason": "No trained model checkpoint is configured."}
            return {"face_spoofing": unavailable, "face_forgery": unavailable, "document_authenticity": unavailable, "unified_assessment": {"status": "unavailable", "reason": "No model output available; no fraud conclusion is made."}}
        raise NotImplementedError("Checkpoint-specific image/video decoding is enabled after a trained checkpoint is registered.")
