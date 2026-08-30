"""PyTorch ingestion around explicit, source-disjoint manifest samples."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from ml.preprocessing.face import create_face_detector, preprocess_face_sequence
from ml.preprocessing.media import decode_image, decode_video_frames
from ml.preprocessing.rppg import extract_chrom_signal
from ml.preprocessing.tensors import rgb_to_tensor

from .base import FraudSample, ManifestDatasetAdapter


class FraudTorchDataset(Dataset[dict[str, Any]]):
    """Loads one task at a time; every item retains source and split metadata."""
    def __init__(self, adapter: ManifestDatasetAdapter, preprocessing: dict, split: str):
        self.samples = list(adapter.samples(split))
        if not self.samples:
            raise ValueError(f"No {split!r} samples found for {adapter.name}; no labels were invented.")
        self.adapter, self.preprocessing = adapter, dict(preprocessing)
        self.detector = create_face_detector(self.preprocessing)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        common = {"task": sample.task, "label": sample.label, "source_id": sample.source_id, "split": sample.split, "input_path": str(sample.input_path), "metadata": sample.metadata}
        if sample.task == "document":
            image = decode_image(sample.input_path)
            image = cv2.resize(image, (int(self.preprocessing["image_size"]), int(self.preprocessing["image_size"])), interpolation=cv2.INTER_AREA)
            return {**common, "document": rgb_to_tensor(image, self.preprocessing)}
        frames = decode_video_frames(sample.input_path, int(self.preprocessing["frames_per_video"]))
        faces = preprocess_face_sequence(frames, self.detector, self.preprocessing)
        rppg = torch.from_numpy(extract_chrom_signal(faces)).float().unsqueeze(0)
        # The lightweight visual encoder consumes one representative aligned frame;
        # rPPG retains the full sampled sequence.
        return {**common, "face": rgb_to_tensor(faces[len(faces) // 2], self.preprocessing), "rppg": rppg}


def fraud_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a model-compatible, task-masked homogeneous-modality batch."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    is_document = ["document" in sample for sample in samples]
    if any(is_document) and not all(is_document):
        raise ValueError("Mixed face/document batches are unsupported; use task-homogeneous DataLoaders.")
    tasks = [sample["task"] for sample in samples]
    labels = {task: torch.zeros(len(samples), dtype=torch.long) for task in ("spoof", "forgery", "document")}
    masks = {task: torch.zeros(len(samples), dtype=torch.bool) for task in labels}
    for position, sample in enumerate(samples):
        labels[sample["task"]][position] = sample["label"]
        masks[sample["task"]][position] = True
    metadata = {key: [sample[key] for sample in samples] for key in ("source_id", "split", "input_path", "metadata")}
    if all(is_document):
        model_inputs = {"document": torch.stack([sample["document"] for sample in samples])}
    else:
        model_inputs = {"face": torch.stack([sample["face"] for sample in samples]), "rppg": torch.stack([sample["rppg"] for sample in samples])}
    return {"model_inputs": model_inputs, "labels": labels, "masks": masks, "tasks": tasks, "metadata": metadata}


def build_dataloader(adapter: ManifestDatasetAdapter, preprocessing: dict, split: str, batch_size: int, num_workers: int = 0, shuffle: bool | None = None) -> DataLoader:
    dataset = FraudTorchDataset(adapter, preprocessing, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train" if shuffle is None else shuffle), num_workers=num_workers, collate_fn=fraud_collate, pin_memory=torch.cuda.is_available())
