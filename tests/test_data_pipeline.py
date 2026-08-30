import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ml.datasets import IDNetAdapter, SiWAdapter, build_dataloader
from ml.models import WonderOfUModel


def test_preprocessing() -> dict:
    return {"image_size": 32, "frames_per_video": 4, "normalization_mean": [0.5, 0.5, 0.5], "normalization_std": [0.5, 0.5, 0.5], "face_detector": "none", "face_min_size": 10, "face_crop_padding": 0.0, "allow_center_crop_fallback": True}


class DataPipelineTests(unittest.TestCase):
    def _manifest(self, root: Path, row: dict[str, str]) -> None:
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "source_id", "split"])
            writer.writeheader(); writer.writerow(row)

    def test_document_manifest_to_model_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(np.full((20, 30, 3), 127, dtype=np.uint8)).save(root / "document.png")
            self._manifest(root, {"path": "document.png", "label": "1", "source_id": "document-1", "split": "train"})
            batch = next(iter(build_dataloader(IDNetAdapter(root), test_preprocessing(), "train", batch_size=1, shuffle=False)))
            self.assertEqual(tuple(batch["model_inputs"]["document"].shape), (1, 3, 32, 32))
            self.assertEqual(batch["metadata"]["source_id"], ["document-1"])
            output = WonderOfUModel(embedding_dim=16, shared_dim=16)(**batch["model_inputs"])
            self.assertEqual(tuple(output["document"].shape), (1, 2))

    def test_video_manifest_to_model_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); video = root / "clip.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            for value in (20, 60, 100, 140): writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
            writer.release()
            self._manifest(root, {"path": "clip.avi", "label": "0", "source_id": "video-1", "split": "train"})
            batch = next(iter(build_dataloader(SiWAdapter(root), test_preprocessing(), "train", batch_size=1, shuffle=False)))
            self.assertEqual(tuple(batch["model_inputs"]["face"].shape), (1, 3, 32, 32))
            self.assertEqual(tuple(batch["model_inputs"]["rppg"].shape), (1, 1, 4))
            self.assertEqual(batch["metadata"]["split"], ["train"])
            output = WonderOfUModel(embedding_dim=16, shared_dim=16)(**batch["model_inputs"])
            self.assertEqual(tuple(output["spoof"].shape), (1, 2))

if __name__ == "__main__":
    unittest.main()
