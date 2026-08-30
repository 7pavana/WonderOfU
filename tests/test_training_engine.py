import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.datasets import SiWAdapter, build_dataloader
from ml.training import create_model, evaluate_model, run_training, validate_source_disjoint_splits


PREPROCESSING = {"image_size": 32, "frames_per_video": 4, "normalization_mean": [0.5] * 3, "normalization_std": [0.5] * 3, "face_detector": "none", "face_min_size": 10, "face_crop_padding": 0.0, "allow_center_crop_fallback": True}


def _video(root: Path, name: str, value: int) -> None:
    writer = cv2.VideoWriter(str(root / name), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    if not writer.isOpened(): raise RuntimeError("Test video codec is unavailable")
    for offset in range(4): writer.write(np.full((32, 32, 3), value + offset, dtype=np.uint8))
    writer.release()


class TrainingEngineTests(unittest.TestCase):
    def test_all_baseline_models_are_constructible(self):
        for kind, expected_task in (("baseline_spoof", "spoof"), ("baseline_forgery", "forgery"), ("baseline_document", "document")):
            model, task = create_model(kind, {"embedding_dim": 16})
            self.assertEqual(task, expected_task)
            self.assertEqual(tuple(model(torch.zeros((1, 3, 32, 32))).shape), (1, 2))

    def _fixture(self, root: Path) -> SiWAdapter:
        rows = []
        for split, label, value, identifier in [("train", 0, 20, "train-a"), ("train", 1, 100, "train-b"), ("val", 0, 40, "val-a"), ("val", 1, 140, "val-b"), ("test", 0, 60, "test-a"), ("test", 1, 160, "test-b")]:
            name = f"{identifier}.avi"; _video(root, name, value); rows.append({"path": name, "label": str(label), "source_id": identifier, "split": split})
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "source_id", "split"]); writer.writeheader(); writer.writerows(rows)
        return SiWAdapter(root)

    def test_training_validation_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); adapter = self._fixture(root); validate_source_disjoint_splits([adapter])
            train = build_dataloader(adapter, PREPROCESSING, "train", 2, shuffle=False); val = build_dataloader(adapter, PREPROCESSING, "val", 2, shuffle=False)
            config = {"seed": 7, "device": "cpu", "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0, "epochs": 1, "patience": 0, "experiment_name": "smoke"}
            first = run_training("unified", {"embedding_dim": 16, "shared_dim": 16, "use_rppg": True}, config, [train], [val], root / "experiments")
            self.assertTrue(Path(first.last_checkpoint).is_file()); self.assertTrue(Path(first.experiment_dir, "metrics.json").is_file())
            metrics = first.history[-1]["validation_metrics"]["spoof"]
            self.assertEqual(metrics["samples"], 2); self.assertIsNotNone(metrics["roc_auc"])
            config["epochs"] = 2
            resumed = run_training("unified", {"embedding_dim": 16, "shared_dim": 16, "use_rppg": True}, config, [train], [val], root / "resumed", resume_from=first.last_checkpoint)
            self.assertEqual(resumed.history[-1]["epoch"], 2)
            test_loader = build_dataloader(adapter, PREPROCESSING, "test", 2, shuffle=False)
            model, baseline_task = create_model("unified", {"embedding_dim": 16, "shared_dim": 16, "use_rppg": True})
            evaluation = evaluate_model(model, [test_loader], device="cpu", baseline_task=baseline_task)
            self.assertEqual(evaluation["metrics"]["spoof"]["samples"], 2)

    def test_rejects_source_split_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _video(root, "one.avi", 10)
            (root / "manifest.csv").write_text("path,label,source_id,split\none.avi,0,same,train\none.avi,1,same,val\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Leakage risk"):
                validate_source_disjoint_splits([SiWAdapter(root)])

if __name__ == "__main__": unittest.main()
