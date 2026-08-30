import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ml.experiments.standard import run_preflight, run_standard_experiment

PREPROCESSING = {"image_size": 32, "frames_per_video": 4, "normalization_mean": [0.5] * 3, "normalization_std": [0.5] * 3, "face_detector": "none", "face_min_size": 10, "face_crop_padding": 0.0, "allow_center_crop_fallback": True}
MODEL = {"embedding_dim": 16, "shared_dim": 16, "use_rppg": True}
TRAINING = {"seed": 3, "batch_size": 2, "num_workers": 0, "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0, "epochs": 1, "patience": 0, "device": "cpu", "experiment_name": "standard_smoke"}


def _video(root: Path, name: str, value: int) -> None:
    writer = cv2.VideoWriter(str(root / name), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    if not writer.isOpened(): raise RuntimeError("Test video codec is unavailable")
    for offset in range(4): writer.write(np.full((32, 32, 3), value + offset, dtype=np.uint8))
    writer.release()


class StandardExperimentTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        rows = []
        for split, label, value, source in [("train", 0, 20, "train-a"), ("train", 1, 80, "train-b"), ("val", 0, 40, "val-a"), ("val", 1, 120, "val-b"), ("test", 0, 60, "test-a"), ("test", 1, 160, "test-b")]:
            name = f"{source}.avi"; _video(root, name, value); rows.append({"path": name, "label": label, "source_id": source, "split": split})
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "source_id", "split"]); writer.writeheader(); writer.writerows(rows)
        return manifest

    def test_valid_preflight_and_standard_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = self._fixture(root)
            report, _ = run_preflight("dfdc", root, manifest, "baseline_forgery", PREPROCESSING)
            self.assertEqual(report.sample_count, 6); self.assertEqual(report.preprocessing_samples["test"], "passed")
            outcome = run_standard_experiment("dfdc", root, manifest, "baseline_forgery", MODEL, TRAINING, PREPROCESSING, root / "experiments")
            directory = Path(outcome["experiment_dir"])
            for name in ("preflight.json", "reproducibility.json", "test_metrics.json", "last.pt", "metrics.json"):
                self.assertTrue((directory / name).is_file())
            self.assertEqual(json.loads((directory / "test_metrics.json").read_text())["metrics"]["forgery"]["samples"], 2)

    def test_invalid_manifests_and_baseline_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                run_preflight("dfdc", root, manifest, "baseline_document", PREPROCESSING)
            manifest.write_text("path,label,split\nmissing.avi,0,train\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain columns"):
                run_preflight("dfdc", root, manifest, "baseline_forgery", PREPROCESSING)
            manifest = self._fixture(root)
            with manifest.open("a", encoding="utf-8") as handle: handle.write("does-not-exist.avi,0,new-source,test\n")
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                run_preflight("dfdc", root, manifest, "baseline_forgery", PREPROCESSING)
            manifest = self._fixture(root)
            with manifest.open("a", encoding="utf-8") as handle: handle.write("train-a.avi,0,train-a,val\n")
            with self.assertRaisesRegex(ValueError, "Leakage risk"):
                run_preflight("dfdc", root, manifest, "baseline_forgery", PREPROCESSING)

if __name__ == "__main__": unittest.main()
