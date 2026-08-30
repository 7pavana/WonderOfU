"""Run data ingestion only against a user-supplied explicit manifest and media."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.config import load_config, preprocessing_config
from ml.datasets import SiWAdapter, build_dataloader

parser = argparse.ArgumentParser(description="Validate an explicit SiW-style video manifest without training.")
parser.add_argument("dataset_root", type=Path, help="Root containing manifest.csv and officially obtained media")
args = parser.parse_args()
config = preprocessing_config(load_config())
loader = build_dataloader(SiWAdapter(args.dataset_root), config, "train", batch_size=1, num_workers=0, shuffle=False)
batch = next(iter(loader))
print({"face": tuple(batch["model_inputs"]["face"].shape), "rppg": tuple(batch["model_inputs"]["rppg"].shape), "source_id": batch["metadata"]["source_id"], "split": batch["metadata"]["split"]})
