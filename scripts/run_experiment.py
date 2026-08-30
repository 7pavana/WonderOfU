"""Standard team entry point. It delegates all ML work to ml.experiments.standard."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.config import load_config, preprocessing_config
from ml.experiments.standard import BASELINES, DATASETS, run_preflight, run_standard_experiment


parser = argparse.ArgumentParser(description="Run a leakage-safe WonderOfU baseline/unified experiment.")
parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
parser.add_argument("--dataset-root", required=True, type=Path)
parser.add_argument("--manifest", required=True, type=Path)
parser.add_argument("--model", choices=["unified", *sorted(BASELINES.values())], default=None)
parser.add_argument("--config", type=Path, default=None)
parser.add_argument("--experiment-root", type=Path, default=None)
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument("--seed", type=int); parser.add_argument("--batch-size", type=int)
parser.add_argument("--epochs", type=int); parser.add_argument("--optimizer", choices=["adamw", "sgd"])
parser.add_argument("--learning-rate", type=float); parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
parser.add_argument("--preflight-only", action="store_true")
args = parser.parse_args()
config = load_config(args.config)
training = dict(config["training"]); training["seed"] = config["seed"]
for argument, key in ((args.seed, "seed"), (args.batch_size, "batch_size"), (args.epochs, "epochs"), (args.optimizer, "optimizer"), (args.learning_rate, "learning_rate"), (args.device, "device")):
    if argument is not None: training[key] = argument
task = DATASETS[args.dataset][1]
model = args.model or BASELINES[task]
kwargs = {"dataset": args.dataset, "dataset_root": args.dataset_root, "manifest_path": args.manifest,
          "model_kind": model, "preprocessing": preprocessing_config(config)}
if args.preflight_only:
    report, _ = run_preflight(**kwargs); print(report)
else:
    outcome = run_standard_experiment(**kwargs, model_config=config["model"], training_config=training,
                                      experiment_root=args.experiment_root or config["output_root"], resume_from=args.resume)
    print(outcome)
