"""Thin team-facing wrapper around the authoritative WonderOfU engine."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ml.datasets import (DFDCAdapter, DocumentRecaptureAdapter, FaceForensicsAdapter,
                         IDNetAdapter, SiWAdapter, build_dataloader)
from ml.datasets.base import ManifestDatasetAdapter
from ml.training import (create_model, evaluate_model, run_training,
                         validate_source_disjoint_splits)


DATASETS: dict[str, tuple[type[ManifestDatasetAdapter], str]] = {
    "siw": (SiWAdapter, "spoof"),
    "faceforensics": (FaceForensicsAdapter, "forgery"),
    "dfdc": (DFDCAdapter, "forgery"),
    "idnet": (IDNetAdapter, "document"),
    "document_recapture": (DocumentRecaptureAdapter, "document"),
}
BASELINES = {"spoof": "baseline_spoof", "forgery": "baseline_forgery", "document": "baseline_document"}
REQUIRED_COLUMNS = ("path", "label", "source_id", "split")


@dataclass(frozen=True)
class PreflightReport:
    dataset: str
    task: str
    dataset_root: str
    manifest_path: str
    manifest_sha256: str
    sample_count: int
    split_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    source_group_counts: dict[str, int]
    preprocessing_samples: dict[str, str]
    model_kind: str


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _adapter(dataset: str, dataset_root: str | Path, manifest_path: str | Path) -> tuple[ManifestDatasetAdapter, str]:
    try:
        adapter_type, task = DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}.") from exc
    return adapter_type(dataset_root, manifest_path), task


def _validate_model_task(model_kind: str, task: str) -> None:
    if model_kind == "unified":
        return
    expected = BASELINES[task]
    if model_kind != expected:
        raise ValueError(f"Model {model_kind!r} is incompatible with {task!r}. Use {expected!r} or 'unified'.")


def _read_manifest_header(manifest: Path) -> None:
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        header = csv.DictReader(handle).fieldnames
    if not header or not set(REQUIRED_COLUMNS).issubset(header):
        raise ValueError(f"Manifest {manifest} must contain columns {list(REQUIRED_COLUMNS)}.")


def run_preflight(dataset: str, dataset_root: str | Path, manifest_path: str | Path,
                  model_kind: str, preprocessing: dict[str, Any]) -> tuple[PreflightReport, ManifestDatasetAdapter]:
    """Fail closed before training, reusing adapters and the existing leakage check."""
    manifest = Path(manifest_path)
    _read_manifest_header(manifest)
    adapter, task = _adapter(dataset, dataset_root, manifest)
    _validate_model_task(model_kind, task)
    samples = list(adapter.samples())
    if not samples:
        raise ValueError("Manifest contains no samples; no training can be started.")
    split_counts = Counter(sample.split for sample in samples)
    missing_splits = {"train", "val", "test"} - set(split_counts)
    if missing_splits:
        raise ValueError(f"Manifest is missing required splits: {sorted(missing_splits)}.")
    for sample in samples:
        if not sample.input_path.is_file():
            raise FileNotFoundError(f"Manifest media file does not exist: {sample.input_path}")
    # This is the authoritative validator; never downgrade leakage failure to a warning.
    validate_source_disjoint_splits([adapter])
    class_counts = {split: {str(label): sum(1 for item in samples if item.split == split and item.label == label) for label in (0, 1)}
                    for split in ("train", "val", "test")}
    source_counts = {split: len({item.source_id for item in samples if item.split == split}) for split in ("train", "val", "test")}
    # Process one real sample from each split before the trainer can open an epoch.
    preprocessing_samples: dict[str, str] = {}
    for split in ("train", "val", "test"):
        loader = build_dataloader(adapter, preprocessing, split, batch_size=1, num_workers=0, shuffle=False)
        next(iter(loader))
        preprocessing_samples[split] = "passed"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return PreflightReport(dataset, task, str(Path(dataset_root).resolve()), str(manifest.resolve()), digest,
                           len(samples), dict(split_counts), class_counts, source_counts, preprocessing_samples, model_kind), adapter


def run_standard_experiment(dataset: str, dataset_root: str | Path, manifest_path: str | Path,
                            model_kind: str, model_config: dict[str, Any], training_config: dict[str, Any],
                            preprocessing: dict[str, Any], experiment_root: str | Path,
                            resume_from: str | Path | None = None) -> dict[str, Any]:
    """Preflight, delegate training, then write held-out-test and reproducibility artifacts."""
    report, adapter = run_preflight(dataset, dataset_root, manifest_path, model_kind, preprocessing)
    batch_size = int(training_config["batch_size"])
    workers = int(training_config.get("num_workers", 0))
    train = build_dataloader(adapter, preprocessing, "train", batch_size, workers)
    val = build_dataloader(adapter, preprocessing, "val", batch_size, workers, shuffle=False)
    test = build_dataloader(adapter, preprocessing, "test", batch_size, workers, shuffle=False)
    result = run_training(model_kind, model_config, training_config, [train], [val], experiment_root, resume_from)
    experiment_dir = Path(result.experiment_dir)
    checkpoint = Path(result.best_checkpoint or result.last_checkpoint)
    model, baseline_task = create_model(model_kind, model_config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    test_result = evaluate_model(model, [test], str(training_config.get("device", "auto")), baseline_task)
    _json(experiment_dir / "preflight.json", asdict(report))
    _json(experiment_dir / "test_metrics.json", test_result)
    reproducibility = {**asdict(report), "preprocessing": preprocessing, "model": model_config,
                       "training": training_config, "epochs_completed": len(result.history),
                       "best_checkpoint": result.best_checkpoint, "last_checkpoint": result.last_checkpoint}
    _json(experiment_dir / "reproducibility.json", reproducibility)
    return {"experiment_dir": str(experiment_dir), "preflight": asdict(report), "training": asdict(result), "test": test_result}
