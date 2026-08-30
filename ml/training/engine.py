"""Reusable, partial-label training/evaluation engine. No dataset is created here."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ml.datasets.base import ManifestDatasetAdapter
from ml.evaluation import summarize_predictions
from ml.losses import masked_multitask_loss
from ml.models import DocumentBaseline, ForgeryBaseline, SpoofBaseline, WonderOfUModel


TASKS = ("spoof", "forgery", "document")


@dataclass
class TrainingResult:
    experiment_dir: str
    best_checkpoint: str | None
    last_checkpoint: str
    history: list[dict[str, Any]]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the installed PyTorch build.")
    if requested not in {"cpu", "cuda"}: raise ValueError("training.device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(requested)


def create_model(model_kind: str, model_config: dict[str, Any]) -> tuple[nn.Module, str | None]:
    """Return a model and the active task for an independent baseline."""
    embedding = int(model_config.get("embedding_dim", 128))
    if model_kind == "unified":
        return WonderOfUModel(embedding, int(model_config.get("shared_dim", 128)), bool(model_config.get("use_rppg", True))), None
    baseline_map = {"baseline_spoof": (SpoofBaseline, "spoof"), "baseline_forgery": (ForgeryBaseline, "forgery"), "baseline_document": (DocumentBaseline, "document")}
    if model_kind not in baseline_map: raise ValueError(f"Unknown model_kind {model_kind!r}.")
    constructor, task = baseline_map[model_kind]
    return constructor(embedding), task


def validate_source_disjoint_splits(adapters: Iterable[ManifestDatasetAdapter]) -> None:
    """Refuse manifests that reuse a source ID across train/validation/test."""
    seen: dict[tuple[str, str], str] = {}
    for adapter in adapters:
        for sample in adapter.samples():
            key = (adapter.name, sample.source_id)
            previous = seen.get(key)
            if previous is not None and previous != sample.split:
                raise ValueError(f"Leakage risk: {adapter.name} source_id {sample.source_id!r} appears in both {previous!r} and {sample.split!r}.")
            seen[key] = sample.split


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return ({key: value.to(device, non_blocking=True) for key, value in batch["model_inputs"].items()},
            {key: value.to(device, non_blocking=True) for key, value in batch["labels"].items()},
            {key: value.to(device, non_blocking=True) for key, value in batch["masks"].items()})


def _forward(model: nn.Module, inputs: dict[str, torch.Tensor], baseline_task: str | None) -> dict[str, torch.Tensor | None]:
    if baseline_task is None: return model(**inputs)
    required = "document" if baseline_task == "document" else "face"
    if required not in inputs: raise ValueError(f"{baseline_task} baseline received incompatible batch inputs.")
    return {task: (model(inputs[required]) if task == baseline_task else None) for task in TASKS}


def _run_epoch(model: nn.Module, loaders: Iterable[DataLoader], optimizer: Optimizer | None, device: torch.device, baseline_task: str | None) -> tuple[float, dict[str, Any]]:
    is_training = optimizer is not None
    model.train(is_training)
    losses: list[float] = []
    predictions: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {task: [] for task in TASKS}
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for loader in loaders:
            for batch in loader:
                inputs, labels, masks = _to_device(batch, device)
                logits = _forward(model, inputs, baseline_task)
                loss = masked_multitask_loss(logits, labels, masks)
                if is_training:
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                losses.append(float(loss.detach().cpu()))
                for task in TASKS:
                    active = masks[task].bool()
                    if logits.get(task) is not None and active.any():
                        predictions[task].append((logits[task][active].detach().cpu(), labels[task][active].detach().cpu()))
    if not losses: raise ValueError("No batches were available for this epoch.")
    return float(np.mean(losses)), summarize_predictions(predictions)


def _optimizer(model: nn.Module, config: dict[str, Any]) -> Optimizer:
    kind = config.get("optimizer", "adamw").lower()
    kwargs = {"lr": float(config["learning_rate"]), "weight_decay": float(config.get("weight_decay", 0.0))}
    if kind == "adamw": return torch.optim.AdamW(model.parameters(), **kwargs)
    if kind == "sgd": return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    raise ValueError("training.optimizer must be 'adamw' or 'sgd'.")


def evaluate_model(model: nn.Module, loaders: Iterable[DataLoader], device: str = "auto", baseline_task: str | None = None) -> dict[str, Any]:
    """Evaluate validation or test DataLoaders without updating model weights."""
    resolved = resolve_device(device)
    model.to(resolved)
    loss, metrics = _run_epoch(model, loaders, None, resolved, baseline_task)
    return {"loss": loss, "metrics": metrics, "device": str(resolved)}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def run_training(model_kind: str, model_config: dict[str, Any], training_config: dict[str, Any], train_loaders: Iterable[DataLoader], val_loaders: Iterable[DataLoader], experiment_root: str | Path, resume_from: str | Path | None = None) -> TrainingResult:
    """Train and validate one unified or baseline model, saving reproducible artifacts."""
    set_seed(int(training_config.get("seed", 42)))
    device = resolve_device(str(training_config.get("device", "auto")))
    model, baseline_task = create_model(model_kind, model_config); model.to(device)
    optimizer = _optimizer(model, training_config)
    root = Path(experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    experiment_dir = root / f"{training_config.get('experiment_name', model_kind)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    experiment_dir.mkdir()
    _write_json(experiment_dir / "config.json", {"model_kind": model_kind, "model": model_config, "training": training_config, "device": str(device)})
    start_epoch, best_loss, history = 1, float("inf"), []
    if resume_from:
        payload = torch.load(Path(resume_from), map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"]); optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch, best_loss, history = int(payload["epoch"]) + 1, float(payload["best_val_loss"]), list(payload.get("history", []))
    best_checkpoint: Path | None = None
    patience, stale = int(training_config.get("patience", 0)), 0
    for epoch in range(start_epoch, int(training_config["epochs"]) + 1):
        train_loss, train_metrics = _run_epoch(model, train_loaders, optimizer, device, baseline_task)
        val_loss, val_metrics = _run_epoch(model, val_loaders, None, device, baseline_task)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss, "train_metrics": train_metrics, "validation_metrics": val_metrics}
        history.append(record); _write_json(experiment_dir / "metrics.json", history)
        payload = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_val_loss": min(best_loss, val_loss), "history": history, "model_kind": model_kind}
        last_path = experiment_dir / "last.pt"; torch.save(payload, last_path)
        if val_loss < best_loss:
            best_loss, stale, best_checkpoint = val_loss, 0, experiment_dir / "best.pt"; torch.save(payload, best_checkpoint)
        else:
            stale += 1
            if patience and stale >= patience: break
    return TrainingResult(str(experiment_dir), str(best_checkpoint) if best_checkpoint else None, str(experiment_dir / "last.pt"), history)
