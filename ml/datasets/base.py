"""Manifest-first dataset abstractions that never infer semantic labels."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

Task = Literal["spoof", "forgery", "document"]


@dataclass(frozen=True)
class FraudSample:
    input_path: Path
    task: Task
    label: int
    source_id: str
    split: str
    metadata: dict[str, str] = field(default_factory=dict)


class ManifestDatasetAdapter:
    """Reads official/curated manifests; labels must be explicitly supplied as 0 or 1."""
    name = "base"
    task: Task = "spoof"
    valid_splits = {"train", "val", "test"}

    def __init__(self, root: str | Path, manifest: str | Path | None = None):
        self.root = Path(root)
        self.manifest = Path(manifest) if manifest else self.root / "manifest.csv"

    @property
    def available(self) -> bool:
        return self.root.is_dir()

    def availability_message(self) -> str:
        if not self.available:
            return "Dataset not available/configured. Adapter is ready, but training for this dataset is currently skipped."
        if not self.manifest.is_file():
            return f"Dataset directory found, but no explicit manifest at {self.manifest}. No labels will be inferred."
        return f"Dataset and explicit manifest available: {self.manifest}"

    def samples(self, split: str | None = None) -> Iterator[FraudSample]:
        if not self.manifest.is_file():
            return
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                required = {"path", "label", "source_id", "split"}
                if not required.issubset(row):
                    raise ValueError(f"{self.manifest} requires columns {sorted(required)}")
                if split and row["split"] != split:
                    continue
                if not all(row[column].strip() for column in required):
                    raise ValueError("Manifest path, label, source_id, and split must all be non-empty.")
                if row["split"] not in self.valid_splits:
                    raise ValueError(f"Split must be one of {sorted(self.valid_splits)}; got {row['split']!r}.")
                if row["label"] not in {"0", "1"}:
                    raise ValueError("Labels must be explicit binary values 0 or 1; unknown labels are rejected.")
                yield FraudSample(self.root / row["path"], self.task, int(row["label"]), row["source_id"], row["split"],
                                  {key: value for key, value in row.items() if key not in required})
