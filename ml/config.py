"""Single, dependency-free configuration entry point."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(path or os.getenv("WONDEROFU_CONFIG", root / "configs" / "default.json"))
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["data_root"] = os.getenv("WONDEROFU_DATA_ROOT", config["data_root"])
    return config


def dataset_path(config: dict[str, Any], dataset_name: str) -> Path:
    return Path(config["data_root"]) / config["datasets"][dataset_name]["path"]


def preprocessing_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the single configuration block used by all media preprocessing."""
    return dict(config["preprocessing"])
