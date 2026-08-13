from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
    override = os.getenv("HALLUCINATE_GATE_CONFIG_DIR")
    if override:
        return Path(override)
    packaged = Path(__file__).resolve().parent / "config"
    if (packaged / "thresholds.yaml").exists():
        return packaged
    return Path(__file__).resolve().parents[2] / "config"


def config_file(name: str) -> Path:
    return config_dir() / name
