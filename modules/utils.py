from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import yaml


def load_settings(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class AppPaths:
    avatars_dir: str
    models_dir: str
    logs_dir: str
    encodings_dir: str
    documents_dir: str
    screenshots_dir: str


def ensure_dirs(paths: AppPaths) -> None:
    for p in paths.__dict__.values():
        os.makedirs(p, exist_ok=True)


def utc_timestamp() -> float:
    return time.time()

