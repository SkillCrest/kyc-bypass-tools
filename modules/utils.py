from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


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
        os.makedirs(str(p), exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    return data


def configure_logging(level: str) -> None:
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def utc_timestamp() -> float:
    return time.time()


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def detect_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def enable_fp16_if_possible() -> bool:
    """Returns True if CUDA + autocast usage is feasible.

    This function doesn't globally set FP16; it only checks feasibility.
    """
    return detect_cuda()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

