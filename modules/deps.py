from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptionalDeps:
    torch: Optional[object] = None
    cv2: Optional[object] = None
    easyocr: Optional[object] = None
    face_recognition: Optional[object] = None


def load_deps() -> OptionalDeps:
    deps = OptionalDeps()

    try:
        import torch  # type: ignore

        deps.torch = torch
    except Exception:
        deps.torch = None

    try:
        import cv2  # type: ignore

        deps.cv2 = cv2
    except Exception:
        deps.cv2 = None

    try:
        import easyocr  # type: ignore

        deps.easyocr = easyocr
    except Exception:
        deps.easyocr = None

    try:
        import face_recognition  # type: ignore

        deps.face_recognition = face_recognition
    except Exception:
        deps.face_recognition = None

    return deps
