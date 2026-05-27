from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Avatar:
    label: str
    image_bgr: np.ndarray


class AvatarEngine:
    """Avatar engine manager.

    This implementation focuses on fast, reliable avatar loading/switching.
    Motion retargeting/LivePortrait integration is implemented as extension
    points so the system remains modular.
    """

    def __init__(self, cfg: Dict[str, Any], avatars_dir: str):
        self.cfg = cfg
        self.avatars_dir = Path(avatars_dir)
        self.avatars: List[Avatar] = []
        self.active_index: int = 0

    def load_avatar(self, image_path: str) -> Avatar:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read avatar image: {p}")
        label = p.stem
        return Avatar(label=label, image_bgr=img)

    def load_avatar_directory(self) -> List[Avatar]:
        self.avatars = []
        if not self.avatars_dir.exists():
            return self.avatars

        allowed = {".png", ".jpg", ".jpeg", ".webp"}
        for entry in sorted(self.avatars_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in allowed:
                continue
            try:
                self.avatars.append(self.load_avatar(str(entry)))
            except Exception:
                continue
        self.active_index = 0
        return self.avatars

    def switch_avatar(self, label_or_index: str | int) -> None:
        if isinstance(label_or_index, int):
            if not (0 <= label_or_index < len(self.avatars)):
                raise IndexError("Avatar index out of range")
            self.active_index = label_or_index
            return

        # label
        for i, a in enumerate(self.avatars):
            if a.label == str(label_or_index):
                self.active_index = i
                return
        raise ValueError(f"Avatar not found: {label_or_index}")

    def _process_motion_extension(
        self,
        frame_bgr: np.ndarray,
        avatar: Avatar,
        face_landmarks: Optional[Any] = None,
    ) -> np.ndarray:
        """Extension point.

        Replace with MediaPipe / LivePortrait / FacePoke integration.
        """
        return self.animate_frame(frame_bgr, avatar)

    def process_webcam_motion(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.avatars:
            raise RuntimeError("No avatars loaded")
        avatar = self.avatars[self.active_index]
        return self._process_motion_extension(frame_bgr=frame_bgr, avatar=avatar)

    def animate_frame(self, frame_bgr: np.ndarray, avatar: Avatar) -> np.ndarray:
        """Fast compositing: overlays avatar image centered on webcam frame."""
        h, w = frame_bgr.shape[:2]

        # Resize avatar to fit the frame height with margins
        ah, aw = avatar.image_bgr.shape[:2]
        target_h = int(h * 0.9)
        scale = target_h / max(ah, 1)
        new_w = max(1, int(aw * scale))
        resized = cv2.resize(avatar.image_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)

        # Center horizontally, bottom aligned
        x1 = max(0, (w - new_w) // 2)
        y1 = max(0, h - target_h)

        out = frame_bgr.copy()
        y2 = min(h, y1 + target_h)
        x2 = min(w, x1 + new_w)

        roi = out[y1:y2, x1:x2]
        src = resized[0 : y2 - y1, 0 : x2 - x1]
        # Simple alpha-free overlay; improve later with masks.
        out[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, src, 0.7, 0)
        return out

