from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class VirtualCameraConfig:
    enabled: bool
    device_index: int
    fps: int
    width: int
    height: int
    pixel_format: str = "bgra"


class VirtualCameraManager:
    def __init__(self, cfg: Dict[str, Any]):
        vc = cfg.get("virtual_camera", {})
        self.config = VirtualCameraConfig(
            enabled=bool(vc.get("enabled", False)),
            device_index=int(vc.get("device_index", 0)),
            fps=int(vc.get("fps", 30)),
            width=int(vc.get("width", 1280)),
            height=int(vc.get("height", 720)),
            pixel_format=str(vc.get("pixel_format", "bgra")),
        )

        self._cam: Optional[Any] = None

    def initialize_virtual_camera(self) -> None:
        if not self.config.enabled:
            return

        try:
            import pyvirtualcam
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyvirtualcam is not installed or failed to import") from e

        # pyvirtualcam expects rgb format order in docs; but supports numpy arrays.
        # We will supply BGRA by default, converting to RGBA if needed later.
        self._cam = pyvirtualcam.Camera(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            fmt=self.config.pixel_format,
            device=self.config.device_index,
        )

    def send_frame(self, frame: np.ndarray) -> None:
        if not self.config.enabled:
            return
        if self._cam is None:
            self.initialize_virtual_camera()

        if frame is None:
            return

        # Ensure size
        if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
            import cv2

            frame = cv2.resize(frame, (self.config.width, self.config.height))

        # Ensure correct channel count if BGRA
        if self.config.pixel_format.lower() == "bgra":
            if frame.shape[2] == 3:
                import cv2

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        self._cam.send(frame)

    def close_virtual_camera(self) -> None:
        if self._cam is not None:
            # pyvirtualcam doesn't always require explicit close; delete reference.
            self._cam = None

