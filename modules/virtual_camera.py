from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def require_dependency(dep: Any, package_name: str) -> None:
    if dep is None:
        raise RuntimeError(f"Missing dependency: {package_name}. Install with: pip install {package_name}")


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

    def _require_numpy(self) -> None:
        require_dependency(np, "numpy")

    def _require_pyvirtualcam(self) -> None:
        try:
            import pyvirtualcam  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: pyvirtualcam. Install with: pip install pyvirtualcam") from e

    # Required functions (per your architecture)
    def init_camera(self) -> None:
        self._require_numpy()
        if not self.config.enabled:
            return

        self._require_pyvirtualcam()
        import pyvirtualcam  # type: ignore

        self._cam = pyvirtualcam.Camera(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            fmt=self.config.pixel_format,
            device=self.config.device_index,
        )

    def send_frame(self, frame: np.ndarray) -> None:
        self._require_numpy()
        if not self.config.enabled:
            return
        if frame is None:
            return

        if self._cam is None:
            self.init_camera()

        # Ensure size
        if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
            import cv2  # type: ignore

            frame = cv2.resize(frame, (self.config.width, self.config.height))

        # Ensure correct channel count if BGRA
        if self.config.pixel_format.lower() == "bgra":
            if frame.ndim == 3 and frame.shape[2] == 3:
                import cv2  # type: ignore

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        self._cam.send(frame)

    def close_camera(self) -> None:
        self._cam = None

    # Backwards-compatible methods (existing controller may call these)
    def initialize_virtual_camera(self) -> None:
        return self.init_camera()

    def close_virtual_camera(self) -> None:
        return self.close_camera()

