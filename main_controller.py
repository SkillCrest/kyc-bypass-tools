from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


from modules.auth import AuthenticationService
from modules.avatar_engine import AvatarEngine
from modules.docs import DocumentVerificationService
from modules.gui import GUIOrchestrator
from modules.utils import AppPaths, configure_logging, ensure_dirs, load_yaml
from modules.virtual_camera import VirtualCameraManager


@dataclass
class SystemState:
    face_verified: bool = False
    doc_verified: bool = False


class AIIdentityAvatarController:
    """Central orchestration.

    FLOW (required):
    Face verification → Document verification → Avatar engine → MediaPipe tracking → Virtual camera stream → GUI control
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.settings: dict[str, Any] = load_yaml(config_path)

        self.project_root = Path(__file__).resolve().parent
        self.paths = AppPaths(
            avatars_dir=str(self.project_root / "avatars"),
            models_dir=str(self.project_root / "models"),
            logs_dir=str(self.project_root / "logs"),
            encodings_dir=str(self.project_root / "models" / "encodings"),
            documents_dir=str(self.project_root / "assets" / "documents"),
            screenshots_dir=str(self.project_root / "assets" / "screenshots"),
        )
        ensure_dirs(self.paths)

        configure_logging(self.settings.get("app", {}).get("log_level", "INFO"))
        self.logger = logging.getLogger("AIIdentityAvatarController")

        # CUDA / FP16 enablement (only feasibility + perf flags)
        self.cuda_available = bool(torch.cuda.is_available())
        self.fp16_enabled = self.cuda_available
        if self.cuda_available:
            torch.backends.cudnn.benchmark = True
            self.logger.info("CUDA detected. FP16 feasible=%s", self.fp16_enabled)
        else:
            self.logger.warning("CUDA not available. Using CPU.")

        # Lazy-load heavy AI models where possible
        self.auth_service = AuthenticationService(cfg=self.settings.get("auth", {}), models_dir=self.paths.encodings_dir)
        self.doc_service = DocumentVerificationService(cfg=self.settings.get("documents", {}))

        self.avatar_engine: Optional[AvatarEngine] = None
        self.virtual_camera: Optional[VirtualCameraManager] = None

        self.state = SystemState()

        self._stop_event = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None

    # ---- Pipeline steps (invoked by GUI) ----

    def face_verification(self) -> tuple[bool, str]:
        cam_cfg = self.settings.get("camera", {})
        device_index = int(cam_cfg.get("device_index", 0))

        known_file = self.settings.get("auth", {}).get("known_faces_encoding_file", "known_faces.pkl")
        known = self.auth_service.load_encoding(known_file)
        if known is None:
            return False, "No enrolled face encoding found. Enroll first."

        ok, label, conf = self.auth_service.verify_face(
            known=known,
            webcam_capture_device_index=device_index,
            max_checks=int(self.settings.get("auth", {}).get("verification_max_checks", 90)),
        )

        if ok:
            self.state.face_verified = True
            return True, f"Face verified: {label} (conf={conf:.2f})"
        return False, "Face verification denied"

    def document_verification(self) -> tuple[bool, str]:
        cam_cfg = self.settings.get("camera", {})
        device_index = int(cam_cfg.get("device_index", 0))

        ok, frame = self._capture_one_frame(device_index=device_index)
        if not ok or frame is None:
            return False, "Document capture failed"

        result = self.doc_service.verify_document(frame)
        if result.approved:
            self.state.doc_verified = True
            return True, "Document OCR approved"
        return False, f"Document denied: {result.reason}"

    def start_streaming(self) -> None:
        if not (self.state.face_verified and self.state.doc_verified):
            raise RuntimeError("Face and document must be verified before streaming")

        self._lazy_init_avatar()
        self._lazy_init_virtual_camera()

        assert self.avatar_engine is not None

        cam_cfg = self.settings.get("camera", {})
        device_index = int(cam_cfg.get("device_index", 0))
        width = int(cam_cfg.get("width", 1280))
        height = int(cam_cfg.get("height", 720))
        fps = int(cam_cfg.get("fps", 30))

        import cv2

        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam for streaming")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        self._stop_event.clear()

        def loop() -> None:
            last = time.time()
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                try:
                    rendered = self.avatar_engine.render_frame(frame)
                except Exception as e:
                    self.logger.exception("Avatar render error: %s", e)
                    rendered = frame

                if self.virtual_camera is not None and self.virtual_camera.config.enabled:
                    try:
                        self.virtual_camera.send_frame(rendered)
                    except Exception as e:
                        self.logger.exception("Virtual camera send error: %s", e)

                now = time.time()
                dt = now - last
                target = 1.0 / max(1, fps)
                if dt < target:
                    time.sleep(target - dt)
                last = time.time()

            cap.release()

        self._stream_thread = threading.Thread(target=loop, daemon=True)
        self._stream_thread.start()

    def stop_streaming(self) -> None:
        self._stop_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
        if self.virtual_camera is not None:
            self.virtual_camera.close_camera()

    def run(self) -> None:
        gui = GUIOrchestrator(controller=self)
        gui_thread = threading.Thread(target=gui.start, daemon=True)
        gui_thread.start()
        gui.wait_until_closed()

    # ---- internal ----

    def _lazy_init_avatar(self) -> None:
        if self.avatar_engine is None:
            self.avatar_engine = AvatarEngine(cfg=self.settings, avatars_dir=self.paths.avatars_dir)
            self.avatar_engine.load_avatars()

    def _lazy_init_virtual_camera(self) -> None:
        if self.virtual_camera is None:
            self.virtual_camera = VirtualCameraManager(cfg=self.settings)
            self.virtual_camera.init_camera()

    def _capture_one_frame(self, device_index: int) -> tuple[bool, Optional[Any]]:
        import cv2

        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return False, None

        try:
            cap.read()
            ok, frame = cap.read()
            return bool(ok), frame
        finally:
            cap.release()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config_path = str(project_root / "config" / "settings.yaml")

    controller = AIIdentityAvatarController(config_path=config_path)
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.logger.info("Interrupted by user")
        controller.stop_streaming()
    except Exception as e:
        controller.logger.exception("Fatal error: %s", e)
        controller.stop_streaming()


if __name__ == "__main__":
    main()

