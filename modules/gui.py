from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np

from modules.avatar_engine import AvatarEngine
from modules.docs import DocumentVerificationService
from modules.networking_stub import StatusBroker  # type: ignore
from modules.virtual_camera import VirtualCameraManager

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyQt6 is required to use the GUI") from e

from modules.auth import AuthenticationService, FaceEncodings
from modules.utils import ensure_dirs, AppPaths


@dataclass
class UIState:
    auth_status: str = "Idle"
    doc_status: str = "Idle"


def _cv_to_qimage(frame_bgr: np.ndarray) -> QtGui.QImage:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    bytes_per_line = w * 3
    return QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()


class AppWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: Dict[str, Any], project_root: str):
        super().__init__()
        self.cfg = cfg
        self.project_root = project_root

        self.setWindowTitle(cfg.get("app", {}).get("title", "AI Identity & Avatar Automation"))
        self.resize(1200, 720)

        paths_cfg = cfg.get("paths", {})
        app_paths = AppPaths(
            avatars_dir=str(paths_cfg.get("avatars_dir", "avatars")),
            models_dir=str(paths_cfg.get("models_dir", "models")),
            logs_dir=str(paths_cfg.get("logs_dir", "logs")),
            encodings_dir=str(paths_cfg.get("encodings_dir", "models/encodings")),
            documents_dir=str(paths_cfg.get("documents_dir", "assets/documents")),
            screenshots_dir=str(paths_cfg.get("screenshots_dir", "assets/screenshots")),
        )
        ensure_dirs(app_paths)

        self.avatar_engine = AvatarEngine(cfg=cfg, avatars_dir=app_paths.avatars_dir)
        self.avatar_engine.load_avatar_directory()

        self.doc_service = DocumentVerificationService(cfg=cfg.get("documents", {}))
        self.virtual_camera = VirtualCameraManager(cfg=cfg)

        self.auth_service = AuthenticationService(cfg=cfg.get("auth", {}), models_dir=app_paths.encodings_dir)

        self.state = UIState()

        self._cap: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._latest_frame: Optional[np.ndarray] = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        # Left: preview
        left = QtWidgets.QVBoxLayout()
        self.preview = QtWidgets.QLabel("Webcam preview")
        self.preview.setMinimumSize(640, 480)
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.preview)

        # Controls
        controls = QtWidgets.QGridLayout()
        left.addLayout(controls)

        self.btn_auth = QtWidgets.QPushButton("Login (Face)")
        self.btn_doc = QtWidgets.QPushButton("Verify Document (OCR)")
        self.btn_toggle_vcam = QtWidgets.QPushButton("Virtual Camera")

        self.txt_auth = QtWidgets.QLabel("Auth: Idle")
        self.txt_doc = QtWidgets.QLabel("Doc: Idle")

        controls.addWidget(self.btn_auth, 0, 0)
        controls.addWidget(self.btn_doc, 0, 1)
        controls.addWidget(self.btn_toggle_vcam, 1, 0, 1, 2)
        controls.addWidget(self.txt_auth, 2, 0, 1, 2)
        controls.addWidget(self.txt_doc, 3, 0, 1, 2)

        layout.addLayout(left, 3)

        # Right: avatar selection + status
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 2)

        right.addWidget(QtWidgets.QLabel("Avatar selector"))
        self.avatar_combo = QtWidgets.QComboBox()
        self.avatar_combo.addItems([a.label for a in self.avatar_engine.avatars] or ["(No avatars found)"])
        self.avatar_combo.currentIndexChanged.connect(self._on_avatar_changed)
        right.addWidget(self.avatar_combo)

        right.addWidget(QtWidgets.QLabel("Status"))
        self.status_list = QtWidgets.QPlainTextEdit()
        self.status_list.setReadOnly(True)
        right.addWidget(self.status_list)

        self.btn_auth.clicked.connect(self._handle_auth)
        self.btn_doc.clicked.connect(self._handle_doc)
        self.btn_toggle_vcam.clicked.connect(self._toggle_vcam)

    def log(self, msg: str) -> None:
        self.status_list.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _on_avatar_changed(self) -> None:
        if not self.avatar_engine.avatars:
            return
        idx = self.avatar_combo.currentIndex()
        try:
            self.avatar_engine.switch_avatar(idx)
        except Exception:
            return

    def start_webcam(self) -> None:
        cam_cfg = self.cfg.get("camera", {})
        device_index = int(cam_cfg.get("device_index", 0))
        width = int(cam_cfg.get("width", 1280))
        height = int(cam_cfg.get("height", 720))

        self._cap = cv2.VideoCapture(device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        def loop() -> None:
            while not self._stop.is_set():
                if self._cap is None:
                    break
                ok, frame = self._cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                # Avatar processing (extension point)
                try:
                    processed = self.avatar_engine.process_webcam_motion(frame)
                except Exception:
                    processed = frame

                self._latest_frame = processed
                # UI update in Qt thread
                self._update_preview(processed)

                time.sleep(0.001)

        self._capture_thread = threading.Thread(target=loop, daemon=True)
        self._capture_thread.start()

    def _update_preview(self, frame_bgr: np.ndarray) -> None:
        img = _cv_to_qimage(frame_bgr)
        self.preview.setPixmap(QtGui.QPixmap.fromImage(img))

    def _handle_auth(self) -> None:
        self.log("Starting face enrollment/verification...")
        self.txt_auth.setText("Auth: Running...")

        # Basic: load known encodings file configured in settings and verify.
        known_file = self.cfg.get("auth", {}).get("known_faces_encoding_file", "known_faces.pkl")
        known = self.auth_service.load_encoding(known_file)
        if known is None:
            self.txt_auth.setText("Auth: No enrolled faces")
            self.log("No encoding file found. Enroll first.")
            return

        def worker() -> None:
            try:
                ok, label, conf = self.auth_service.verify_face(
                    known,
                    webcam_capture_device_index=int(self.cfg.get("camera", {}).get("device_index", 0)),
                )
                if ok:
                    self.txt_auth.setText(f"Auth: Approved ({label}) conf={conf:.2f}")
                    self.log(f"Face verified: {label} (conf={conf:.2f})")
                else:
                    self.txt_auth.setText("Auth: Denied")
                    self.log("Face verification denied")
            except Exception as e:
                self.txt_auth.setText("Auth: Error")
                self.log(f"Auth error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _handle_doc(self) -> None:
        self.log("Starting document OCR...")
        self.txt_doc.setText("Doc: Running...")

        frame = self._latest_frame
        if frame is None:
            self.txt_doc.setText("Doc: No frame")
            return

        def worker() -> None:
            try:
                result = self.doc_service.verify_document(frame)
                if result.approved:
                    self.txt_doc.setText(f"Doc: Approved")
                else:
                    self.txt_doc.setText(f"Doc: Denied ({result.reason})")
                self.log(f"Document OCR: approved={result.approved}, reason={result.reason}")
            except Exception as e:
                self.txt_doc.setText("Doc: Error")
                self.log(f"Doc error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_vcam(self) -> None:
        if self.virtual_camera.config.enabled:
            # Keep it simple: initialize once; sending happens in webcam loop.
            self.virtual_camera.initialize_virtual_camera()
            self.log("Virtual camera initialized")
        else:
            self.log("Virtual camera disabled in config")

    def closeEvent(self, event):  # noqa: N802
        self._stop.set()
        if self._cap is not None:
            self._cap.release()
        self.virtual_camera.close_virtual_camera()
        event.accept()


def build_and_run_gui(settings: Dict[str, Any], project_root: str) -> None:
    app = QtWidgets.QApplication([])
    win = AppWindow(cfg=settings, project_root=project_root)
    win.show()
    win.start_webcam()
    app.exec()

