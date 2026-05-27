from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
except ImportError:  # pragma: no cover
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]


from modules.utils import load_yaml  # noqa: F401


@dataclass
class GUIState:
    face_ok: bool = False
    doc_ok: bool = False
    streaming: bool = False


def _require_pyqt6() -> None:
    if QtCore is None or QtWidgets is None or QtGui is None:
        raise RuntimeError("Missing dependency: PyQt6. Install with: pip install PyQt6")


if QtCore is None or QtWidgets is None or QtGui is None:
    class GUIOrchestrator:  # type: ignore[no-redef]
        """Import-safe stub when PyQt6 is not installed."""

        def __init__(self, controller: Any) -> None:
            _require_pyqt6()
else:

    def _cv_to_qimage(frame_bgr: "np.ndarray") -> "QtGui.QImage":
        assert cv2 is not None
        assert np is not None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        bytes_per_line = w * 3
        return QtGui.QImage(
            frame_rgb.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        ).copy()

    class GUIOrchestrator(QtCore.QObject):
        """GUI wrapper that controls controller lifecycle and user actions."""

        def __init__(self, controller: Any) -> None:
            super().__init__()
            self.controller = controller
            self.state = GUIState()

            self._app: Optional[QtWidgets.QApplication] = None
            self._win: Optional[QtWidgets.QMainWindow] = None

            self._closed = threading.Event()

            self._cap: Optional["cv2.VideoCapture"] = None
            self._preview_thread: Optional[threading.Thread] = None
            self._stop_preview = threading.Event()

            self._build()

        def _build(self) -> None:
            assert QtWidgets is not None
            assert QtCore is not None
            assert QtGui is not None
            assert cv2 is not None

            self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

            self._win = QtWidgets.QMainWindow()
            self._win.setWindowTitle(
                self.controller.settings.get("app", {}).get("title", "AI Identity & Avatar Automation")
            )
            self._win.resize(1220, 740)

            central = QtWidgets.QWidget()
            self._win.setCentralWidget(central)

            layout = QtWidgets.QHBoxLayout(central)

            left = QtWidgets.QVBoxLayout()
            layout.addLayout(left, 3)

            self.preview_label = QtWidgets.QLabel("Live Webcam Preview")
            self.preview_label.setMinimumSize(720, 540)
            self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            left.addWidget(self.preview_label)

            controls = QtWidgets.QGridLayout()
            left.addLayout(controls)

            self.btn_face_verify = QtWidgets.QPushButton("Run Face Verification")
            self.btn_doc_verify = QtWidgets.QPushButton("Run Document Verification")
            self.btn_start_stream = QtWidgets.QPushButton("Start Streaming")
            self.btn_stop_stream = QtWidgets.QPushButton("Stop")

            controls.addWidget(self.btn_face_verify, 0, 0)
            controls.addWidget(self.btn_doc_verify, 0, 1)
            controls.addWidget(self.btn_start_stream, 1, 0, 1, 2)
            controls.addWidget(self.btn_stop_stream, 2, 0, 1, 2)

            self.lbl_face = QtWidgets.QLabel("Face: Idle")
            self.lbl_doc = QtWidgets.QLabel("Document: Idle")
            self.lbl_vcam = QtWidgets.QLabel("Virtual Camera: Disabled")
            self.lbl_stream = QtWidgets.QLabel("Streaming: No")

            controls.addWidget(self.lbl_face, 3, 0, 1, 2)
            controls.addWidget(self.lbl_doc, 4, 0, 1, 2)
            controls.addWidget(self.lbl_vcam, 5, 0, 1, 2)
            controls.addWidget(self.lbl_stream, 6, 0, 1, 2)

            right = QtWidgets.QVBoxLayout()
            layout.addLayout(right, 2)

            right.addWidget(QtWidgets.QLabel("Avatar selection (from /avatars)"))
            self.avatar_combo = QtWidgets.QComboBox()

            avatars = getattr(self.controller, "avatar_engine", None)
            if avatars is not None and getattr(avatars, "avatars", None):
                self.avatar_combo.addItems([a.label for a in avatars.avatars])
            else:
                self.avatar_combo.addItems(["(Load streaming to populate)"])

            self.btn_refresh_avatars = QtWidgets.QPushButton("Refresh Avatars")
            right.addWidget(self.avatar_combo)
            right.addWidget(self.btn_refresh_avatars)

            right.addWidget(QtWidgets.QLabel("Status Logs"))
            self.log_box = QtWidgets.QPlainTextEdit()
            self.log_box.setReadOnly(True)
            right.addWidget(self.log_box, 1)

            self.btn_face_verify.clicked.connect(self._on_face_verify)
            self.btn_doc_verify.clicked.connect(self._on_doc_verify)
            self.btn_start_stream.clicked.connect(self._on_start_stream)
            self.btn_stop_stream.clicked.connect(self._on_stop_stream)
            self.btn_refresh_avatars.clicked.connect(self._on_refresh_avatars)
            self.avatar_combo.currentIndexChanged.connect(self._on_avatar_changed)

            enabled = bool(self.controller.settings.get("virtual_camera", {}).get("enabled", False))
            self.lbl_vcam.setText(f"Virtual Camera: {'Enabled' if enabled else 'Disabled'}")

            self._start_preview()

        def log(self, msg: str) -> None:
            ts = time.strftime("%H:%M:%S")
            self.log_box.appendPlainText(f"[{ts}] {msg}")

        def _start_preview(self) -> None:
            cam_cfg = self.controller.settings.get("camera", {})
            device_index = int(cam_cfg.get("device_index", 0))
            width = int(cam_cfg.get("width", 1280))
            height = int(cam_cfg.get("height", 720))

            self._cap = cv2.VideoCapture(device_index)
            if not self._cap.isOpened():
                self.log("Webcam preview: failed to open")
                return

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            self._stop_preview.clear()

            def loop() -> None:
                while not self._stop_preview.is_set():
                    ok, frame = self._cap.read()
                    if not ok or frame is None:
                        time.sleep(0.01)
                        continue
                    qimg = _cv_to_qimage(frame)
                    self.preview_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
                    time.sleep(0.001)

            self._preview_thread = threading.Thread(target=loop, daemon=True)
            self._preview_thread.start()

            def handle_close(_: Any = None) -> None:
                self._stop_preview.set()
                try:
                    if self._cap is not None:
                        self._cap.release()
                except Exception:
                    pass
                try:
                    self.controller.stop_streaming()
                except Exception:
                    pass
                self._closed.set()

            assert self._win is not None
            assert self._app is not None
            self._win.closeEvent = lambda event: (handle_close(), event.accept())  # type: ignore[method-assign]

        def start(self) -> None:
            assert self._win is not None
            assert self._app is not None
            self._win.show()
            self._app.exec()

        def wait_until_closed(self) -> None:
            self._closed.wait()

        def _run_in_thread(self, fn: Any) -> None:
            threading.Thread(target=fn, daemon=True).start()

        def _on_face_verify(self) -> None:
            self.log("Starting face verification...")
            self.lbl_face.setText("Face: Running...")

            def work() -> None:
                try:
                    ok, msg = self.controller.face_verification()
                    self.state.face_ok = ok
                    self.lbl_face.setText(f"Face: {'Approved' if ok else 'Denied'}")
                    self.log(msg)
                except Exception as e:
                    self.lbl_face.setText("Face: Error")
                    self.log(f"Face verification error: {e}")

            self._run_in_thread(work)

        def _on_doc_verify(self) -> None:
            self.log("Starting document verification...")
            self.lbl_doc.setText("Document: Running...")

            def work() -> None:
                try:
                    ok, msg = self.controller.document_verification()
                    self.state.doc_ok = ok
                    self.lbl_doc.setText(f"Document: {'Approved' if ok else 'Denied'}")
                    self.log(msg)
                except Exception as e:
                    self.lbl_doc.setText("Document: Error")
                    self.log(f"Document verification error: {e}")

            self._run_in_thread(work)

        def _on_start_stream(self) -> None:
            if not (self.state.face_ok and self.state.doc_ok):
                self.log("Cannot start: run face + document verification first")
                return

            self.log("Starting avatar + streaming pipeline...")
            self.lbl_stream.setText("Streaming: Starting...")

            def work() -> None:
                try:
                    self.controller.start_streaming()
                    self.state.streaming = True
                    self.lbl_stream.setText("Streaming: Yes")
                    self.log("Streaming started")
                    self._refresh_avatars_ui()
                except Exception as e:
                    self.lbl_stream.setText("Streaming: Error")
                    self.log(f"Start streaming error: {e}")

            self._run_in_thread(work)

        def _on_stop_stream(self) -> None:
            self.log("Stopping streaming...")
            try:
                self.controller.stop_streaming()
            except Exception as e:
                self.log(f"Stop error: {e}")
            self.state.streaming = False
            self.lbl_stream.setText("Streaming: No")

        def _on_avatar_changed(self) -> None:
            engine = getattr(self.controller, "avatar_engine", None)
            if engine is None:
                return
            try:
                idx = self.avatar_combo.currentIndex()
                engine.switch_avatar(idx)
            except Exception:
                pass

        def _refresh_avatars_ui(self) -> None:
            engine = getattr(self.controller, "avatar_engine", None)
            if engine is None or not getattr(engine, "avatars", None):
                self.avatar_combo.clear()
                self.avatar_combo.addItems(["(No avatars loaded)"])
                return
            self.avatar_combo.clear()
            self.avatar_combo.addItems([a.label for a in engine.avatars])
            self.log("Avatars loaded")

        def _on_refresh_avatars(self) -> None:
            try:
                self._refresh_avatars_ui()
            except Exception as e:
                self.log(f"Refresh avatars error: {e}")
