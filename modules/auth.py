from __future__ import annotations


import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import face_recognition  # type: ignore
except ImportError:  # pragma: no cover
    face_recognition = None  # type: ignore[assignment]

from modules.utils import utc_timestamp


def require_dependency(dep: Any, package_name: str) -> None:
    if dep is None:
        raise RuntimeError(f"Missing dependency: {package_name}. Install it, e.g.: pip install {package_name}")



@dataclass
class FaceEncodings:
    encodings: List[Any]
    labels: List[str]


class AuthenticationService:
    
    def _require_face_recognition(self) -> None:
        require_dependency(face_recognition, "face_recognition")

    def _require_cv2(self) -> None:
        require_dependency(cv2, "opencv-python")

    
    def __init__(self, cfg: Dict[str, Any], models_dir: str):
        self.cfg = cfg
        self.models_dir = models_dir
        self.tolerance = float(cfg.get("tolerance", 0.55))
        self.model = str(cfg.get("face_detection_model", "hog"))

    def _encoding_path(self, filename: str) -> Path:
        return Path(self.models_dir) / filename

    def load_encoding(self, filename: str) -> Optional[FaceEncodings]:
        path = self._encoding_path(filename)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        return FaceEncodings(encodings=data["encodings"], labels=data["labels"])

    def save_encoding(self, filename: str, encodings: List[Any], labels: List[str]) -> None:
        path = self._encoding_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"encodings": encodings, "labels": labels}, f)

    def enroll_face(
        self,
        person_label: str,
        webcam_capture_device_index: int = 0,
        max_attempts: int = 30,
    ) -> FaceEncodings:
        """Enroll a face from webcam.

        Captures frames, detects face, and stores the first stable encoding found.
        """
        self._require_cv2()
        self._require_face_recognition()

        cap = cv2.VideoCapture(webcam_capture_device_index)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam")

        known_encodings: List[Any] = []
        labels: List[str] = []

        try:
            attempts = 0
            while attempts < max_attempts and not known_encodings:
                ok, frame = cap.read()
                if not ok:
                    attempts += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb, model=self.model)
                encs = face_recognition.face_encodings(rgb, boxes)
                if encs:
                    # Use first detected face encoding
                    known_encodings.append(encs[0])
                    labels.append(person_label)
                    break

                attempts += 1
        finally:
            cap.release()

        if not known_encodings:
            raise RuntimeError("No face encoding found during enrollment")

        return FaceEncodings(encodings=known_encodings, labels=labels)

    def verify_face(
        self,
        known: FaceEncodings,
        webcam_capture_device_index: int = 0,
        max_checks: int = 60,
    ) -> Tuple[bool, Optional[str], float]:
        """Verify face from webcam against known encodings."""
        if not known.encodings:
            return False, None, 0.0

        cap = cv2.VideoCapture(webcam_capture_device_index)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam")

        try:
            checks = 0
            while checks < max_checks:
                ok, frame = cap.read()
                if not ok:
                    checks += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb, model=self.model)
                if not boxes:
                    checks += 1
                    continue

                encs = face_recognition.face_encodings(rgb, boxes)
                if not encs:
                    checks += 1
                    continue

                # Compare first face
                face_encoding = encs[0]
                matches = face_recognition.compare_faces(
                    known.encodings,
                    face_encoding,
                    tolerance=self.tolerance,
                )

                if any(matches):
                    best_index = matches.index(True)
                    # Approx confidence: smaller distance => better
                    distances = face_recognition.face_distance(known.encodings, face_encoding)
                    conf = float(1.0 - min(distances) / 2.0)  # heuristic
                    return True, known.labels[best_index], conf

                checks += 1
        finally:
            cap.release()

        return False, None, 0.0

