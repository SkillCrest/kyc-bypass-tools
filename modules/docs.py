from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None


@dataclass
class DocumentResult:
    approved: bool
    text: str
    reason: str = ""


class DocumentVerificationService:
    """Document OCR + basic verification.

    Lightweight modular implementation.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.languages = cfg.get("languages", ["en"])
        self.min_text_chars = int(cfg.get("min_text_chars", 10))
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.4))

        self.reader = None
        if easyocr is not None:
            # EasyOCR uses CPU by default; gpu=True will use CUDA when available.
            self.reader = easyocr.Reader(self.languages, gpu=True)

    def capture_document(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("No document frame provided")
        return frame

    def preprocess_document(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
        return th

    def extract_text(self, preprocessed: np.ndarray) -> Tuple[str, float]:
        if self.reader is None:
            return "", 0.0

        if len(preprocessed.shape) == 2:
            rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        results = self.reader.readtext(rgb)
        if not results:
            return "", 0.0

        texts = []
        confs = []
        for _, text, conf in results:
            if text:
                texts.append(text)
                confs.append(float(conf))

        combined = "\n".join(texts).strip()
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return combined, avg_conf

    def _basic_verify(self, text: str, avg_conf: float) -> DocumentResult:
        if len(text) < self.min_text_chars:
            return DocumentResult(False, text, reason="Insufficient OCR text")
        if avg_conf < self.confidence_threshold:
            return DocumentResult(False, text, reason="Low OCR confidence")
        return DocumentResult(True, text, reason="Approved")

    def verify_document(self, frame_bgr: np.ndarray) -> DocumentResult:
        captured = self.capture_document(frame_bgr)
        pre = self.preprocess_document(captured)
        text, avg_conf = self.extract_text(pre)
        return self._basic_verify(text, avg_conf)

