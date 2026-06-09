"""
face_detector.py
================
Detects faces/persons in images using a two-stage approach:
  1. OpenCV DNN (SSD ResNet-10) — high-precision face detector.
  2. Haar Cascade fallback — lightweight, works without model weights.

Download the DNN weights from:
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
Place them in the models/ directory.
"""

import os
import logging

import cv2
import numpy as np

from config import (
    FACE_PROTO, FACE_MODEL,
    FACE_CONF_THRESHOLD, DNN_INPUT_SIZE,
)

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects faces in BGR images.

    Tries OpenCV DNN first (if model weights are present).
    Falls back to Haar Cascade if DNN weights are missing.
    """

    def __init__(self,
                 proto: str = FACE_PROTO,
                 model: str = FACE_MODEL,
                 conf_threshold: float = FACE_CONF_THRESHOLD):
        self.conf_threshold = conf_threshold
        self._net = None
        self._haar = None
        self._mode = "none"

        # ── Try DNN ──────────────────────────────────────────────────────────
        if os.path.exists(proto) and os.path.exists(model):
            try:
                self._net = cv2.dnn.readNetFromCaffe(proto, model)
                self._mode = "dnn"
                logger.info("FaceDetector: using OpenCV DNN (SSD ResNet-10).")
            except Exception as exc:
                logger.warning("DNN load failed (%s). Falling back to Haar.", exc)

        # ── Fallback: MTCNN ───────────────────────────────────────────────────
        if self._mode == "none":
            try:
                from mtcnn import MTCNN
                self._mtcnn = MTCNN()
                self._mode = "mtcnn"
                logger.info("FaceDetector: using MTCNN fallback.")
            except ImportError:
                raise RuntimeError("Could not load MTCNN. Run pip install mtcnn.")

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect all faces in *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3).

        Returns
        -------
        list of (x, y, w, h) bounding boxes.
        """
        if self._mode == "dnn":
            return self._detect_dnn(frame)
        return self._detect_mtcnn(frame)

    def _detect_dnn(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, DNN_INPUT_SIZE),
            scalefactor=1.0,
            size=DNN_INPUT_SIZE,
            mean=(104.0, 177.0, 123.0),
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < self.conf_threshold:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def _detect_mtcnn(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mtcnn.detect_faces(img_rgb)
        boxes = []
        for res in results:
            if res.get('confidence', 0) >= self.conf_threshold:
                x, y, w, h = res['box']
                # clamp
                x, y = max(0, x), max(0, y)
                boxes.append((int(x), int(y), int(w), int(h)))
        return boxes

    @property
    def mode(self) -> str:
        """Return 'dnn' or 'haar' depending on which backend is active."""
        return self._mode
