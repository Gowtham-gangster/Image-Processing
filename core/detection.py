"""
detection.py
============
Real-time face and upper-body detection for CCTV frames.

Detection strategy (tried in order)
-------------------------------------
1. **OpenCV DNN** (SSD ResNet-10) — extremely fast face detector.
2. **YOLOv8** (yolov8n.pt) — state-of-the-art person detector that isolates bodies and pairs them with faces.

The module exposes:
  - :class:`Detector`       — main class; use in the recognition pipeline.
  - :func:`run_live`        — convenience function for a quick webcam demo.
  - :func:`detect_from_image` — one-shot detection on a file path.

Usage (module)
--------------
    from detection import Detector
    from person_identifier import PersonIdentifier
    from database import PersonDatabase

    detector = Detector()
    db       = PersonDatabase()
    ident    = PersonIdentifier(db=db)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        detections = detector.detect(frame)           # list[DetectionResult]
        for det in detections:
            result = ident.identify(det.face_roi, is_masked=False)
            # … render, log, etc.

Usage (CLI / demo)
-------------------
    python detection.py                         # live webcam
    python detection.py --source image.jpg      # single image
    python detection.py --source video.mp4      # video file
    python detection.py --backend yolo          # force YOLOv8
    python detection.py --backend dnn           # force DNN (needs weights)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import (
    FACE_PROTO, FACE_MODEL,
    FACE_CONF_THRESHOLD, DNN_INPUT_SIZE,
    COLOUR_KNOWN, COLOUR_UNKNOWN, FONT,
    CAMERA_SOURCE, LOG_LEVEL,
)
from face_detector import FaceDetector

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# Minimum size to filter tiny spurious detections
_MIN_FACE_SIZE       = (40, 40)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    Holds one detected region extracted from a CCTV frame.

    Attributes
    ----------
    bbox        : (x, y, w, h) in pixel coordinates.
    face_roi    : BGR crop of the detected region (ready for the recogniser).
    backend     : 'dnn', 'haar_face', 'haar_profile', or 'haar_upper'.
    confidence  : Detection confidence in [0, 1] (DNN only; 1.0 for Haar).
    """
    bbox: tuple[int, int, int, int]
    face_roi: np.ndarray
    backend: str = "unknown"
    confidence: float = 1.0

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    def __repr__(self) -> str:
        x, y, w, h = self.bbox
        return (
            f"DetectionResult(box=({x},{y},{w},{h}), "
            f"backend={self.backend}, conf={self.confidence:.2f})"
        )


# ── Detector ──────────────────────────────────────────────────────────────────

class Detector:
    """
    Multi-backend face / upper-body detector for CCTV frames.

    Parameters
    ----------
    backend : str
        ``'auto'`` (default) – try DNN first, fall back to YOLO.
        ``'dnn'``            – force OpenCV DNN face detector.
        ``'yolo'``           – force YOLOv8 person detector.
    conf_threshold : float
        Minimum confidence for detections.
    """

    BACKENDS = ("auto", "dnn", "yolo")

    def __init__(
        self,
        backend: str = "auto",
        conf_threshold: float = FACE_CONF_THRESHOLD,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}")

        self.conf_threshold = conf_threshold
        self._face_detector = None
        self._yolo_detector = None

        # ── Load FaceDetector (DNN/Haar) ─────────────────────────────────────
        if backend in ("auto", "dnn"):
            try:
                self._face_detector = FaceDetector(conf_threshold=self.conf_threshold)
            except Exception as e:
                logger.warning("FaceDetector could not be loaded: %s", e)
                if backend == "dnn":
                    raise

        # ── Load YOLO ─────────────────────────────────────────────────────────
        if backend in ("auto", "yolo"):
            try:
                from yolo_person_detector import YoloPersonDetector
                from face_alignment import FaceAligner
                self._yolo_detector = YoloPersonDetector(
                    aligner=FaceAligner(), 
                    conf_threshold=self.conf_threshold
                )
            except Exception as e:
                logger.warning("YOLO detector could not be loaded: %s", e)
                if backend == "yolo":
                    raise

        # Choose effective backend
        if backend == "dnn":
            self._active_backend = "dnn"
        elif backend == "yolo":
            self._active_backend = "yolo"
        else:  # auto
            self._active_backend = "dnn" if self._face_detector and self._face_detector.mode == "dnn" else "yolo"

        logger.info("Detector ready | backend=%s", self._active_backend)

    def detect(self, frame: np.ndarray, max_width: int = 0) -> list[DetectionResult]:
        """
        Detect faces (and optionally upper bodies) in *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3).
        max_width : int
            If > 0, resize frame to this width for faster DNN inference.

        Returns
        -------
        list[DetectionResult]
            Sorted by descending bounding-box area (largest person first).
        """
        if frame is None or frame.size == 0:
            return []

        raw_detections: List[Dict[str, Any]] = []
        results: list[DetectionResult] = []

        if self._active_backend == "dnn" and self._face_detector:
            # FaceDetector.detect_faces returns list of (x,y,w,h)
            boxes = self._face_detector.detect_faces(frame)
            for box in boxes:
                results.append(DetectionResult(
                    bbox=box,
                    face_roi=np.empty((0,)),
                    backend=self._face_detector.mode,
                    confidence=1.0, # FaceDetector doesn't expose confidence per box easily
                ))

        elif self._active_backend == "yolo" and self._yolo_detector:
            # YOLO returns {person_bbox, face_bbox, body_crop, face_crop}
            yolo_results = self._yolo_detector.detect(frame)
            for res in yolo_results:
                # detection.py typically expects face bounded results if possible
                # If we have a face, use it. Else, fall back to body (for bounding box metrics).
                best_box = res["face_bbox"] if res["face_bbox"] is not None else res["person_bbox"]
                if best_box is not None:
                    results.append(DetectionResult(
                        bbox=tuple(best_box),
                        face_roi=np.empty((0,)),
                        backend="yolo",
                        confidence=res["person_conf"]
                    ))

        # Deduplicate overlapping boxes (primarily for Haar, but good for DNN too)
        results = self._nms(results)

        # Extract ROI for every detection
        final: list[DetectionResult] = []
        for det in results:
            roi = self._extract_roi(frame, det.bbox)
            if roi is not None:
                det.face_roi = roi
                final.append(det)

        # Sort biggest first
        final.sort(key=lambda d: d.area, reverse=True)
        return final

    def detect_largest(self, frame: np.ndarray, max_width: int = 0) -> Optional[DetectionResult]:
        """Return only the largest detected region, or None."""
        results = self.detect(frame, max_width=max_width)
        return results[0] if results else None

    @property
    def backend(self) -> str:
        return self._active_backend

    # ── ROI extraction ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_roi(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding: float = 0.10,
    ) -> Optional[np.ndarray]:
        """
        Crop the bounding box from *frame* with an optional padding margin.

        Parameters
        ----------
        padding : float
            Fraction of box width/height to pad on each side.

        Returns
        -------
        np.ndarray (BGR) or None if the crop is invalid.
        """
        h_img, w_img = frame.shape[:2]
        x, y, w, h = bbox

        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return roi

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _nms(
        results: list[DetectionResult],
        iou_threshold: float = 0.40,
    ) -> list[DetectionResult]:
        """Non-maximum suppression to remove duplicate boxes."""
        if not results:
            return []

        # Convert DetectionResult to format suitable for NMSBoxes
        bboxes_list = [list(d.bbox) for d in results]
        scores = np.array([d.confidence for d in results], dtype=np.float32)

        # NMSBoxes expects (x, y, w, h) for bboxes
        indices = cv2.dnn.NMSBoxes(
            bboxes    = bboxes_list,
            scores    = scores.tolist(),
            score_threshold = 0.0, # Keep all detections for NMS, filter by confidence earlier
            nms_threshold   = iou_threshold,
        )

        if len(indices) == 0:
            return [] # If NMS returns nothing, return empty list

        idx = [int(i) for i in np.array(indices).flatten()]
        return [results[i] for i in idx]


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: list[DetectionResult],
    labels: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and optional labels on *frame*.

    Parameters
    ----------
    frame      : BGR image.
    detections : List returned by :meth:`Detector.detect`.
    labels     : Optional list of strings (one per detection) to overlay.

    Returns
    -------
    Annotated BGR image (same array, modified in-place).
    """
    for i, det in enumerate(detections):
        x, y, w, h = det.bbox
        colour = COLOUR_KNOWN if (labels and i < len(labels)) else COLOUR_UNKNOWN

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

        # Backend tag
        tag = det.backend.upper()
        cv2.putText(frame, tag, (x, y - 6), FONT, 0.42, colour, 1, cv2.LINE_AA)

        # Optional recognition label
        if labels and i < len(labels):
            cv2.putText(
                frame, labels[i],
                (x, y + h + 16), FONT, 0.50, colour, 1, cv2.LINE_AA,
            )

    return frame


# ── Convenience functions ─────────────────────────────────────────────────────

def detect_from_image(
    path: str,
    backend: str = "auto",
    display: bool = True,
    recognition_fn=None,
) -> list[DetectionResult]:
    """
    Run detection on a single image file.

    Parameters
    ----------
    path : str
        Path to the input image.
    backend : str
        'auto', 'dnn', or 'yolo'.
    display : bool
        Show the annotated image in an OpenCV window.
    recognition_fn : callable, optional
        ``fn(face_roi) → label_str``  If provided, called for each detection
        and the result is drawn on the frame.

    Returns
    -------
    list[DetectionResult]
    """
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    detector = Detector(backend=backend)
    detections = detector.detect(frame)

    labels = None
    if recognition_fn is not None:
        labels = [recognition_fn(det.face_roi) for det in detections]

    annotated = draw_detections(frame.copy(), detections, labels)

    logger.info("Detected %d region(s) in '%s'.", len(detections), path)
    for i, det in enumerate(detections):
        lbl = labels[i] if labels else "-"
        logger.info("  [%d] %r → identity=%s", i, det, lbl)

    if display:
        cv2.imshow("detection.py – " + os.path.basename(path), annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def run_live(
    source=CAMERA_SOURCE,
    backend: str = "auto",
    recognition_fn=None,
) -> None:
    """
    Run real-time detection on a webcam or video file.

    Parameters
    ----------
    source : int | str
        Camera index or path to a video file.
    backend : str
        'auto', 'dnn', or 'yolo'.
    recognition_fn : callable, optional
        ``fn(face_roi) → label_str``  Called for each detection on every frame.
    """
    detector = Detector(backend=backend)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    logger.info("Live detection started. Press 'q' to quit.")
    fps_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Stream ended.")
            break

        frame_idx += 1
        detections = detector.detect(frame)

        labels = None
        if recognition_fn is not None:
            labels = [recognition_fn(det.face_roi) for det in detections]

        draw_detections(frame, detections, labels)

        # FPS counter
        elapsed = time.time() - fps_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame, f"FPS: {fps:.1f}  Faces: {len(detections)}",
            (8, frame.shape[0] - 10), FONT, 0.50, (180, 220, 180), 1, cv2.LINE_AA,
        )

        cv2.imshow("detection.py – Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face / upper-body detection for CCTV frames."
    )
    parser.add_argument(
        "--source", default=None,
        help="Image path, video path, or camera index (default: webcam 0).",
    )
    parser.add_argument(
        "--backend", choices=["auto", "dnn", "yolo"], default="auto",
        help="Detection backend (default: auto).",
    )
    parser.add_argument(
        "--with-recognition", action="store_true",
        help="Load PersonIdentifier and annotate with identity labels.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Suppress OpenCV windows (useful in headless environments).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Optionally attach the recognition pipeline
    recog_fn = None
    if args.with_recognition:
        try:
            from database import PersonDatabase
            from person_identifier import PersonIdentifier

            _db   = PersonDatabase()
            _ident = PersonIdentifier(db=_db)
            if _ident.is_ready():
                recog_fn = lambda roi: str(_ident.identify(roi))
            else:
                logger.warning("Recognizer not ready — run model_trainer.py first.")
        except Exception as exc:
            logger.warning("Could not load recognizer: %s", exc)

    source = args.source

    # ── Image mode ────────────────────────────────────────────────────────────
    if source and os.path.isfile(source) and not source.endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    ):
        detect_from_image(
            source,
            backend=args.backend,
            display=not args.no_display,
            recognition_fn=recog_fn,
        )

    # ── Video / Camera mode ───────────────────────────────────────────────────
    else:
        cam = int(source) if (source and source.isdigit()) else (source or CAMERA_SOURCE)
        run_live(
            source=cam,
            backend=args.backend,
            recognition_fn=recog_fn,
        )
