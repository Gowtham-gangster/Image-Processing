"""
recognition.py
==============
End-to-end person recognition pipeline for the
Mask-Aware Hybrid Person Identification System.

Pipeline per input image
------------------------
1. Detect face / upper-body regions  (detection.py → Detector)
2. Extract MobileNetV2 embedding     (embeddings.py → EmbeddingExtractor)
3. SVM / KNN classification          (train_model.py model → pipeline.predict)
4. Cosine-similarity check           against the embedding store
5. Threshold gate                    below threshold → "Unknown"
6. DB attribute lookup               (database.py → PersonDatabase)
7. Return RecognitionResult          (dataclass with all metadata)

Two complementary confidence signals are combined:
  • ``clf_confidence``   : max class-probability from the trained classifier.
  • ``embed_similarity`` : cosine similarity to the nearest training embedding.
Both must exceed their respective thresholds for a "known" result.

Usage (module)
--------------
    from recognition import Recognizer

    rec = Recognizer()

    # From a BGR numpy array already in memory
    results = rec.recognize(frame)

    # From a file path
    results = rec.recognize_file("path/to/face.jpg")

    for r in results:
        print(r)           # RecognitionResult __str__

Usage (CLI)
-----------
    python recognition.py --input photo.jpg
    python recognition.py --input cctv.mp4
    python recognition.py --input 0                  # webcam
    python recognition.py --input photo.jpg --clf-thresh 0.7 --sim-thresh 0.65
    python recognition.py --input photo.jpg --no-display
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import (
    MODELS_DIR, LOG_LEVEL, CAMERA_SOURCE,
    COLOUR_KNOWN, COLOUR_UNKNOWN, COLOUR_MASKED, FONT,
)
from database import PersonDatabase
from detection import Detector, DetectionResult, draw_detections
from embeddings import EmbeddingExtractor
from train_model import load_model

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CLF_THRESHOLD = 0.55    # min classifier confidence  (0–1)
DEFAULT_SIM_THRESHOLD = 0.50    # min cosine similarity       (0–1)
UNKNOWN_LABEL         = "Unknown"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RecognitionResult:
    """
    Full output of one face recognition attempt.

    Fields
    ------
    person_id        : folder name / CSV key, or None.
    name             : attribute from persons.csv, or "Unknown".
    is_known         : True when both confidence gates pass.
    clf_confidence   : classifier max-class probability  [0, 1].
    embed_similarity : cosine similarity to nearest stored embedding [0, 1].
    bbox             : (x, y, w, h) detection box in the source frame.
    attributes       : full persons.csv row dict (empty if unknown).
    detection        : raw DetectionResult from the detector.
    """
    person_id:        Optional[str]  = None
    name:             str            = UNKNOWN_LABEL
    is_known:         bool           = False
    clf_confidence:   float          = 0.0
    embed_similarity: float          = 0.0
    bbox:             tuple          = (0, 0, 0, 0)
    attributes:       dict           = field(default_factory=dict)
    detection:        Optional[DetectionResult] = None

    def __str__(self) -> str:
        status = "KNOWN" if self.is_known else "UNKNOWN"
        return (
            f"[{status}] {self.name}"
            + (f" | ID={self.person_id}" if self.person_id else "")
            + f" | clf={self.clf_confidence:.2%}"
            + f" | sim={self.embed_similarity:.2%}"
        )

    def to_dict(self) -> dict:
        base = {
            "person_id":        self.person_id,
            "name":             self.name,
            "is_known":         self.is_known,
            "clf_confidence":   round(self.clf_confidence, 4),
            "embed_similarity": round(self.embed_similarity, 4),
            "bbox":             self.bbox,
        }
        base.update(self.attributes)
        return base


# ── Recognizer ────────────────────────────────────────────────────────────────

class Recognizer:
    """
    Combines detection + embedding extraction + classification + DB lookup
    into a single, callable object.

    Parameters
    ----------
    model_path : str
        Path to the joblib model file produced by train_model.py.
    clf_threshold : float
        Minimum classifier confidence.  Below → Unknown.
    sim_threshold : float
        Minimum cosine similarity to stored embeddings.  Below → Unknown.
    dl_backend : str
        Embedding extractor backend: 'auto', 'tf', or 'torch'.
    detect_backend : str
        Face detector backend: 'auto', 'dnn', or 'haar'.
    detect_upper_body : bool
        Pass True to also run upper-body detection as a fallback.
    """

    def __init__(
        self,
        model_path: str = os.path.join(MODELS_DIR, "person_identifier.pkl"),
        clf_threshold: float = DEFAULT_CLF_THRESHOLD,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        dl_backend: str = "auto",
        detect_backend: str = "auto",
        detect_upper_body: bool = False,
    ) -> None:
        self.clf_threshold = clf_threshold
        self.sim_threshold = sim_threshold

        logger.info("Initialising Recognizer …")

        # ── Classifier ────────────────────────────────────────────────────────
        payload           = load_model(model_path)
        self._pipeline    = payload["pipeline"]
        self._label_map   = payload["label_map"]    # {int → person_id}
        self._clf_name    = payload["classifier"]
        logger.info("Classifier loaded: %s | %d identities.",
                    self._clf_name.upper(), len(self._label_map))

        # ── Embedding extractor ───────────────────────────────────────────────
        self._extractor = EmbeddingExtractor(backend=dl_backend)

        # ── Embedding store (for similarity check) ────────────────────────────
        try:
            self._store = EmbeddingExtractor.load_store()
            logger.info("Embedding store loaded: %d samples.",
                        self._store["embeddings"].shape[0])
        except FileNotFoundError:
            logger.warning(
                "Embedding store not found — similarity check disabled. "
                "Run  python train_model.py  to build it."
            )
            self._store = None

        # ── Detector ─────────────────────────────────────────────────────────
        self._detector = Detector(
            backend=detect_backend
        )

        # ── Database ──────────────────────────────────────────────────────────
        self._db = PersonDatabase()

        logger.info("Recognizer ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def recognize(self, frame: np.ndarray) -> list[RecognitionResult]:
        """
        Run the full pipeline on a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3).

        Returns
        -------
        list[RecognitionResult]  – one entry per detected face, sorted by
        descending classifier confidence.
        """
        if frame is None or frame.size == 0:
            return []

        detections = self._detector.detect(frame)
        if not detections:
            return []

        results = []
        for det in detections:
            result = self._recognize_roi(det)
            results.append(result)

        results.sort(key=lambda r: r.clf_confidence, reverse=True)
        return results

    def recognize_file(
        self,
        path: str,
        display: bool = False,
    ) -> list[RecognitionResult]:
        """
        Load an image from *path* and run recognition.

        Parameters
        ----------
        display : bool
            If True, show an annotated OpenCV window.

        Returns
        -------
        list[RecognitionResult]
        """
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: '{path}'")

        results = self.recognize(frame)

        if display:
            self._annotate_and_show(frame, results, window_name=os.path.basename(path))

        return results

    def recognize_frame(
        self,
        frame: np.ndarray,
        annotate: bool = True,
    ) -> tuple[list[RecognitionResult], np.ndarray]:
        """
        Recognize faces in *frame* and return both results and an annotated copy.

        Parameters
        ----------
        annotate : bool
            Draw boxes + labels on a copy of frame.

        Returns
        -------
        (results, annotated_frame)
        """
        results = self.recognize(frame)
        vis = self._draw(frame.copy(), results) if annotate else frame.copy()
        return results, vis

    # ── Private — core logic ──────────────────────────────────────────────────

    def _recognize_roi(self, det: DetectionResult) -> RecognitionResult:
        """
        Run embedding → classify → threshold → DB lookup for one detection.
        """
        face_roi = det.face_roi
        result = RecognitionResult(bbox=det.bbox, detection=det)

        if face_roi is None or face_roi.size == 0:
            logger.debug("Empty ROI — returning Unknown.")
            return result

        # ── Extract embedding ────────────────────────────────────────────────
        try:
            emb = self._extractor.extract(face_roi)          # (1280,) float32
        except Exception as exc:
            logger.warning("Embedding extraction failed: %s", exc)
            return result

        # ── Classifier prediction ────────────────────────────────────────────
        try:
            proba     = self._pipeline.predict_proba([emb])[0]   # (n_classes,)
            label_int = int(np.argmax(proba))
            clf_conf  = float(proba[label_int])
        except Exception as exc:
            logger.warning("Classifier predict failed: %s", exc)
            return result

        result.clf_confidence = clf_conf

        # ── Cosine similarity check ──────────────────────────────────────────
        sim = self._cosine_sim(emb)
        result.embed_similarity = sim

        # ── Threshold gate ───────────────────────────────────────────────────
        if clf_conf < self.clf_threshold or sim < self.sim_threshold:
            logger.debug(
                "Below threshold — clf=%.3f (min %.2f), sim=%.3f (min %.2f) → Unknown.",
                clf_conf, self.clf_threshold, sim, self.sim_threshold,
            )
            return result   # is_known remains False

        # ── Known person ─────────────────────────────────────────────────────
        person_id = self._label_map.get(label_int)
        if person_id is None:
            logger.warning("label_int %d not in label_map — Unknown.", label_int)
            return result

        attrs = self._db.get_person(person_id) or {}

        result.person_id        = person_id
        result.name             = attrs.get("name", person_id)
        result.is_known         = True
        result.attributes       = attrs

        logger.info(
            "Recognised: %s (clf=%.2f%%, sim=%.2f%%)",
            result.name, clf_conf * 100, sim * 100,
        )
        return result

    def _cosine_sim(self, emb: np.ndarray) -> float:
        """
        Return cosine similarity between *emb* and the nearest stored embedding.
        Returns 1.0 if no store is loaded (disables the similarity gate).
        """
        if self._store is None:
            return 1.0
        hits = EmbeddingExtractor.nearest_neighbour(emb, self._store, top_k=1)
        return hits[0]["similarity"] if hits else 0.0

    # ── Drawing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _draw(frame: np.ndarray, results: list[RecognitionResult]) -> np.ndarray:
        """Annotate *frame* with bounding boxes, names, and confidence scores."""
        for res in results:
            x, y, w, h = res.bbox

            # Colour coding
            colour = COLOUR_KNOWN if res.is_known else COLOUR_UNKNOWN

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

            # Label background
            lines = _build_label(res)
            lh    = 20
            pad   = 4
            py    = max(y - len(lines) * lh - pad, 0)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, py), (x + w, y), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            # Text lines
            for i, line in enumerate(lines):
                cv2.putText(
                    frame, line,
                    (x + 4, py + (i + 1) * lh - 2),
                    FONT, 0.48, colour, 1, cv2.LINE_AA,
                )

            # Attribute panel (known persons only)
            if res.is_known and res.attributes:
                _draw_attr_panel(frame, res, (x, y + h + 6))

        return frame

    def _annotate_and_show(
        self,
        frame: np.ndarray,
        results: list[RecognitionResult],
        window_name: str = "Recognition",
    ) -> None:
        annotated = self._draw(frame.copy(), results)
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _build_label(res: RecognitionResult) -> list[str]:
    if res.is_known:
        return [
            res.name,
            f"ID: {res.person_id}",
            f"Clf: {res.clf_confidence:.0%}  Sim: {res.embed_similarity:.0%}",
        ]
    return [
        UNKNOWN_LABEL,
        f"Clf: {res.clf_confidence:.0%}  Sim: {res.embed_similarity:.0%}",
    ]


def _draw_attr_panel(
    frame: np.ndarray,
    res: RecognitionResult,
    origin: tuple[int, int],
) -> None:
    """Draw a small attribute panel below the bounding box."""
    attrs = res.attributes
    lines = [
        f"Gender : {attrs.get('gender','N/A')}",
        f"Age    : {attrs.get('age','N/A')}",
        f"Phone  : {attrs.get('phone','N/A')}",
        f"Addr   : {attrs.get('address','N/A')}",
    ]
    lh, pw = 18, 260
    px, py = origin
    ph     = len(lines) * lh + 6
    h_img, w_img = frame.shape[:2]

    # Clamp to image bounds
    if py + ph > h_img:
        py = h_img - ph - 2
    if px + pw > w_img:
        px = w_img - pw - 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), COLOUR_KNOWN, 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (px + 6, py + (i + 1) * lh),
                    FONT, 0.40, (210, 210, 210), 1, cv2.LINE_AA)


# ── Live / video helper ───────────────────────────────────────────────────────

def run_live(
    source,
    recognizer: Recognizer,
    window_title: str = "recognition.py",
) -> None:
    """Stream recognition output from webcam / video file."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    logger.info("Stream started — press 'q' to quit.")
    fps_t   = time.time()
    n_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n_frame += 1

        results, vis = recognizer.recognize_frame(frame, annotate=True)

        fps = n_frame / max(time.time() - fps_t, 1e-6)
        cv2.putText(vis, f"FPS {fps:.1f}  Faces {len(results)}",
                    (8, vis.shape[0] - 10), FONT, 0.5, (160, 220, 160), 1, cv2.LINE_AA)

        cv2.imshow(window_title, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask-Aware Person Recognition — full pipeline."
    )
    parser.add_argument(
        "--input", default=None,
        help="Image path, video path, or camera index (default: webcam 0).",
    )
    parser.add_argument(
        "--clf-thresh", type=float, default=DEFAULT_CLF_THRESHOLD,
        help=f"Min classifier confidence (default: {DEFAULT_CLF_THRESHOLD}).",
    )
    parser.add_argument(
        "--sim-thresh", type=float, default=DEFAULT_SIM_THRESHOLD,
        help=f"Min cosine similarity (default: {DEFAULT_SIM_THRESHOLD}).",
    )
    parser.add_argument(
        "--dl-backend", choices=["auto", "tf", "torch"], default="auto",
        help="Embedding extractor backend (default: auto).",
    )
    parser.add_argument(
        "--det-backend", choices=["auto", "dnn", "haar"], default="auto",
        help="Face detector backend (default: auto).",
    )
    parser.add_argument(
        "--upper-body", action="store_true",
        help="Also run upper-body cascade as a detection fallback.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Suppress display window (headless mode).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    try:
        rec = Recognizer(
            clf_threshold=args.clf_thresh,
            sim_threshold=args.sim_thresh,
            dl_backend=args.dl_backend,
            detect_backend=args.det_backend,
            detect_upper_body=args.upper_body,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    src = args.input

    # ── Static image ──────────────────────────────────────────────────────────
    if src and os.path.isfile(src) and not src.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    ):
        results = rec.recognize_file(src, display=not args.no_display)
        print()
        if not results:
            print("  No faces detected in the image.")
        for i, r in enumerate(results, 1):
            print(f"  Face {i}: {r}")
            if r.is_known:
                for k, v in r.attributes.items():
                    print(f"           {k:<12}: {v}")
        print()

    # ── Video / camera ────────────────────────────────────────────────────────
    else:
        src = int(src) if (src and src.isdigit()) else (src or CAMERA_SOURCE)
        if args.no_display:
            logger.warning("--no-display with live stream: frames processed but not shown.")
            cap = cv2.VideoCapture(src)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = rec.recognize(frame)
                for r in results:
                    logger.info("%s", r)
            cap.release()
        else:
            run_live(src, rec)
