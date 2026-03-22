"""
onnx_inference.py
=================
Drop-in ONNX Runtime wrappers for all three DNN inference models.

Each class matches the public API of its original counterpart so existing
callers require zero changes — just swap the instance when INFERENCE_BACKEND
is set to "onnx".

Classes
-------
  OnnxFaceNetEmbedder   → replaces FaceNetEmbedder  (embedding_model.py)
  OnnxMaskDetector      → replaces MaskDetector      (mask_detector.py)
  OnnxFaceDetector      → replaces FaceDetector      (face_detector.py)

Provider auto-selection order
------------------------------
  1. TensorrtExecutionProvider  (Jetson / CUDA host with TRT installed)
  2. CUDAExecutionProvider      (CUDA GPU)
  3. CPUExecutionProvider       (fallback)

Usage
-----
    from onnx_inference import OnnxFaceNetEmbedder, OnnxMaskDetector, OnnxFaceDetector
    import numpy as np

    emb  = OnnxFaceNetEmbedder()
    vec  = emb.extract(face_crop)          # (128,) float32

    mask = OnnxMaskDetector()
    is_m, conf = mask.is_masked(face_roi)  # (bool, float)

    det  = OnnxFaceDetector()
    boxes = det.detect_faces(frame)        # list of (x,y,w,h)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

from config import (
    ONNX_EMBEDDER_PATH, ONNX_MASK_PATH, ONNX_DETECTOR_PATH,
    FACE_CONF_THRESHOLD, MASK_CONF_THRESHOLD, LOG_LEVEL,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


# ── Provider selection ────────────────────────────────────────────────────────

def _get_providers() -> list[str]:
    """Return ORT providers in priority order based on available hardware."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    priority  = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    chosen = [p for p in priority if p in available]
    logger.info("ONNX Runtime providers: %s", chosen)
    return chosen


def _load_session(model_path: str):
    """Load an InferenceSession with best available hardware providers."""
    try:
        import onnxruntime as ort
        providers = _get_providers()
        
        # Prioritize quantized integer paths for extreme memory/bandwidth optimizations
        quant_path = model_path.replace(".onnx", "_quant.onnx")
        if os.path.exists(quant_path):
            logger.info("Hardware optimization: Substituting quantized model -> %s", os.path.basename(quant_path))
            model_path = quant_path
            
        sess = ort.InferenceSession(model_path, providers=providers)
        logger.info("Loaded ONNX InferenceSession: %s", os.path.basename(model_path))
        return sess
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ONNX model at '{model_path}': {e}\n"
            "Run  python export_onnx.py --all  to generate the ONNX files first."
        ) from e


# ── OnnxFaceNetEmbedder ───────────────────────────────────────────────────────

class OnnxFaceNetEmbedder:
    """
    ONNX Runtime version of FaceNetEmbedder.
    Produces 128-D L2-normalised embeddings from 96×96 BGR face crops.
    """

    INPUT_SIZE = (96, 96)

    def __init__(self, model_path: str = ONNX_EMBEDDER_PATH):
        self._session = _load_session(model_path)
        self._input_name = self._session.get_inputs()[0].name

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a BGR face crop to a 128-D L2-normalised embedding.

        Parameters
        ----------
        face_bgr : np.ndarray   BGR image of any size.

        Returns
        -------
        np.ndarray of shape (128,), dtype float32.
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty image passed to OnnxFaceNetEmbedder.")

        face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized   = cv2.resize(face_rgb, self.INPUT_SIZE)
        blob      = resized.astype(np.float32) / 255.0           # (96,96,3)
        blob      = np.transpose(blob, (2, 0, 1))[np.newaxis]    # (1,3,96,96)

        output    = self._session.run(None, {self._input_name: blob})
        embedding = output[0].flatten()                          # (128,)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)


# ── OnnxMaskDetector ──────────────────────────────────────────────────────────

class OnnxMaskDetector:
    """
    ONNX Runtime version of MaskDetector.
    Classifies a face ROI as masked (True) or unmasked (False).
    """

    INPUT_SIZE = (224, 224)

    def __init__(
        self,
        model_path: str = ONNX_MASK_PATH,
        conf_threshold: float = MASK_CONF_THRESHOLD,
    ):
        self._session       = _load_session(model_path)
        self._input_name    = self._session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        # Sniff output shape to handle both sigmoid (1,) and softmax (1,2) outputs
        out_shape = self._session.get_outputs()[0].shape
        self._n_classes = out_shape[-1] if len(out_shape) > 1 else 1

    def is_masked(self, face_roi: np.ndarray) -> tuple[bool, float]:
        """
        Predict whether a face is masked.

        Returns
        -------
        (is_masked, confidence)
        """
        if face_roi is None or face_roi.size == 0:
            return False, 0.0

        img    = cv2.resize(face_roi, self.INPUT_SIZE)
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob   = img[np.newaxis]                                 # (1, 224, 224, 3) NHWC

        preds  = self._session.run(None, {self._input_name: blob})[0][0]

        if self._n_classes >= 2:
            mask_conf = float(preds[1])
        else:
            mask_conf = float(preds[0])

        return mask_conf >= self.conf_threshold, mask_conf

    @property
    def mode(self) -> str:
        return "onnx"


# ── OnnxFaceDetector ──────────────────────────────────────────────────────────

class OnnxFaceDetector:
    """
    ONNX Runtime version of the SSD-ResNet10 face detector.
    Produces (x, y, w, h) bounding boxes from a full BGR frame.
    """

    INPUT_SIZE = (300, 300)

    def __init__(
        self,
        model_path: str = ONNX_DETECTOR_PATH,
        conf_threshold: float = FACE_CONF_THRESHOLD,
    ):
        self._session       = _load_session(model_path)
        self._input_name    = self._session.get_inputs()[0].name
        self.conf_threshold = conf_threshold

    def detect_faces(
        self, frame: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        Detect faces in a BGR frame.

        Returns
        -------
        List of (x, y, w, h) pixel coordinates.
        """
        if frame is None or frame.size == 0:
            return []

        h_img, w_img = frame.shape[:2]

        # Preprocess: BGR(H,W,3) → float32 NCHW (1,3,300,300), mean-subtracted
        resized = cv2.resize(frame, self.INPUT_SIZE)
        mean    = np.array([104.0, 177.0, 123.0], dtype=np.float32)
        blob    = (resized.astype(np.float32) - mean)
        blob    = np.transpose(blob, (2, 0, 1))[np.newaxis]     # (1,3,300,300)

        detections = self._session.run(None, {self._input_name: blob})[0]
        # SSD output: (1, 1, N, 7)  columns: [img_id, class, conf, x1,y1,x2,y2]
        detections = detections.reshape(-1, 7)

        boxes = []
        for det in detections:
            conf = float(det[2])
            if conf < self.conf_threshold:
                continue
            x1 = int(det[3] * w_img)
            y1 = int(det[4] * h_img)
            x2 = int(det[5] * w_img)
            y2 = int(det[6] * h_img)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            w, h   = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                boxes.append((x1, y1, w, h))

        return boxes

    @property
    def mode(self) -> str:
        return "onnx"


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "embedder"

    if model == "embedder":
        emb  = OnnxFaceNetEmbedder()
        vec  = emb.extract(np.zeros((200, 200, 3), dtype=np.uint8))
        print(f"Embedder → shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")

    elif model == "mask":
        det  = OnnxMaskDetector()
        is_m, conf = det.is_masked(np.zeros((224, 224, 3), dtype=np.uint8))
        print(f"MaskDetector → is_masked={is_m}, confidence={conf:.3f}")

    elif model == "detector":
        fdet = OnnxFaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = fdet.detect_faces(frame)
        print(f"FaceDetector → {len(boxes)} boxes detected")
