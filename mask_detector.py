"""
mask_detector.py
================
Classifies a face ROI as "masked" or "no_mask".

Two operation modes
-------------------
1. **Model mode** (preferred) – loads a trained Keras/TF model from
   models/mask_detector.model. Supply your own trained model or use one of
   the many public mask-detection models (e.g. from PyImageSearch).

2. **Heuristic mode** (fallback) – a lightweight, OpenCV-only heuristic that
   examines the lower-half of the face for skin-pixel coverage. It is not
   production-accurate but requires no additional model weights.
"""

import os
import logging

import cv2
import numpy as np

from config import MASK_MODEL_PATH, MASK_CONF_THRESHOLD

logger = logging.getLogger(__name__)


class MaskDetector:
    """
    Predicts whether a person is wearing a face mask.

    Parameters
    ----------
    model_path : str
        Path to a Keras SavedModel / .h5 file.
    conf_threshold : float
        Minimum confidence to accept a mask-classification result.
    """

    def __init__(self,
                 model_path: str = MASK_MODEL_PATH,
                 conf_threshold: float = MASK_CONF_THRESHOLD):
        self.conf_threshold = conf_threshold
        self._model = None
        self._mode = "heuristic"

        if os.path.exists(model_path):
            try:
                from tensorflow.keras.models import load_model   # type: ignore
                self._model = load_model(model_path)
                self._mode = "model"
                logger.info("MaskDetector: loaded Keras model from %s.", model_path)
            except Exception as exc:
                logger.warning(
                    "Could not load mask model (%s). Falling back to heuristic.", exc
                )
        else:
            logger.info(
                "Mask model not found at %s. Using heuristic fallback.", model_path
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_masked(self, face_roi: np.ndarray) -> tuple[bool, float]:
        """
        Determine whether the face in *face_roi* is masked.

        Parameters
        ----------
        face_roi : np.ndarray
            BGR image cropped to the face bounding box.

        Returns
        -------
        (is_masked, confidence)
            ``is_masked`` – True if a mask is detected.
            ``confidence`` – float in [0, 1].
        """
        if self._mode == "model":
            return self._predict_model(face_roi)
        return self._predict_heuristic(face_roi)

    # ── Model prediction ──────────────────────────────────────────────────────

    def _predict_model(self, face_roi: np.ndarray) -> tuple[bool, float]:
        """Run the Keras model on a pre-cropped face ROI."""
        img = cv2.resize(face_roi, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)          # (1, 224, 224, 3)

        preds = self._model.predict(img, verbose=0)[0]
        # Assumes two-class output: [no_mask_prob, mask_prob]
        if len(preds) == 2:
            mask_conf = float(preds[1])
        else:
            # Single sigmoid output → probability of 'mask'
            mask_conf = float(preds[0])

        return mask_conf >= self.conf_threshold, mask_conf

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def _predict_heuristic(self, face_roi: np.ndarray) -> tuple[bool, float]:
        """
        Lightweight heuristic:
          - Examine the lower half of the face.
          - If the skin-tone pixel ratio is low (< 0.25), assume mask.

        This is approximate and should be replaced with a proper model.
        """
        if face_roi is None or face_roi.size == 0:
            return False, 0.0

        h, w = face_roi.shape[:2]
        lower_half = face_roi[h // 2:, :]          # bottom half

        # Convert to YCrCb and apply skin-colour thresholds
        ycrcb = cv2.cvtColor(lower_half, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_ratio = np.count_nonzero(skin_mask) / (skin_mask.size + 1e-6)

        masked = skin_ratio < 0.25
        confidence = 1.0 - skin_ratio if masked else skin_ratio
        return masked, float(confidence)

    @property
    def mode(self) -> str:
        """Return 'model' or 'heuristic'."""
        return self._mode


# ── Backend factory ───────────────────────────────────────────────────────────

def get_mask_detector(
    model_path: str = MASK_MODEL_PATH,
    conf_threshold: float = MASK_CONF_THRESHOLD,
):
    """
    Return the best available mask detector based on INFERENCE_BACKEND.

    MASKAWARE_BACKEND=original  →  MaskDetector  (Keras / heuristic, default)
    MASKAWARE_BACKEND=onnx      →  OnnxMaskDetector  (ONNX Runtime)
    MASKAWARE_BACKEND=trt       →  OnnxMaskDetector  (with TRT provider)
    """
    from config import INFERENCE_BACKEND
    if INFERENCE_BACKEND in ("onnx", "trt"):
        try:
            from onnx_inference import OnnxMaskDetector
            logger.info("MaskDetector backend: ONNX Runtime (%s)", INFERENCE_BACKEND.upper())
            return OnnxMaskDetector(conf_threshold=conf_threshold)
        except Exception as exc:
            logger.warning(
                "ONNX mask detector unavailable (%s) — falling back to Keras/heuristic.", exc
            )
    return MaskDetector(model_path=model_path, conf_threshold=conf_threshold)
