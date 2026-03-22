"""
feature_extractor.py
====================
Extracts identity-preserving features from face images.

Hybrid strategy
---------------
- **Unmasked faces** → LBPH features (texture) + HOG features (shape).
- **Masked faces**  → HOG on upper face region (eyes/forehead) only,
                      or deep embedding if a backbone model is available.

LBPH recognition is handled by cv2.face.LBPHFaceRecognizer (see model_trainer.py).
This module provides the additional HOG and upper-face utilities used for
mask-robust identification.
"""

import logging

import cv2
import numpy as np
from skimage.feature import hog  # type: ignore

from config import FACE_SIZE

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts discriminative feature vectors from face ROIs.

    Methods
    -------
    extract(face_roi, masked)
        Return a 1-D float32 feature vector.
    extract_upper(face_roi)
        HOG features from the upper half (eyes / forehead / bridge of nose).
    """

    def __init__(self, face_size: tuple[int, int] = FACE_SIZE):
        self.face_size = face_size

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, face_roi: np.ndarray, masked: bool = False) -> np.ndarray:
        """
        Extract a feature vector appropriate for the mask state.

        Parameters
        ----------
        face_roi : np.ndarray
            BGR (or gray) image of the face region.
        masked : bool
            True if the face is wearing a mask.

        Returns
        -------
        np.ndarray  shape (N,)  dtype float32
        """
        gray = self._to_gray(face_roi)
        resized = cv2.resize(gray, self.face_size)

        if masked:
            # Only use the upper portion of the face (above the nose)
            return self.extract_upper(resized)
        else:
            # Full-face: concatenate HOG + LBP histogram
            hog_vec = self._hog_features(resized)
            lbp_vec = self._lbp_histogram(resized)
            return np.concatenate([hog_vec, lbp_vec]).astype(np.float32)

    def extract_upper(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from the upper 60 % of the face.
        Robust to mask occlusion.
        """
        h = gray_face.shape[0]
        upper = gray_face[:int(h * 0.60), :]
        upper_resized = cv2.resize(upper, self.face_size)
        return self._hog_features(upper_resized).astype(np.float32)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _hog_features(gray: np.ndarray) -> np.ndarray:
        """
        Compute HOG descriptor.
        Returns a 1-D float64 array.
        """
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return features

    @staticmethod
    def _lbp_histogram(gray: np.ndarray, radius: int = 1, n_points: int = 8,
                       n_bins: int = 256) -> np.ndarray:
        """
        Compute a uniform LBP histogram.

        Uses a pure-NumPy implementation to avoid sklearn dependency version
        conflicts; falls back to skimage if available.
        """
        try:
            from skimage.feature import local_binary_pattern  # type: ignore
            lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-7
            return hist
        except ImportError:
            # Minimal fallback: return flattened pixel values as feature
            logger.warning("skimage not available; using raw pixel histogram.")
            hist, _ = np.histogram(gray.ravel(), bins=n_bins, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-7
            return hist
