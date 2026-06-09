"""
embedding_model.py
==================
Feature extraction using a pretrained CNN to generate 128-dimensional embeddings.

This module downloads and wraps OpenCV's OpenFace model (a FaceNet derivative),
which natively produces highly discriminant 128-D unit vectors (embeddings)
from 96x96 face crops.

Usage
-----
    from embedding_model import FaceNetEmbedder
    import cv2

    embedder = FaceNetEmbedder()
    face_image = cv2.imread("face.jpg")

    # Outputs a (128,) numpy float32 array
    vector = embedder.extract(face_image)
"""

import os
import urllib.request
import logging
import cv2
import numpy as np

from config import MODELS_DIR, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# The widely used OpenFace PyTorch/Torch model (128-D embeddings based on FaceNet paper)
OPENFACE_URL = "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7"
MODEL_PATH = os.path.join(MODELS_DIR, "openface_nn4.small2.v1.t7")


class FaceNetEmbedder:
    """
    128-Dimensional Embedding generator using OpenFace (FaceNet derivative).
    Downloads the required .t7 model automatically if missing.
    """

    def __init__(self) -> None:
        self._ensure_model_exists()
        logger.info("Loading OpenFace 128-D embedding model via OpenCV DNN…")
        self.net = cv2.dnn.readNetFromTorch(MODEL_PATH)

    def _ensure_model_exists(self) -> None:
        """Download the .t7 weights if they are not present."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            logger.info("Downloading OpenFace 128-D model weights (approx 30 MB)…")
            urllib.request.urlretrieve(OPENFACE_URL, MODEL_PATH)
            logger.info("Download complete.")

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a BGR face crop into a 128-D L2-normalized embedding.

        Parameters
        ----------
        face_bgr : np.ndarray
            OpenCV BGR image array.

        Returns
        -------
        embedding : np.ndarray
            A 1D numpy array of shape (128,) containing float32 values.
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty image provided to embedder.")

        # OpenFace expects RGB 96x96 inputs normalized such that values are scaled appropriately
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        # Mean subtraction typically used for this specific model variant is (0,0,0) 
        # but pixel values should be scaled by 1/255.0. 
        blob = cv2.dnn.blobFromImage(
            face_rgb,
            scalefactor=1.0 / 255.0,
            size=(96, 96),
            mean=(0, 0, 0),
            swapRB=False,  # already RGB
            crop=False
        )

        self.net.setInput(blob)
        embedding = self.net.forward()

        # The output is shape (1, 128). We flatten it to (128,).
        embedding = embedding.flatten()

        # L2 Normalize the vector to ensure proper cosine similarity behaviour
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)


# ── Backend factory ───────────────────────────────────────────────────────────

def get_embedder():
    """
    Return the best available embedder based on the INFERENCE_BACKEND config.

    MASKAWARE_BACKEND=original  →  FaceNetEmbedder  (cv2.dnn, default)
    MASKAWARE_BACKEND=onnx      →  OnnxFaceNetEmbedder  (ONNX Runtime)
    MASKAWARE_BACKEND=trt       →  OnnxFaceNetEmbedder  (with TRT provider)

    Usage
    -----
        from embedding_model import get_embedder
        embedder = get_embedder()   # picks the right backend automatically
    """
    from config import INFERENCE_BACKEND
    if INFERENCE_BACKEND in ("onnx", "trt"):
        try:
            from onnx_inference import OnnxFaceNetEmbedder
            logger.info("Embedder backend: ONNX Runtime (%s)", INFERENCE_BACKEND.upper())
            return OnnxFaceNetEmbedder()
        except Exception as exc:
            logger.warning(
                "ONNX embedder unavailable (%s) — falling back to cv2.dnn.", exc
            )
    return FaceNetEmbedder()


if __name__ == "__main__":
    # Quick test
    embedder = get_embedder()
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    vec = embedder.extract(dummy_img)
    print(f"Generated embedding shape: {vec.shape}")
    print(f"Embedding L2 norm: {np.linalg.norm(vec):.4f}")
    print(f"Backend: {type(embedder).__name__}")
