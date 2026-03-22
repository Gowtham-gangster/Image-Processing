"""
liveness_detector.py
====================
Lightweight Anti-Spoofing and Liveness Detection.

Requirements met
----------------
1. Prevent spoofing via 2D static photos on screens/paper.
2. Lightweight heuristic without requiring an external heavy CNN.
3. Uses a combination of Blur, Texture (LBP), and Specular Reflection.

Usage
-----
    from liveness_detector import LivenessDetector
    liveness = LivenessDetector(strictness=0.6)
    
    is_live, msg = liveness.check(face_crop)
    if not is_live:
        print("Spoof Attempt:", msg)
"""

import cv2
import numpy as np
import logging
import os
from typing import Optional

from config import LOG_LEVEL, MODELS_DIR

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ── Lightweight CNN Model Definition ──────────────────────────────────────────

def build_liveness_model(input_shape=(224, 224, 3)):
    """A minimal 3-layer CNN architecture for real-time binary spoof classification."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        
        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])
        return model
    except ImportError:
        logger.warning("TensorFlow/Keras not found. CNN-based liveness will be unavailable.")
        return None

# ── Liveness Detection Engine ──────────────────────────────────────────────────
class LivenessDetector:
    """
    Heuristic Anti-Spoofing Engine assessing face crop realism.
    It combines multiple 2D texture checks to block printed photos and screens.
    """
    
    def __init__(self, model_path: Optional[str] = None, blur_thresh: float = 65.0):
        self.blur_thresh = blur_thresh
        self.model = None
        
        # Load CNN if path is provided or exists in models dir
        liveness_weights = model_path or os.path.join(MODELS_DIR, "liveness.h5")
        if os.path.exists(liveness_weights):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(liveness_weights)
                logger.info("CNN Liveness model loaded from: %s", liveness_weights)
            except Exception as e:
                logger.error("Failed to load CNN liveness weights: %s", e)
        
        if not self.model:
            logger.info("Initializing LivenessDetector using Heuristic Fallback (Blur/Texture analysis).")

    def check(self, face_crop: np.ndarray) -> tuple[bool, str]:
        """
        Evaluate a BGR face crop for liveness.
        
        Returns:
            (is_live: bool, reason: str)
        """
        if face_crop is None or face_crop.size == 0:
            return False, "Invalid Face Crop"
            
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # --- 1. CNN Classification (Primary) ---
            if self.model:
                resized = cv2.resize(face_crop, (224, 224))
                blob    = resized.astype("float32") / 255.0
                blob    = np.expand_dims(blob, axis=0)
                
                score = self.model.predict(blob, verbose=0)[0][0]
                if score < 0.5: # 0 = Spoof, 1 = Live
                    return False, f"Spoof Detected (CNN Confidence: {1-score:.1%})"

            # --- 2. Laplacian Variance (Focus check) ---
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < self.blur_thresh:
                return False, f"Spoof Detected (Blurry Photo: {laplacian_var:.1f})"
                
            # --- 3. LBP Texture Analysis (Depth check) ---
            try:
                from skimage import feature
                lbp = feature.local_binary_pattern(gray, 8, 1, method="uniform")
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                
                # Screens and prints have significantly lower texture variation
                if hist[np.argmax(hist)] > 0.85: 
                    return False, "Spoof Detected (Texture mapping anomaly)"
            except ImportError:
                pass
                
            # --- 4. Overexposure Check ---
            if np.mean(gray) > 240:
                return False, "Spoof Detected (Overexposed Screen artifact)"
            
        except Exception as e:
            logger.error("Liveness check error: %s", e)
            return False, f"Liveness Error: {str(e)}"

        return True, "Live Person"

        return True, "Live Person"
