"""
adaptive_identifier.py
======================
Implements adaptive recognition switching between facial features and body features.

Logic:
    if face detected and not masked:
        use face recognition only (128-D)
    elif face detected but masked:
        combine face and body features (Late fusion / Score averaging)
    else:
        use body recognition only (2048-D)

Usage
-----
    from adaptive_identifier import AdaptiveIdentifier
    
    identifier = AdaptiveIdentifier()
    best_id, confidence, strategy = identifier.identify(
        face_img=face_crop, 
        is_masked=True, 
        body_img=upper_body_crop
    )
"""

import logging
import numpy as np
from typing import Tuple, Optional

from config import LOG_LEVEL
from embedding_model import FaceNetEmbedder
from embedding_database import EmbeddingDatabase
from body_feature_extractor import BodyFeatureExtractor
from body_embedding_database import BodyEmbeddingDatabase
from similarity_matcher import SimilarityMatcher
from unknown_detector import UnknownDetector

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class AdaptiveIdentifier:
    """
    Coordinates multi-modal recognition between face vectors and body vectors.
    Dynamically weights the importance of each modality depending on mask occlusion.
    """

    def __init__(self, face_threshold: float = 0.65, body_threshold: float = 0.65) -> None:
        logger.info("Initializing AdaptiveIdentifier...")
        
        # 1. Face Subsystems
        self.face_embedder = FaceNetEmbedder()
        self.face_db = EmbeddingDatabase()
        self.face_matcher = SimilarityMatcher(threshold=0.0) # threshold handled separately
        
        # 2. Body Subsystems
        # Wrap the import in try/except in case TF is not installed
        try:
            self.body_embedder = BodyFeatureExtractor()
        except ImportError:
            logger.error("TensorFlow required for body features. Adaptive switching degraded.")
            self.body_embedder = None
            
        self.body_db = BodyEmbeddingDatabase()
        self.body_matcher = SimilarityMatcher(threshold=0.0)
        
        # 3. Thresholds
        self.face_threshold = face_threshold
        self.body_threshold = body_threshold

    def identify(self, 
                 face_img: Optional[np.ndarray], 
                 is_masked: bool, 
                 body_img: Optional[np.ndarray]) -> Tuple[str, float, str]:
        """
        Executes the adaptive recognition pipeline.

        Parameters
        ----------
        face_img : np.ndarray or None
            A cropped face image. None if no face detected.
        is_masked : bool
            True if the face is classified as wearing a mask.
        body_img : np.ndarray or None
            A cropped upper body image. None if no body detected.

        Returns
        -------
        best_id : str
            The identified person, or "Unknown Person".
        confidence : float
            The combined confidence score (0.0 to 1.0).
        strategy : str
            Logging string indicating which pathway was used ("Face-Only", "Combined", "Body-Only", "Failed").
        """
        has_face = face_img is not None and face_img.size > 0
        has_body = body_img is not None and body_img.size > 0 and self.body_embedder is not None
        
        # ---------- LOGIC PATHWAY 1: Face detected and NOT masked ----------
        if has_face and not is_masked:
            # High confidence full-face biometrics
            face_vec = self.face_embedder.extract(face_img)
            pid, score = self.face_matcher.match(face_vec, self.face_db)
            
            if score >= self.face_threshold and pid != "Unknown":
                return pid, score, "Face-Only"
            return UnknownDetector.UNKNOWN_LABEL, score, "Face-Only"

        # ---------- LOGIC PATHWAY 2: Face detected BUT masked ----------
        elif has_face and is_masked and has_body:
            # We have both partial face and body evidence.
            # Extract both embeddings.
            face_vec = self.face_embedder.extract(face_img)
            body_vec = self.body_embedder.extract(body_img) # type: ignore
            
            # Predict separately against both DBs
            f_pid, f_score = self.face_matcher.match(face_vec, self.face_db)
            b_pid, b_score = self.body_matcher.match(body_vec, self.body_db)
            
            # Late Fusion strategy (score averaging if they agree)
            if f_pid == b_pid and f_pid != "Unknown":
                combined_score = (f_score * 0.4) + (b_score * 0.6) # Trust body sightly more if masked
                
                # We apply a slightly lower, forgiving threshold since we have dual verification
                dual_threshold = min(self.face_threshold, self.body_threshold) - 0.05
                
                if combined_score >= dual_threshold:
                    return f_pid, combined_score, "Combined (Agreement)"
                    
            # If they disagree, prefer the body match purely because the face is obfuscated.
            if b_pid != "Unknown" and b_score >= self.body_threshold:
                return b_pid, b_score, "Combined (Body Preferred)"
                
            return UnknownDetector.UNKNOWN_LABEL, max(f_score, b_score), "Combined (Failed)"

        # ---------- LOGIC PATHWAY 3: Face masked but NO body OR Face Missing ----------
        elif has_body:
            # e.g., Face is entirely turned away from camera, but we see the clothing
            body_vec = self.body_embedder.extract(body_img) # type: ignore
            pid, score = self.body_matcher.match(body_vec, self.body_db)
            
            if score >= self.body_threshold and pid != "Unknown":
                return pid, score, "Body-Only"
            return UnknownDetector.UNKNOWN_LABEL, score, "Body-Only"
            
        # ---------- LOGIC PATHWAY 4: Nothing valid ---------
        elif has_face: # Face exists but we rejected it as masked without a body fallback
            face_vec = self.face_embedder.extract(face_img)
            pid, score = self.face_matcher.match(face_vec, self.face_db)
            if score >= self.face_threshold and pid != "Unknown":
                return pid, score, "Face-Only (Masked Penalty)"
            return UnknownDetector.UNKNOWN_LABEL, score, "Face-Only (Masked Penalty)"

        return UnknownDetector.UNKNOWN_LABEL, 0.0, "Failed (No Data)"

if __name__ == "__main__":
    # Smoke test initialization
    try:
        adaptive = AdaptiveIdentifier()
        print("Adaptive Identifier initialized successfully.")
    except Exception as e:
        print(f"Init Error: {e}")
