"""
fusion_engine.py
================
Calculates a fused identity confidence score combining multiple modalities.

Requirements met
----------------
1. Inputs: face similarity, body similarity, mask detection confidence.
2. Output: fused identity score.
3. Formula used: 0.6*face_score + 0.3*body_score + 0.1*detection_confidence

Usage
-----
    from fusion_engine import FusionEngine
    
    engine = FusionEngine()
    final_score = engine.fuse_scores(
        face_score=0.85,
        body_score=0.70,
        detection_confidence=0.99
    )
"""

import logging

from config import LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class FusionEngine:
    """
    Implements a weighted late-fusion strategy to combine multiple biometric scores.
    """

    def __init__(self, face_weight: float = 0.6, body_weight: float = 0.3, det_weight: float = 0.1) -> None:
        """
        Initialize the Fusion Engine with specific modality weights.
        Default adheres to the requested formula:
            final_score = 0.6*face_score + 0.3*body_score + 0.1*detection_confidence
        """
        self.face_weight = face_weight
        self.body_weight = body_weight
        self.det_weight = det_weight
        
        # Verify valid weighting
        total = self.face_weight + self.body_weight + self.det_weight
        if abs(total - 1.0) > 1e-5:
            logger.warning(
                "FusionEngine weights (%.2f, %.2f, %.2f) sum to %.2f, not 1.0. "
                "The resulting score might exceed 1.0.",
                self.face_weight, self.body_weight, self.det_weight, total
            )

    def fuse_scores(self, face_score: float, body_score: float, detection_confidence: float) -> float:
        """
        Calculates the final fused identity score.

        Parameters
        ----------
        face_score : float
            Cosine similarity score from the face embedding (0.0 to 1.0).
        body_score : float
            Cosine similarity score from the body embedding (0.0 to 1.0).
        detection_confidence : float
            Confidence of the mask/person detector bounding box (0.0 to 1.0).

        Returns
        -------
        final_score : float
            The weighted combination of all three inputs.
        """
        # Constrain inputs technically (optional safety bounds)
        f_s = max(0.0, min(1.0, face_score))
        b_s = max(0.0, min(1.0, body_score))
        d_c = max(0.0, min(1.0, detection_confidence))
        
        final_score = (
            (self.face_weight * f_s) + 
            (self.body_weight * b_s) + 
            (self.det_weight * d_c)
        )
        
        logger.debug(
            "Fusion Calculated -> Face: %.2f * %.1f | Body: %.2f * %.1f | Det: %.2f * %.1f == Final: %.3f",
            f_s, self.face_weight,
            b_s, self.body_weight,
            d_c, self.det_weight,
            final_score
        )
        
        return final_score

    def identify_fused(self, 
                       f_identity: str, f_score: float,
                       b_identity: str, b_score: float,
                       det_conf: float,
                       threshold: float = 0.65) -> tuple[str, float]:
        """
        Higher-level helper that applies the fusion logic to identity predictions.
        
        If Face and Body networks agree on identity, it returns the mathematical fusion.
        If they disagree, it falls back to the strongest individual predictor instead 
        of blending incompatible probabilities.
        """
        # They agree
        if f_identity == b_identity and f_identity != "Unknown Person":
            fused = self.fuse_scores(f_score, b_score, det_conf)
            if fused >= threshold:
                return f_identity, fused
            return "Unknown Person", fused
            
        # They disagree, return highest un-fused confidence to prevent "frankenstein" scores
        if f_score >= b_score and f_score >= threshold:
            return f_identity, f_score
        if b_score > f_score and b_score >= threshold:
            return b_identity, b_score
            
        return "Unknown Person", max(f_score, b_score)


if __name__ == "__main__":
    # Smoke test requirements formula
    engine = FusionEngine()
    
    # Test case 1: Very strong face, moderate body, high UI detection box confidence
    res = engine.fuse_scores(face_score=0.90, body_score=0.60, detection_confidence=0.98)
    
    # Expected: (0.6*0.90) + (0.3*0.60) + (0.1*0.98)
    # Expected: 0.54 + 0.18 + 0.098 = 0.818
    print(f"Calculated Fused Score: {res:.3f} (Expected: ~0.818)")
