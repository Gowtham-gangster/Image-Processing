"""
unknown_detector.py
===================
Wraps similarity matching explicitly for identifying Unknown Persons.

Requirements met
----------------
1. Calculate similarity between input embedding and stored embeddings.
2. If similarity < threshold, label as "Unknown Person".
3. Allow threshold adjustment for testing.

Usage
-----
    from unknown_detector import UnknownDetector
    from embedding_database import EmbeddingDatabase
    import numpy as np

    db = EmbeddingDatabase()
    
    # Init with strict threshold
    detector = UnknownDetector(threshold=0.70)
    
    # Adjust on the fly for testing
    detector.set_threshold(0.65)
    
    # Process a 128-D embedding 
    vec = np.random.rand(128).astype(np.float32)
    label, confidence = detector.identify(vec, db)
    
    if label == "Unknown Person":
        print("Intruder detected!")
"""

from typing import Tuple
import numpy as np
import logging

from config import LOG_LEVEL
from similarity_matcher import SimilarityMatcher
from embedding_database import EmbeddingDatabase

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class UnknownDetector:
    """
    Identifies if a face embedding represents a known person or an 'Unknown Person'.

    Parameters
    ----------
    threshold : float
        The minimum cosine similarity required (0.0 to 1.0) to register a match as 'known'.
        If the best match falls below this, the result is forced to "Unknown Person".
    """

    UNKNOWN_LABEL = "Unknown Person"

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        self._matcher = SimilarityMatcher(threshold=self.threshold)
        logger.info("UnknownDetector initialized with Euclidean threshold=%.2f via SimilarityMatcher", self.threshold)

    def set_threshold(self, new_threshold: float) -> None:
        """
        Adjust the similarity threshold dynamically.

        Parameters
        ----------
        new_threshold : float
            New strictness level between 0.0 and 1.0. 
            Higher means more likely to return 'Unknown Person'.
        """
        if not (0.0 <= new_threshold <= 4.0):
            raise ValueError(f"Euclidean threshold must be between 0.0 and 4.0 (got {new_threshold})")
            
        old = self.threshold
        self.threshold = new_threshold
        self._matcher.threshold = new_threshold
        logger.info("UnknownDetector threshold adjusted: %.2f -> %.2f", old, self.threshold)

    def identify(self, embedding: np.ndarray, db: EmbeddingDatabase) -> Tuple[str, float]:
        """
        Calculates similarity against the database and applies the unknown gate.

        Parameters
        ----------
        embedding : np.ndarray
            A 128-D L2-normalized numpy array representing the face.
        db : EmbeddingDatabase
            The loaded database of known individuals.

        Returns
        -------
        label : str
            The identified person_id, or "Unknown Person".
        score : float
            The highest cosine similarity score found (0.0 to 1.0).
        """
        # Explicit Recognition Step via the new SimilarityMatcher Euclidean pipeline
        best_id, distance = self._matcher.match(embedding, db)

        # UI confidence format is still required for the dashboard.
        # So we project the raw L2 distance into a 0.0 -> 1.0 score mapping
        sim = max(0.0, 1.0 - ((distance**2) / 2.0))
        
        # 1. DB Empty or Matcher returned Exception
        if best_id == "Unknown" and distance == 999.0:
            logger.debug("Database is empty or match failed. Returning '%s'.", self.UNKNOWN_LABEL)
            return self.UNKNOWN_LABEL, 0.0
            
        # 2. Matcher logic yielded Unknown (distance >= threshold)
        if best_id == "Unknown":
            # The matcher already verified distance >= self.threshold
            return self.UNKNOWN_LABEL, float(sim)
            
        # 3. Valid Known Person match
        return best_id, float(sim)

if __name__ == "__main__":
    # Quick Test
    db = EmbeddingDatabase("test_unknown_db.npz")
    db_vec = np.zeros(128, dtype=np.float32)
    db_vec[0] = 1.0
    db.add_embedding("Alice", db_vec)

    detector = UnknownDetector(threshold=0.80)
    
    # 0.99 cosine similarity (Should be Alice)
    q1 = np.zeros(128, dtype=np.float32)
    q1[0] = 0.99
    q1[1] = 0.141
    print(f"Test 1 (Sim ~0.99): {detector.identify(q1, db)}")
    
    # 0.50 cosine similarity (Should be Unknown Person)
    q2 = np.zeros(128, dtype=np.float32)
    q2[0] = 0.50
    q2[1] = 0.866
    print(f"Test 2 (Sim ~0.50): {detector.identify(q2, db)}")
    
    # Adjust threshold down
    detector.set_threshold(0.40)
    print("Lowered threshold to 0.40")
    print(f"Test 3 (Sim ~0.50): {detector.identify(q2, db)}")
    
    import os
    if os.path.exists("test_unknown_db.npz"):
        os.remove("test_unknown_db.npz")
