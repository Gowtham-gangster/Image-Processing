"""
similarity_matcher.py
=====================
Computes L2 (Euclidean) distance for face recognition.
Ensures embeddings are strictly normalized before matching.
"""

import numpy as np
from typing import Tuple

from embedding_database import EmbeddingDatabase

class SimilarityMatcher:
    """
    Performs Euclidean L2 distance matching between a query vector and a database.

    Parameters
    ----------
    threshold : float
        Maximum L2 distance required to confirm a match. 
        Distance < 0.6 returns the identity, else Unknown.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def match(self, query_vector: np.ndarray, db: EmbeddingDatabase) -> Tuple[str, float]:
        """
        1. Normalize query
        2. Query FAISS (which uses IndexFlatL2)
        3. Print distance for debugging
        4. Validate against threshold
        """
        # Normalize embedding BEFORE matching
        query_vector = self.normalize_embedding(query_vector)
        q_vec = query_vector.astype(np.float32)
        
        faiss_engine = db.get_faiss_engine()
        
        try:
            best_id, distance = faiss_engine.search(q_vec, k=1)
        except Exception:
            return "Unknown", 999.0
            
        decision = "Known" if distance < self.threshold else "Unknown"
        final_id = best_id if distance < self.threshold else "Unknown"
        
        print(f"[DEBUG] Detected person name: {best_id}")
        print(f"[DEBUG] Similarity distance: {distance:.4f}")
        print(f"[DEBUG] Threshold value: {self.threshold:.4f}")
        print(f"[DEBUG] Final decision: {decision}")
        
        return final_id, distance

if __name__ == "__main__":
    matcher = SimilarityMatcher(threshold=0.6)
    vec = np.array([1, 2, 3], dtype=np.float32)
    normed = matcher.normalize_embedding(vec)
    print(f"Normalized: {normed}")
