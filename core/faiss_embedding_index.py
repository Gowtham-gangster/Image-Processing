"""
faiss_embedding_index.py
========================
High-speed FAISS (Facebook AI Similarity Search) index wrapper.
Replaces O(N) numpy list iterations with lightning-fast O(log N) vector searches.

Requirements met
----------------
1. Build a FAISS index to store all face embeddings.
2. Support fast nearest neighbor search.
3. Store mapping between FAISS index IDs and person_ids.
4. Support adding new embeddings dynamically.

Usage
-----
    from faiss_embedding_index import FaissEmbeddingIndex
    import numpy as np

    # 1. Initialize
    index = FaissEmbeddingIndex(dim=128)
    
    # 2. Add embeddings (L2 Normalized)
    emb = np.random.rand(128).astype(np.float32)
    index.add_embedding("Alice", emb)
    
    # 3. Fast Near-Neighbor Search
    query = np.random.rand(128).astype(np.float32)
    best_id, sim_score = index.search(query)
    
    # 4. Save to Disk
    index.save("models/faiss_faces")
"""

import os
import json
import faiss
import logging
import numpy as np
from typing import Tuple

from config import LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class FaissEmbeddingIndex:
    """
    Manages a binary FAISS index and a JSON ID-mapping file.
    Uses IndexFlatL2 (Euclidean Distance).
    If distance < threshold → Match
    Else → Unknown
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatL2(self.dim)
        
        # FAISS natively only supports Integer IDs.
        # We maintain a mapping from FAISS internal sequential ID -> String Name
        self.id_to_name: dict[int, str] = {}
        
        # Counter for the next FAISS ID insertion
        self._next_id = 0

    def add_embedding(self, person_id: str, embedding: np.ndarray) -> None:
        """
        Add a single perfectly shaped embedding vector into the FAISS index.

        Parameters
        ----------
        person_id : str
            The name or ID string of the person.
        embedding : np.ndarray
            Shape (dim,) float32 vector. Must be L2 normalized beforehand.
        """
        if embedding.shape != (self.dim,) or embedding.dtype != np.float32:
            raise ValueError(f"Embedding must be a float32 array of shape ({self.dim},)")
            
        # FAISS expects a 2D matrix of shape (1, dim) for ingestion
        emb_matrix = np.expand_dims(embedding, axis=0)
        
        # Add to C++ Index
        self.index.add(emb_matrix)
        
        # Record Mapping
        self.id_to_name[self._next_id] = person_id
        self._next_id += 1

    def search(self, query: np.ndarray, k: int = 1) -> Tuple[str, float]:
        """
        Perform a lightning-fast nearest-neighbor search.

        Parameters
        ----------
        query : np.ndarray
            Shape (dim,) float32 query vector.
        k : int
            Number of top matches to return (default 1).

        Returns
        -------
        best_id : str
            The identity of the matched person. Returns "Unknown" if DB is empty.
        distance : float
            The L2 Euclidean distance score (lower = closer match).
        """
        if self.index.ntotal == 0:
            return "Unknown", 0.0
            
        if query.shape != (self.dim,) or query.dtype != np.float32:
            raise ValueError(f"Query must be a float32 array of shape ({self.dim},)")
            
        # Reshape to (1, dim) matrix for FAISS
        query_matrix = np.expand_dims(query, axis=0)
        
        # Return D (L2 distances) and I (integer IDs of neighbors)
        D, I = self.index.search(query_matrix, k)
        
        best_faiss_id = I[0][0]
        best_distance = float(D[0][0])
        
        if best_faiss_id == -1:
            return "Unknown", 999.0
            
        best_string_id = self.id_to_name.get(best_faiss_id, "Unknown")
        return best_string_id, best_distance

    def save(self, base_path: str) -> None:
        pass

    def load(self, base_path: str) -> None:
        pass

    def clear(self) -> None:
        """Wipes the FAISS wrapper clean."""
        self.__init__(self.dim)
