"""
body_embedding_database.py
==========================
Manages the storage and retrieval of 2048-dimensional body embeddings
using SQLite persistence and in-memory caching.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple

from database import DatabaseManager

logger = logging.getLogger(__name__)

class BodyEmbeddingDatabase:
    """
    In-memory and SQLite-persisted database for 2048-D body appearance embeddings.
    """

    def __init__(self, db: DatabaseManager = None) -> None:
        self.db = db or DatabaseManager()
        self.data: Dict[str, List[np.ndarray]] = {}
        self._load()

    def _load(self) -> None:
        self.data.clear()
        vecs, labels = self.db.get_all_body_embeddings()
        for vec, pid in zip(vecs, labels):
            if pid not in self.data:
                self.data[pid] = []
            self.data[pid].append(vec)
        logger.info("Loaded body embeddings database with %d identities.", len(self.data))

    def save(self) -> None:
        """No-op. Handled automatically via SQLite."""
        pass

    def add_embedding(self, person_id: str, embedding: np.ndarray) -> None:
        """Add embedding to SQLite and memory."""
        if embedding.shape != (2048,):
            raise ValueError(f"Expected embedding of shape (2048,), got {embedding.shape}")

        self.db.add_body_embedding(person_id, embedding)
        
        if person_id not in self.data:
            self.data[person_id] = []
        self.data[person_id].append(embedding.astype(np.float32))

    def get_all_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        all_vecs = []
        all_ids = []
        
        for pid, vecs in self.data.items():
            for v in vecs:
                all_vecs.append(v)
                all_ids.append(pid)
                
        if not all_vecs:
            return np.zeros((0, 2048), dtype=np.float32), np.array([], dtype=str)
            
        return np.vstack(all_vecs), np.array(all_ids, dtype=str)

    def clear(self) -> None:
        self.db.clear_body_embeddings()
        self.data.clear()

if __name__ == "__main__":
    db = BodyEmbeddingDatabase()
    db.add_embedding("user1", np.ones(2048, dtype=np.float32))
    x, y = db.get_all_embeddings()
    print(f"Matrix shape: {x.shape}, Labels shape: {y.shape}")
