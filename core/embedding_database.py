"""
embedding_database.py
=====================
Manages FaceNet 128-D embeddings using FAISS in memory, but solidly persisted
into the centralized SQLite relational database.
"""

import logging
import numpy as np
from database import DatabaseManager
from faiss_embedding_index import FaissEmbeddingIndex

logger = logging.getLogger(__name__)

class EmbeddingDatabase:
    """ SQLite-persisted FAISS database wrapper. """
    def __init__(self, db: DatabaseManager = None) -> None:
        self.db = db or DatabaseManager()
        self.faiss_engine = FaissEmbeddingIndex(dim=128)
        self._load()

    def _load(self) -> None:
        """Initialize FAISS instantly from SQLite BLOB vectors."""
        self.faiss_engine.clear()
        vecs, labels = self.db.get_all_face_embeddings()
        for vec, pid in zip(vecs, labels):
            self.faiss_engine.add_embedding(pid, vec)
        logger.info("Loaded FAISS engine with %d vectors from SQLite.", len(vecs))

    def save(self) -> None:
        """No-op as SQLite inserts immediately upon adding."""
        pass

    def add_embedding(self, person_id: str, embedding: np.ndarray) -> None:
        """
        Normalize embedding before storing and before matching.
        Write instantly to SQLite and update the in-memory fast indexing engine.
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        self.db.add_face_embedding(person_id, embedding)
        self.faiss_engine.add_embedding(person_id, embedding)

    def get_faiss_engine(self) -> FaissEmbeddingIndex:
        return self.faiss_engine

    def clear(self) -> None:
        self.db.clear_face_embeddings()
        self.faiss_engine.clear()

    def train_from_dataset(self) -> None:
        """
        Enforced Training Pipeline:
        1. Detect face
        2. Extract embeddings
        3. Normalize embeddings
        4. Store embeddings in database
        5. Build FAISS index
        """
        from dataset_loader import DatasetLoader
        from face_alignment import FaceAligner
        from embedding_model import FaceNetEmbedder
        
        logger.info("Starting strict dataset training pipeline...")
        loader = DatasetLoader()
        aligner = FaceAligner()
        embedder = FaceNetEmbedder()
        
        self.clear()
        
        count = 0
        for person_id, img_path, img_bgr in loader.load_training_data():
            # 1. Detect face
            faces = aligner.align(img_bgr)
            if not faces:
                continue
                
            best_face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
            aligned_crop = best_face["aligned_crop"]
            
            # 2. Extract embeddings & 3. Normalize embeddings
            # FaceNetEmbedder's extract internally computes features & L2 normalizes them
            emb = embedder.extract(aligned_crop)
            
            # 4. Store embeddings in database & 5. Build FAISS index
            self.add_embedding(person_id, emb)
            count += 1
            
        logger.info("Training complete. Indexed %d embeddings.", count)

if __name__ == "__main__":
    db = EmbeddingDatabase()
    db.add_embedding("user1", np.ones(128, dtype=np.float32))
    best_id, score = db.get_faiss_engine().search(np.ones(128, dtype=np.float32))
    print(f"Test match: {best_id} at {score:.2f}")
