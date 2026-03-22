"""
database.py
===========
Centralized SQLite Database Manager for the Mask-Aware Hybrid ID system.
Handles persons, face embeddings, body embeddings, and logs.
"""

import os
import sqlite3
import logging
import numpy as np
DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "persons.db")

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        if not os.path.exists(os.path.dirname(self.db_path)):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    gender TEXT,
                    age TEXT,
                    phone TEXT,
                    address TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    embedding BLOB,
                    FOREIGN KEY(person_id) REFERENCES persons(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS body_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    embedding BLOB,
                    FOREIGN KEY(person_id) REFERENCES persons(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    FOREIGN KEY(person_id) REFERENCES persons(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT,
                    camera_id TEXT,
                    person_id TEXT,
                    confidence REAL,
                    timestamp TEXT
                )
            """)
        logger.info("Unified SQLite Database initialized at %s.", self.db_path)

    # ── Persons API ────────────────────────────────────────────────────────
    
    def get_person(self, person_id: str) -> dict | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def add_person(self, person_id: str, name: str, gender: str, age: str, phone: str, address: str) -> None:
        if person_id is None:
            return
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO persons (id, name, gender, age, phone, address)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (person_id, name, gender, age, phone, address))

    def all_persons(self) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM persons")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_name(self, person_id: str) -> str:
        person = self.get_person(person_id)
        return person["name"] if person else "Unknown Person"
        
    def reload(self) -> None:
        pass

    # ── Face Embeddings API ────────────────────────────────────────────────
    
    def add_face_embedding(self, person_id: str, embedding: np.ndarray) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO face_embeddings (person_id, embedding) VALUES (?, ?)",
                (person_id, embedding.astype(np.float32).tobytes())
            )

    def get_all_face_embeddings(self) -> tuple[list[np.ndarray], list[str]]:
        vecs = []
        labels = []
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT person_id, embedding FROM face_embeddings")
            for row in cursor.fetchall():
                pid = row["person_id"]
                emb_bytes = row["embedding"]
                if emb_bytes:
                    vec = np.frombuffer(emb_bytes, dtype=np.float32)
                    # Reshape or keep 1D, since FAISS expects contiguous matrix later
                    vecs.append(vec)
                    labels.append(pid)
        return vecs, labels

    def clear_face_embeddings(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM face_embeddings")

    # ── Body Embeddings API ────────────────────────────────────────────────
    
    def add_body_embedding(self, person_id: str, embedding: np.ndarray) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO body_embeddings (person_id, embedding) VALUES (?, ?)",
                (person_id, embedding.astype(np.float32).tobytes())
            )

    def get_all_body_embeddings(self) -> tuple[list[np.ndarray], list[str]]:
        vecs = []
        labels = []
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT person_id, embedding FROM body_embeddings")
            for row in cursor.fetchall():
                pid = row["person_id"]
                emb_bytes = row["embedding"]
                if emb_bytes:
                    vec = np.frombuffer(emb_bytes, dtype=np.float32)
                    vecs.append(vec)
                    labels.append(pid)
        return vecs, labels

    def clear_body_embeddings(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM body_embeddings")

    # ── Logs API ───────────────────────────────────────────────────────────
    
    def add_log(self, person_id: str, confidence: float, timestamp: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO logs (person_id, confidence, timestamp) VALUES (?, ?, ?)",
                (person_id, confidence, timestamp)
            )

    def get_recent_logs(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM logs ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ── Alerts API ─────────────────────────────────────────────────────────

    def add_alert(self, alert_type: str, camera_id: str, person_id: str, confidence: float, timestamp: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO alerts (alert_type, camera_id, person_id, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
                (alert_type, camera_id, person_id, confidence, timestamp)
            )

    def get_recent_alerts(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            return [dict(row) for row in cursor.fetchall()]

# Alias for drop-in compatibility with legacy modules referencing PersonDatabase
PersonDatabase = DatabaseManager
