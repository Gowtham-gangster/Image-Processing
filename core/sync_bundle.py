"""
sync_bundle.py
==============
Build and apply a single zip bundle for local → Railway sync.
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile

from config import ROOT_DIR

SYNC_FILES = (
    ("database/persons.db", "database/persons.db"),
    ("dataset/persons.csv", "dataset/persons.csv"),
    ("alerts_config.json", "alerts_config.json"),
    ("camera_config.json", "camera_config.json"),
)

SYNC_EMBEDDING_NAMES = (
    "faiss_index.index",
    "body_faiss.index",
    "attr_faiss.index",
    "multi_labels.pkl",
    "labels.pkl",
)


def build_sync_zip(root: str | None = None) -> bytes:
    """Pack persons DB, embeddings, alerts, and optional dataset metadata."""
    root = root or ROOT_DIR
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_src, arc_name in SYNC_FILES:
            src = os.path.join(root, rel_src)
            if os.path.exists(src):
                zf.write(src, arc_name)

        emb_dir = os.path.join(root, "embeddings")
        if os.path.isdir(emb_dir):
            for name in SYNC_EMBEDDING_NAMES:
                path = os.path.join(emb_dir, name)
                if os.path.exists(path):
                    zf.write(path, f"embeddings/{name}")

    buf.seek(0)
    return buf.read()


def apply_sync_zip(zip_bytes: bytes, data_dir: str) -> dict:
    """Extract sync bundle into DATA_DIR (Railway volume)."""
    os.makedirs(data_dir, exist_ok=True)
    extracted: list[str] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            target = os.path.join(data_dir, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(name)

    return {"data_dir": data_dir, "files": extracted}
