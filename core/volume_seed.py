"""
volume_seed.py
==============
On Railway, copy bundled railway_seed/ into the persistent volume when empty.
Runs once — after that the volume keeps data across redeploys.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3

from config import ROOT_DIR

logger = logging.getLogger(__name__)

SEED_DIR = os.path.join(ROOT_DIR, "railway_seed")


def _person_count(db_path: str) -> int:
    if not os.path.exists(db_path):
        return 0
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT COUNT(*) FROM persons").fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _copy_tree(src: str, dst: str) -> list[str]:
    copied: list[str] = []
    if not os.path.isdir(src):
        return copied
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target_root, exist_ok=True)
        for name in files:
            s = os.path.join(root, name)
            d = os.path.join(target_root, name)
            if not os.path.exists(d):
                shutil.copy2(s, d)
                copied.append(os.path.relpath(d, dst))
    return copied


def ensure_volume_seeded(data_dir: str) -> dict:
    """
    If DATA_DIR has no persons yet, seed from railway_seed/ shipped in the Docker image.
  """
    db_path = os.path.join(data_dir, "database", "persons.db")
    existing = _person_count(db_path)

    if existing > 0:
        logger.info("Volume already has %d persons at %s — skipping seed.", existing, db_path)
        return {"seeded": False, "persons": existing, "reason": "volume_not_empty"}

    if not os.path.isdir(SEED_DIR):
        logger.warning("No railway_seed/ bundle in image — volume stays empty until sync.")
        return {"seeded": False, "persons": 0, "reason": "no_seed_bundle"}

    logger.info("Empty volume detected — seeding from %s into %s", SEED_DIR, data_dir)
    copied: list[str] = []

    for sub in ("database", "embeddings", "dataset"):
        src = os.path.join(SEED_DIR, sub)
        if os.path.isdir(src):
            copied.extend(_copy_tree(src, os.path.join(data_dir, sub)))

    alerts_seed = os.path.join(SEED_DIR, "alerts_config.json")
    alerts_dst = os.path.join(data_dir, "alerts_config.json")
    if os.path.exists(alerts_seed) and not os.path.exists(alerts_dst):
        shutil.copy2(alerts_seed, alerts_dst)
        copied.append("alerts_config.json")

    persons = _person_count(db_path)
    logger.info("Volume seed complete: %d persons, %d files copied.", persons, len(copied))
    return {"seeded": True, "persons": persons, "files_copied": len(copied), "files": copied[:20]}
