"""
data_seed.py
============
Import persons from dataset/persons.csv when the database is empty.
"""

from __future__ import annotations

import csv
import logging
import os

from config import DATASET_DIR, ROOT_DIR
from database import DatabaseManager

logger = logging.getLogger(__name__)


def persons_csv_path() -> str:
    data_root = os.environ.get("DATA_DIR", ROOT_DIR)
    for path in (
        os.path.join(data_root, "dataset", "persons.csv"),
        os.path.join(DATASET_DIR, "persons.csv"),
    ):
        if os.path.exists(path):
            return path
    return os.path.join(data_root, "dataset", "persons.csv")


def seed_persons_from_csv(db: DatabaseManager | None = None) -> int:
    """Import persons.csv rows when the persons table is empty. Returns count imported."""
    db = db or DatabaseManager()
    if db.all_persons():
        logger.info("Persons table already has %d rows — skipping CSV seed.", len(db.all_persons()))
        return 0

    csv_path = persons_csv_path()
    if not os.path.exists(csv_path):
        logger.warning("No persons.csv at %s — cannot seed.", csv_path)
        return 0

    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = (row.get("person_id") or "").strip()
            if not pid:
                continue
            db.add_person(
                pid,
                row.get("name", "Unknown"),
                row.get("gender", "N/A"),
                row.get("age", "N/A"),
                row.get("phone", "N/A"),
                row.get("address", "N/A"),
            )
            count += 1

    logger.info("Seeded %d persons from %s", count, csv_path)
    return count
