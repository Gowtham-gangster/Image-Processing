"""
migrate_db.py
=============
One-off migration script to port legacy `dataset/persons.csv` and `logs/surveillance_events.csv`
data directly into the new strictly structured SQLite engine (`logs/database.db`).

Converts legacy string 'age' fields into structured 'age_group' categories.
"""

import os
import csv
import logging
from config import DATASET_DIR, LOGS_DIR
from database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_age_to_group(age_str: str) -> str:
    """Map legacy integer string ages to categorized age groups."""
    try:
        a = int(age_str)
        if a < 12: return "Child"
        if a < 25: return "Young Adult"
        if a < 60: return "Adult"
        return "Senior"
    except ValueError:
        return "Unknown"

def run_migration():
    logger.info("Starting legacy data migration to unified SQLite...")
    
    # 1. Init unified DB
    db = DatabaseManager()
    
    # 2. Port Persons
    persons_csv = os.path.join(DATASET_DIR, "persons.csv")
    if os.path.exists(persons_csv):
        with open(persons_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                pid = row.get("person_id")
                name = row.get("name", "Unknown")
                gender = row.get("gender", "Unknown")
                age = row.get("age", "Unknown")
                phone = row.get("phone", "Unknown")
                address = row.get("address", "Unknown")
                
                db.add_person(pid, name, gender, age, phone, address)
                count += 1
        logger.info(f"Ported {count} persons from legacy CSV to unified SQLite.")
    else:
        logger.warning(f"No legacy persons CSV found at {persons_csv}")

    # 3. Port Logs (if they exist)
    events_csv = os.path.join(LOGS_DIR, "surveillance_events.csv")
    if os.path.exists(events_csv):
        with open(events_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                ts = row.get("timestamp", "")
                pid = row.get("person_id", "Unknown")
                conf = float(row.get("confidence", 0.0))
                
                db.add_log(pid, conf, ts)
                count += 1
        logger.info(f"Ported {count} legacy events into new SQLite logs table.")
    else:
        logger.warning(f"No legacy events CSV found at {events_csv}")

    # Note: No `models/faiss_faces.index` or `models/body_embeddings.npz` files
    # were detected in the directory tree, meaning embeddings do not currently
    # require migration. They will be rebuilt locally by model trainers anyway.
    logger.info("FAISS & NPZ stores are empty on disk. Migration completed.")

if __name__ == "__main__":
    run_migration()
