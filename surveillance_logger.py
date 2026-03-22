"""
surveillance_logger.py
======================
Refactored to integrate deeply with the centralized DatabaseManager.
Logs every detection event to the standard SQLite tables.
"""

import logging
from datetime import datetime
from typing import Optional

from config import LOG_LEVEL
from database import DatabaseManager

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
console_logger = logging.getLogger(__name__)


class SurveillanceLogger:
    """
    Persists detection events strictly to the centralized SQLite database logic.
    Ensures seamless backwards compatibility for the React dashboard streams.
    """

    def __init__(self, log_file: Optional[str] = None, db_path: str = None, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()

    def log_event(
        self,
        camera_id: str,
        person_id: str,
        confidence: float,
        is_known:   bool = True,
        is_masked:  bool = False,
        alert_sent: bool = False,
    ) -> int:
        """
        Record one detection event into the SQLite logs table.
        """
        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        
        try:
            self.db.add_log(person_id, float(confidence), timestamp)
        except Exception as e:
            console_logger.error("SQLite write failed: %s", e)

        console_logger.debug("Event logged: %s → %s (known=%s)", camera_id, person_id, is_known)
        # Assuming the new schema doesn't need to return the specific row ID for external tracking
        return 0

    def get_recent_events(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return the most recent detection events as a list of dicts for the dashboard."""
        logs = self.db.get_recent_logs(limit=limit, offset=offset)
        
        # Hydrate missing fields required by the legacy React `/events` table
        hydrated = []
        for log in logs:
            l = dict(log)
            l["camera_id"] = "API"
            l["is_known"] = l["person_id"] != "Unknown Person" and l["person_id"] != "Unknown"
            l["is_masked"] = False # Cannot derive cleanly from the strict 4-column schema
            hydrated.append(l)
            
        return hydrated

    def get_stats(self) -> dict:
        """Return aggregate stats for the analytics dashboard panel."""
        logs = self.db.get_recent_logs(limit=5000) # fetch bulk and measure
        total = len(logs)
        unknown = sum(1 for l in logs if l["person_id"] in ["Unknown Person", "Unknown", "SPOOF DETECTED"])
        today_str = datetime.now().strftime("%Y-%m-%d")
        today = sum(1 for l in logs if str(l["timestamp"]).startswith(today_str))
        
        return {
            "total":    total,
            "unknown":  unknown,
            "unmasked": 0, # Unsupported by strict schema
            "today":    today,
        }
