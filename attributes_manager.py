"""
attributes_manager.py
=====================
Loads person attributes from persons.csv and maps person_id to their details.

Requirements met
----------------
1. Load person attributes from dataset/persons.csv.
2. Map person_id to attributes.
3. Return name, gender, age, phone, address when a person is identified.
4. Append new persons dynamically via GUI registration.

Usage
-----
    from attributes_manager import AttributesManager
    
    manager = AttributesManager()
    
    # Upon identifying "person1":
    attrs = manager.get_attributes("person1")
    
    print(attrs["name"])
    print(attrs["age_group"])
    print(attrs["gender"])
"""

import logging
from typing import Dict, Optional

from config import LOG_LEVEL
from database import PersonDatabase

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class AttributesManager:
    """
    In-memory cache of person attributes, backed by the Central PersonDatabase.
    """

    def __init__(self, db: Optional[PersonDatabase] = None) -> None:
        self.db = db or PersonDatabase()
        self._cache: Dict[str, Dict[str, str]] = {}
        self.reload()

    def reload(self) -> None:
        """
        Reload the cache from the database.
        """
        records = self.db.all_persons()
        self._cache.clear()
        
        if not records:
            logger.warning("Attributes database is empty. Cache is empty.")
            return
            
        # Convert list of rows to dict cache
        for row in records:
            pid = str(row.get("id", ""))
            if not pid: continue
            
            self._cache[pid] = {
                "name": str(row.get("name", "Unknown")).strip(),
                "gender": str(row.get("gender", "N/A")).strip(),
                "age": str(row.get("age", "N/A")).strip(),
                "phone": str(row.get("phone", "N/A")).strip(),
                "address": str(row.get("address", "N/A")).strip(),
            }
                
        logger.info("Loaded %d person attributes into cache", len(self._cache))

    def get_attributes(self, person_id: str) -> Optional[Dict[str, str]]:
        """
        Retrive the attributes map for a specific person.
        """
        pid = str(person_id).strip()
        if pid == "Unknown Person" or pid not in self._cache:
            return None
        return self._cache[pid]

    def add_person(self, person_id: str, name: str, gender: str, age: str, phone: str, address: str) -> None:
        """
        Appends a new person to the database and reloads the memory cache.
        """
        pid = str(person_id).strip()
        if not pid:
            raise ValueError("person_id cannot be empty")
            
        self.db.add_person(pid, name, gender, age, phone, address)
        self.reload()

    def __contains__(self, person_id: str) -> bool:
        return str(person_id).strip() in self._cache

if __name__ == "__main__":
    manager = AttributesManager()
    manager.add_person("p001", "Jane Doe", "Female", "28", "555-1234", "123 Main St")
    attrs = manager.get_attributes("p001")
    print("Retrieved identity test:")
    print(f"Name   : {attrs['name']}")
    print(f"Phone  : {attrs['phone']}")
