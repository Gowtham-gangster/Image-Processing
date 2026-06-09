"""
attributes.py
=============
Load and query person attributes from dataset/persons.csv.

CSV schema
----------
person_id, name, gender, age, phone, address

Usage (module)
--------------
    from attributes import AttributeStore

    store = AttributeStore()

    # Get full attribute dict
    attrs = store.get("person3")
    # {'name': 'Mary', 'gender': 'Female', 'age': '20',
    #  'phone': '9876512345', 'address': 'Chennai'}

    # Get individual fields
    print(store.name("person3"))     # Mary
    print(store.gender("person3"))   # Female
    print(store.age("person3"))      # 20
    print(store.phone("person3"))    # 9876512345
    print(store.address("person3"))  # Chennai

    # Pretty-print a person card
    store.display("person3")

    # List all persons
    for pid, row in store.all():
        print(pid, row["name"])

Usage (CLI)
-----------
    python attributes.py person3
    python attributes.py person3 --field name
    python attributes.py             # lists all persons
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd

from config import PERSONS_CSV

# ── AttributeStore ────────────────────────────────────────────────────────────

class AttributeStore:
    """
    Thin wrapper over persons.csv that exposes per-person attribute lookups.

    Parameters
    ----------
    csv_path : str
        Path to the persons CSV file.  Defaults to config.PERSONS_CSV.
    """

    # All columns expected in the CSV
    COLUMNS = ["person_id", "name", "gender", "age", "phone", "address"]

    def __init__(self, csv_path: str = PERSONS_CSV) -> None:
        self.csv_path = csv_path
        self._df: pd.DataFrame = pd.DataFrame()
        self._load()

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Read CSV into a DataFrame indexed by person_id."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"persons.csv not found at '{self.csv_path}'.\n"
                "Expected path: dataset/persons.csv"
            )

        df = pd.read_csv(self.csv_path, dtype=str)

        # Validate columns
        missing = set(self.COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"persons.csv is missing required columns: {missing}\n"
                f"Expected: {self.COLUMNS}"
            )

        # Strip surrounding whitespace from all string cells
        df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

        self._df = df.set_index("person_id")

    def reload(self) -> None:
        """Re-read the CSV from disk (useful after external edits)."""
        self._load()

    # ── Core lookup ───────────────────────────────────────────────────────────

    def get(self, person_id: str) -> Optional[dict[str, str]]:
        """
        Return all attributes for *person_id* as a dict, or ``None``.

        Returns
        -------
        dict with keys: name, gender, age, phone, address
        or ``None`` if *person_id* is not found.
        """
        person_id = str(person_id).strip()
        if person_id not in self._df.index:
            return None
        return self._df.loc[person_id].to_dict()

    def exists(self, person_id: str) -> bool:
        """Return True if *person_id* exists in the database."""
        return str(person_id).strip() in self._df.index

    # ── Field accessors ───────────────────────────────────────────────────────

    def _field(self, person_id: str, column: str,
               default: str = "N/A") -> str:
        attrs = self.get(person_id)
        if attrs is None:
            return default
        return attrs.get(column, default)

    def name(self,    person_id: str, default: str = "Unknown") -> str:
        """Return the person's name, or *default* if not found."""
        return self._field(person_id, "name", default)

    def gender(self,  person_id: str, default: str = "N/A") -> str:
        """Return the person's gender."""
        return self._field(person_id, "gender", default)

    def age(self,     person_id: str, default: str = "N/A") -> str:
        """Return the person's age."""
        return self._field(person_id, "age", default)

    def phone(self,   person_id: str, default: str = "N/A") -> str:
        """Return the person's phone number."""
        return self._field(person_id, "phone", default)

    def address(self, person_id: str, default: str = "N/A") -> str:
        """Return the person's address."""
        return self._field(person_id, "address", default)

    # ── Display helpers ───────────────────────────────────────────────────────

    def display(self, person_id: str) -> None:
        """
        Pretty-print a person card to stdout.
        Prints a "not found" message if the person_id is invalid.
        """
        attrs = self.get(person_id)
        print()
        if attrs is None:
            print(f"  ✖  person_id '{person_id}' not found in persons.csv.")
            print()
            return

        sep = "─" * 40
        print(sep)
        print(f"  Person ID : {person_id}")
        print(f"  Name      : {attrs.get('name',    'N/A')}")
        print(f"  Gender    : {attrs.get('gender',  'N/A')}")
        print(f"  Age       : {attrs.get('age',     'N/A')}")
        print(f"  Phone     : {attrs.get('phone',   'N/A')}")
        print(f"  Address   : {attrs.get('address', 'N/A')}")
        print(sep)
        print()

    def display_all(self) -> None:
        """Print a summary table of all persons."""
        if self._df.empty:
            print("  (no records in persons.csv)")
            return

        col_w = [12, 20, 8, 5, 14, 30]
        headers = ["person_id", "name", "gender", "age", "phone", "address"]
        hdr = "  " + "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
        sep = "  " + "─" * (sum(col_w) + 2 * len(col_w))

        print()
        print(sep)
        print(hdr)
        print(sep)
        for pid, row in self._df.iterrows():
            vals = [str(pid)] + [str(row.get(c, "")) for c in headers[1:]]
            print("  " + "  ".join(v.ljust(col_w[i]) for i, v in enumerate(vals)))
        print(sep)
        print(f"  Total: {len(self._df)} person(s)")
        print()

    # ── Iteration ─────────────────────────────────────────────────────────────

    def all(self) -> list[tuple[str, dict[str, str]]]:
        """
        Return a list of (person_id, attributes_dict) for all records.
        """
        return [
            (pid, row.to_dict())
            for pid, row in self._df.iterrows()
        ]

    def person_ids(self) -> list[str]:
        """Return all person_ids in the database."""
        return list(self._df.index)

    def count(self) -> int:
        """Return the number of persons in the database."""
        return len(self._df)

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, person_id: str) -> bool:
        return self.exists(person_id)

    def __repr__(self) -> str:
        return (
            f"AttributeStore(csv='{self.csv_path}', "
            f"persons={len(self._df)})"
        )


# ── Module-level convenience functions ────────────────────────────────────────

_default_store: Optional[AttributeStore] = None


def _get_store() -> AttributeStore:
    global _default_store
    if _default_store is None:
        _default_store = AttributeStore()
    return _default_store


def get_attributes(person_id: str) -> Optional[dict[str, str]]:
    """
    Module-level shortcut: return all attributes for *person_id*, or ``None``.

    Example
    -------
        from attributes import get_attributes

        attrs = get_attributes("person3")
        if attrs:
            print(attrs["name"], attrs["phone"])
    """
    return _get_store().get(person_id)


def get_name(person_id: str, default: str = "Unknown") -> str:
    """Return just the name for *person_id*."""
    return _get_store().name(person_id, default)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query person attributes from persons.csv."
    )
    parser.add_argument(
        "person_id", nargs="?", default=None,
        help="person_id to look up (omit to list all persons).",
    )
    parser.add_argument(
        "--field",
        choices=["name", "gender", "age", "phone", "address"],
        default=None,
        help="Return only this specific field.",
    )
    parser.add_argument(
        "--csv", default=PERSONS_CSV,
        help=f"Path to persons.csv (default: {PERSONS_CSV}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    try:
        store = AttributeStore(csv_path=args.csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── List all persons ──────────────────────────────────────────────────────
    if args.person_id is None:
        store.display_all()
        sys.exit(0)

    # ── Lookup a specific person_id ───────────────────────────────────────────
    if not store.exists(args.person_id):
        print(f"\n  ✖  '{args.person_id}' not found in persons.csv.\n",
              file=sys.stderr)
        sys.exit(1)

    if args.field:
        # Print only the requested field value
        val = store._field(args.person_id, args.field)
        print(val)
    else:
        # Full card
        store.display(args.person_id)
