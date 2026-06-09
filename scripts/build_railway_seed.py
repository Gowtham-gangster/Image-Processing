#!/usr/bin/env python3
"""
Build railway_seed/ from local data — commit this folder so Railway deploys with data built-in.

Usage (run once when local data changes):
  python scripts/build_railway_seed.py
"""

from __future__ import annotations

import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = os.path.join(ROOT, "railway_seed")

EMBEDDING_NAMES = (
    "faiss_index.index",
    "body_faiss.index",
    "attr_faiss.index",
    "multi_labels.pkl",
    "labels.pkl",
)


def main() -> None:
    if os.path.exists(SEED):
        shutil.rmtree(SEED)
    os.makedirs(SEED, exist_ok=True)

    pairs = [
        (os.path.join(ROOT, "database", "persons.db"), os.path.join(SEED, "database", "persons.db")),
        (os.path.join(ROOT, "dataset", "persons.csv"), os.path.join(SEED, "dataset", "persons.csv")),
    ]
    for src, dst in pairs:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  + {os.path.relpath(dst, ROOT)}")

    emb_src = os.path.join(ROOT, "embeddings")
    emb_dst = os.path.join(SEED, "embeddings")
    if os.path.isdir(emb_src):
        os.makedirs(emb_dst, exist_ok=True)
        for name in EMBEDDING_NAMES:
            path = os.path.join(emb_src, name)
            if os.path.exists(path):
                shutil.copy2(path, os.path.join(emb_dst, name))
                print(f"  + railway_seed/embeddings/{name}")

    alerts_src = os.path.join(ROOT, "alerts_config.json")
    alerts_dst = os.path.join(SEED, "alerts_config.json")
    if os.path.exists(alerts_src):
        with open(alerts_src, encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg.get("email"), dict) and cfg["email"].get("password"):
            cfg["email"] = dict(cfg["email"])
            cfg["email"]["password"] = ""
        with open(alerts_dst, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print("  + railway_seed/alerts_config.json (password stripped for git)")

    db = os.path.join(SEED, "database", "persons.db")
    size_kb = os.path.getsize(db) / 1024 if os.path.exists(db) else 0
    print(f"\nDone. railway_seed/ ready ({size_kb:.0f} KB DB). Commit and push — Railway will auto-seed the volume.")


if __name__ == "__main__":
    main()
