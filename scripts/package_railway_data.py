#!/usr/bin/env python3
"""
Package local database, embeddings, and dataset for Railway volume upload.

Usage:
  python scripts/package_railway_data.py

Creates railway_data.zip with:
  database/persons.db
  embeddings/*
  dataset/train/*
  dataset/persons.csv

Upload to Railway volume mounted at /app/data (extract zip contents there).
"""

from __future__ import annotations

import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "railway_data.zip")
STAGE = os.path.join(ROOT, "_railway_data_staging")


def main() -> None:
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE, exist_ok=True)

    copies = [
        (os.path.join(ROOT, "database", "persons.db"), os.path.join(STAGE, "database", "persons.db")),
        (os.path.join(ROOT, "dataset", "persons.csv"), os.path.join(STAGE, "dataset", "persons.csv")),
    ]
    for src, dst in copies:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  + {os.path.relpath(src, ROOT)}")
        else:
            print(f"  ! missing {src}")

    emb_src = os.path.join(ROOT, "embeddings")
    emb_dst = os.path.join(STAGE, "embeddings")
    if os.path.isdir(emb_src):
        shutil.copytree(
            emb_src,
            emb_dst,
            ignore=shutil.ignore_patterns("*.npz", "person*.npz", "embeddings.pkl"),
        )
        print(f"  + embeddings/ (indexes + pkl)")

    train_src = os.path.join(ROOT, "dataset", "train")
    train_dst = os.path.join(STAGE, "dataset", "train")
    if os.path.isdir(train_src):
        shutil.copytree(train_src, train_dst)
        print(f"  + dataset/train/ ({len(os.listdir(train_src))} persons)")

    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, _, files in os.walk(STAGE):
            for name in files:
                full = os.path.join(folder, name)
                arc = os.path.relpath(full, STAGE)
                zf.write(full, arc)

    shutil.rmtree(STAGE)
    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    print(f"\nCreated {OUT} ({size_mb:.1f} MB)")
    print("\nRailway volume steps:")
    print("  1. Mount volume at /app/data")
    print("  2. Set env DATA_DIR=/app/data")
    print("  3. Extract railway_data.zip into /app/data")
    print("  4. Redeploy — persons + embeddings load automatically")


if __name__ == "__main__":
    main()
