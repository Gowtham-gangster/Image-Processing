#!/usr/bin/env python3
"""
Sync local persons, embeddings, alerts, and analytics to Railway in one command.

Usage:
  python scripts/sync_to_railway.py
  python scripts/sync_to_railway.py https://image-processing-production-60da.up.railway.app

Optional env:
  RAILWAY_API_URL  — backend URL (same as Vercel VITE_API_URL)
  SYNC_SECRET      — must match Railway SYNC_SECRET if set
"""

from __future__ import annotations

import os
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from sync_bundle import build_sync_zip  # noqa: E402


def _resolve_url() -> str:
    if len(sys.argv) > 1:
        url = sys.argv[1].strip()
    else:
        url = os.environ.get("RAILWAY_API_URL", "").strip()
    if not url:
        url = input("Railway API URL (https://....up.railway.app): ").strip()
    if not url:
        sys.exit("No URL. Set RAILWAY_API_URL or pass as argument.")
    if not url.startswith("http"):
        url = f"https://{url}"
    return url.rstrip("/")


def main() -> None:
    url = _resolve_url()
    payload = build_sync_zip(ROOT)
    size_kb = len(payload) / 1024

    print(f"Syncing {size_kb:.1f} KB -> {url}")
    print("  - persons.db (names + analytics logs + alert history)")
    print("  - embeddings/ (FAISS indexes)")
    print("  - alerts_config.json")

    headers: dict[str, str] = {}
    secret = os.environ.get("SYNC_SECRET")
    if secret:
        headers["X-Sync-Secret"] = secret

    try:
        resp = requests.post(
            f"{url}/admin/sync",
            files={"bundle": ("sync.zip", payload, "application/zip")},
            headers=headers,
            timeout=300,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"\nSync failed: {exc}")
        if getattr(exc, "response", None) is not None:
            print(exc.response.text)
        sys.exit(1)

    data = resp.json()
    summary = data.get("summary", {})
    print("\nSync complete.")
    print(f"  Persons in cloud DB : {summary.get('persons_in_db', '?')}")
    print(f"  Events (analytics)  : {summary.get('events_in_db', '?')}")
    print(f"  Alert history rows  : {summary.get('alerts_in_db', '?')}")
    print(f"  FAISS face vectors  : {summary.get('faiss', {}).get('face', '?')}")
    print("\nRefresh your Vercel dashboard — Persons, Analytics, and Image Test should match local.")


if __name__ == "__main__":
    main()
