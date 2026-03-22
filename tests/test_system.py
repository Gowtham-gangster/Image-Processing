"""
test_system.py
==============
End-to-end testing script for the Mask-Aware Hybrid Person Identification System.

Steps
-----
1. Load the trained classifier  (models/person_identifier.pkl)
2. Read every image from        dataset/test/<person_id>/
3. Detect face in each image    (detection.py → Detector)
4. Extract embedding            (embeddings.py → EmbeddingExtractor)
5. Predict person_id            (SVM / KNN pipeline)
6. Retrieve attributes          (attributes.py → AttributeStore)
7. Display results + accuracy summary

Example output
--------------
═══════════════════════════════════════════════════════════════
 Image     : dataset/test/person1/img001.jpg
 True ID   : person1
───────────────────────────────────────────────────────────────
 Name      : John
 Gender    : Male
 Age       : 22
 Phone     : 9876543210
 Address   : Hyderabad
 Confidence: 91.3%  |  Similarity: 87.6%
═══════════════════════════════════════════════════════════════

 Image     : dataset/test/person9/img001.jpg
 True ID   : person9
───────────────────────────────────────────────────────────────
 ✖  Unknown Person
═══════════════════════════════════════════════════════════════

Usage
-----
    python test_system.py
    python test_system.py --clf-thresh 0.6 --sim-thresh 0.55
    python test_system.py --show                 # display annotated images
    python test_system.py --save-log             # write results to logs/
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import sys

import cv2

from config import TEST_DIR, LOGS_DIR, LOG_LEVEL
from attributes import AttributeStore
from recognition import Recognizer, RecognitionResult

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── Display constants ─────────────────────────────────────────────────────────
_WIDE  = "═" * 63
_THIN  = "─" * 63
_BLANK = " " * 63


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_test_images(test_dir: str) -> list[tuple[str, str]]:
    """
    Walk test_dir and return sorted (person_id, image_path) pairs.
    """
    items: list[tuple[str, str]] = []
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory not found: '{test_dir}'"
        )
    for pid in sorted(os.listdir(test_dir)):
        person_dir = os.path.join(test_dir, pid)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                items.append((pid, os.path.join(person_dir, fname)))
    return items


def _print_result(true_pid: str, img_path: str, result: RecognitionResult) -> None:
    """Pretty-print one recognition result to stdout."""
    print(_WIDE)
    # Shorten the path for readability
    short_path = os.path.relpath(img_path)
    print(f" Image     : {short_path}")
    print(f" True ID   : {true_pid}")
    print(_THIN)

    if result.is_known:
        attrs = result.attributes
        print(f" Name      : {attrs.get('name',    'N/A')}")
        print(f" Gender    : {attrs.get('gender',  'N/A')}")
        print(f" Age       : {attrs.get('age',     'N/A')}")
        print(f" Phone     : {attrs.get('phone',   'N/A')}")
        print(f" Address   : {attrs.get('address', 'N/A')}")
        match = "✔  Correct" if result.person_id == true_pid else "✘  Wrong"
        print(f" Match     : {match}  (predicted: {result.person_id})")
        print(
            f" Confidence: {result.clf_confidence:.1%}"
            f"  |  Similarity: {result.embed_similarity:.1%}"
        )
    else:
        print(" ✖  Unknown Person")
        print(
            f"    (clf={result.clf_confidence:.1%}"
            f"  sim={result.embed_similarity:.1%}"
            "  — below threshold)"
        )


def _print_summary(stats: dict) -> None:
    """Print overall accuracy summary."""
    total   = stats["total"]
    known   = stats["known"]
    unknown = stats["unknown"]
    correct = stats["correct"]
    wrong   = stats["wrong"]
    no_face = stats["no_face"]

    acc = correct / total if total else 0.0

    print()
    print("═" * 63)
    print("  SUMMARY")
    print("─" * 63)
    print(f"  Total images   : {total}")
    print(f"  Faces detected : {total - no_face}  (no face: {no_face})")
    print(f"  Known          : {known}  |  Unknown : {unknown}")
    print(f"  Correct IDs    : {correct}  |  Wrong   : {wrong}")
    print(f"  Accuracy       : {acc:.2%}")
    print("═" * 63)
    print()


def _save_log(rows: list[dict], log_dir: str = LOGS_DIR) -> str:
    """Write per-image results to a CSV log file."""
    os.makedirs(log_dir, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"test_results_{ts}.csv")
    fieldnames = [
        "image_path", "true_person_id", "predicted_person_id",
        "is_known", "correct", "name", "gender", "age", "phone", "address",
        "clf_confidence", "embed_similarity",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ── Main test routine ─────────────────────────────────────────────────────────

def run_test(
    test_dir: str = TEST_DIR,
    clf_threshold: float = 0.55,
    sim_threshold: float = 0.50,
    dl_backend: str = "auto",
    det_backend: str = "auto",
    show: bool = False,
    save_log: bool = False,
) -> dict:
    """
    Run the full test pipeline and return a stats dict.

    Parameters
    ----------
    test_dir      : Root of dataset/test/.
    clf_threshold : Min classifier confidence to accept a prediction.
    sim_threshold : Min cosine similarity to accept a prediction.
    dl_backend    : 'auto', 'tf', or 'torch'.
    det_backend   : 'auto', 'dnn', or 'haar'.
    show          : Display annotated image windows (press any key to continue).
    save_log      : Write per-image results to logs/test_results_*.csv.
    """

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("\n  Loading model and components …")
    try:
        rec = Recognizer(
            clf_threshold=clf_threshold,
            sim_threshold=sim_threshold,
            dl_backend=dl_backend,
            detect_backend=det_backend,
        )
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}\n", file=sys.stderr)
        sys.exit(1)

    store = AttributeStore()

    # ── Walk test images ───────────────────────────────────────────────────────
    try:
        items = _list_test_images(test_dir)
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}\n", file=sys.stderr)
        sys.exit(1)

    if not items:
        print(f"\n  No test images found in '{test_dir}'.\n")
        sys.exit(0)

    print(f"  Found {len(items)} test image(s) across "
          f"{len(set(p for p, _ in items))} person(s).\n")

    # ── Per-image processing ───────────────────────────────────────────────────
    stats = {"total": 0, "known": 0, "unknown": 0,
             "correct": 0, "wrong": 0, "no_face": 0}
    log_rows: list[dict] = []

    for true_pid, img_path in items:
        stats["total"] += 1

        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning("Cannot read '%s' — skipping.", img_path)
            stats["no_face"] += 1
            continue

        results = rec.recognize(frame)

        # If no face detected at all, create a blank unknown result
        if not results:
            stats["no_face"] += 1
            from recognition import RecognitionResult
            result = RecognitionResult()
        else:
            result = results[0]   # best detection (highest confidence first)

        # Counters
        if result.is_known:
            stats["known"] += 1
            if result.person_id == true_pid:
                stats["correct"] += 1
            else:
                stats["wrong"] += 1
        else:
            stats["unknown"] += 1

        # Print card
        _print_result(true_pid, img_path, result)

        # Display annotated image (optional)
        if show:
            _, vis = rec.recognize_frame(frame, annotate=True)
            cv2.imshow("Test — press any key", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Accumulate log row
        if save_log:
            attrs = result.attributes
            log_rows.append({
                "image_path":        img_path,
                "true_person_id":    true_pid,
                "predicted_person_id": result.person_id or "",
                "is_known":          result.is_known,
                "correct":           result.is_known and result.person_id == true_pid,
                "name":              attrs.get("name",    ""),
                "gender":            attrs.get("gender",  ""),
                "age":               attrs.get("age",     ""),
                "phone":             attrs.get("phone",   ""),
                "address":           attrs.get("address", ""),
                "clf_confidence":    round(result.clf_confidence,   4),
                "embed_similarity":  round(result.embed_similarity, 4),
            })

    # ── Summary ────────────────────────────────────────────────────────────────
    _print_summary(stats)

    # ── Save log ───────────────────────────────────────────────────────────────
    if save_log and log_rows:
        log_path = _save_log(log_rows)
        print(f"  Results saved → {log_path}\n")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the Mask-Aware Person Identification System on dataset/test/."
    )
    parser.add_argument(
        "--test-dir", default=TEST_DIR,
        help=f"Path to test split root (default: {TEST_DIR}).",
    )
    parser.add_argument(
        "--clf-thresh", type=float, default=0.55,
        help="Min classifier confidence (default: 0.55).",
    )
    parser.add_argument(
        "--sim-thresh", type=float, default=0.50,
        help="Min cosine similarity (default: 0.50).",
    )
    parser.add_argument(
        "--dl-backend", choices=["auto", "tf", "torch"], default="auto",
        help="Embedding extractor backend (default: auto).",
    )
    parser.add_argument(
        "--det-backend", choices=["auto", "dnn", "haar"], default="auto",
        help="Face detector backend (default: auto).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display annotated image for each test sample.",
    )
    parser.add_argument(
        "--save-log", action="store_true",
        help="Save per-image results to logs/test_results_<timestamp>.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_test(
        test_dir      = args.test_dir,
        clf_threshold = args.clf_thresh,
        sim_threshold = args.sim_thresh,
        dl_backend    = args.dl_backend,
        det_backend   = args.det_backend,
        show          = args.show,
        save_log      = args.save_log,
    )
