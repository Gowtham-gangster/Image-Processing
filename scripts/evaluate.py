"""
evaluate.py
===========
Evaluate recognition accuracy on the test split.

Usage
-----
    python evaluate.py

Prints per-class accuracy, overall accuracy, and writes a report to logs/.
"""

import logging
import sys
import os
import csv
import datetime

from config import TEST_DIR, LOGS_DIR, LOG_LEVEL
from database import PersonDatabase
from face_detector import FaceDetector
from mask_detector import MaskDetector
from person_identifier import PersonIdentifier

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def evaluate(test_dir: str = TEST_DIR) -> dict:
    """
    Walk test_dir, run identification on each image, and compute metrics.

    Returns
    -------
    dict with keys: total, correct, accuracy, per_class
    """
    import cv2

    db = PersonDatabase()
    detector = FaceDetector()
    mask_det = MaskDetector()
    identifier = PersonIdentifier(db=db)

    if not identifier.is_ready():
        logger.error("Recognizer not loaded. Run model_trainer.py first.")
        sys.exit(1)

    total = correct = 0
    per_class: dict[str, dict] = {}

    for person_id in sorted(os.listdir(test_dir)):
        person_dir = os.path.join(test_dir, person_id)
        if not os.path.isdir(person_dir):
            continue

        per_class[person_id] = {"total": 0, "correct": 0}

        for fname in os.listdir(person_dir):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                continue
            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            boxes = detector.detect_faces(img)
            if not boxes:
                logger.debug("No face in %s — skipping.", img_path)
                continue

            (x, y, w, h) = max(boxes, key=lambda b: b[2] * b[3])
            face_roi = img[y:y + h, x:x + w]

            is_masked, _ = mask_det.is_masked(face_roi)
            result = identifier.identify(face_roi, is_masked=is_masked)

            total += 1
            per_class[person_id]["total"] += 1

            if result.person_id == person_id:
                correct += 1
                per_class[person_id]["correct"] += 1

    accuracy = correct / total if total else 0.0

    # Print report
    print("\n── Evaluation Report ──────────────────────────────────────────")
    print(f"  Total samples : {total}")
    print(f"  Correct       : {correct}")
    print(f"  Accuracy      : {accuracy:.2%}")
    print("\n  Per-class breakdown:")
    for pid, stats in per_class.items():
        t, c = stats["total"], stats["correct"]
        acc = c / t if t else 0
        print(f"    {pid:20s}  {c}/{t}  ({acc:.0%})")
    print("────────────────────────────────────────────────────────────────\n")

    # Save CSV report
    os.makedirs(LOGS_DIR, exist_ok=True)
    report_path = os.path.join(
        LOGS_DIR,
        f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "total", "correct", "accuracy"])
        for pid, stats in per_class.items():
            t, c = stats["total"], stats["correct"]
            writer.writerow([pid, t, c, f"{c/t:.4f}" if t else "0"])
        writer.writerow(["TOTAL", total, correct, f"{accuracy:.4f}"])
    logger.info("Evaluation report saved → %s", report_path)

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_class": per_class,
    }


if __name__ == "__main__":
    evaluate()
