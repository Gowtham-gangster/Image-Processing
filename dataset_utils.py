"""
dataset_utils.py
================
Image loading, preprocessing utilities, and label-map management.
"""

import os
import pickle
import logging
from typing import Generator

import cv2
import numpy as np

from config import TRAIN_DIR, FACE_SIZE, LABEL_MAP_PATH

logger = logging.getLogger(__name__)

# Supported image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Label map helpers ─────────────────────────────────────────────────────────

def save_label_map(label_map: dict, path: str = LABEL_MAP_PATH) -> None:
    """Persist int→person_id mapping to disk."""
    with open(path, "wb") as f:
        pickle.dump(label_map, f)
    logger.info("Label map saved (%d labels).", len(label_map))


def load_label_map(path: str = LABEL_MAP_PATH) -> dict:
    """Load int→person_id mapping from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Label map not found at {path}. Run model_trainer.py first."
        )
    with open(path, "rb") as f:
        label_map = pickle.load(f)
    logger.info("Label map loaded (%d labels).", len(label_map))
    return label_map


# ── Image utilities ───────────────────────────────────────────────────────────

def preprocess_face(face_roi: np.ndarray,
                    target_size: tuple = FACE_SIZE) -> np.ndarray:
    """
    Convert a BGR face ROI to a normalised grayscale image ready for LBPH.

    Steps:
        1. Convert to grayscale
        2. Resize to FACE_SIZE
        3. Equalise histogram for lighting invariance
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    equalised = cv2.equalizeHist(resized)
    return equalised


def iter_person_images(base_dir: str = TRAIN_DIR
                       ) -> Generator[tuple[str, np.ndarray], None, None]:
    """
    Walk base_dir and yield (person_id, bgr_image) for every valid image.

    Directory layout expected:
        base_dir/
            <person_id>/
                img1.jpg
                img2.jpg
                ...
    """
    for person_id in sorted(os.listdir(base_dir)):
        person_dir = os.path.join(base_dir, person_id)
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _IMG_EXTS:
                continue
            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Could not read image: %s", img_path)
                continue
            yield person_id, img


def build_training_data(face_detector,
                        base_dir: str = TRAIN_DIR
                        ) -> tuple[list[np.ndarray], list[int], dict[int, str]]:
    """
    Build (faces, labels, label_map) suitable for cv2.face.LBPHFaceRecognizer.

    Parameters
    ----------
    face_detector : FaceDetector
        An initialised FaceDetector whose ``detect_faces`` method is called.

    Returns
    -------
    faces      : list of grayscale np.ndarray
    labels     : list of int (numeric label)
    label_map  : dict  int → person_id
    """
    faces: list[np.ndarray] = []
    labels: list[int] = []
    label_map: dict[int, str] = {}
    person_to_int: dict[str, int] = {}
    counter = 0

    for person_id, img in iter_person_images(base_dir):
        detected = face_detector.detect_faces(img)
        if not detected:
            logger.debug("No face found in image for %s — skipping.", person_id)
            continue

        # Use the largest face detected
        (x, y, w, h) = max(detected, key=lambda r: r[2] * r[3])
        face_roi = img[y:y + h, x:x + w]
        proc = preprocess_face(face_roi)

        if person_id not in person_to_int:
            person_to_int[person_id] = counter
            label_map[counter] = person_id
            counter += 1

        faces.append(proc)
        labels.append(person_to_int[person_id])

    logger.info(
        "Training data built: %d samples across %d identities.",
        len(faces), len(label_map),
    )
    return faces, labels, label_map
