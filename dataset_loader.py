"""
dataset_loader.py
=================
Strictly enforces dataset boundaries during training and recognition as per requirements.
1. ONLY dataset/train folder is used for training.
2. dataset/test folder is used ONLY for testing/recognition.
3. dataset/unknown folder must NOT be used in training.
"""

import os
import cv2
import logging
from config import DATASET_DIR

logger = logging.getLogger(__name__)

# Strict Directory Definitions
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
UNKNOWN_DIR = os.path.join(DATASET_DIR, "unknown")

class DatasetLoader:
    def __init__(self):
        self._ensure_directories()

    def _ensure_directories(self):
        """Creates the required folders if they don't exist."""
        os.makedirs(TRAIN_DIR, exist_ok=True)
        os.makedirs(TEST_DIR, exist_ok=True)
        os.makedirs(UNKNOWN_DIR, exist_ok=True)

    def load_training_data(self):
        """
        Yields (person_id, image_path, BGR_image) ONLY from dataset/train.
        Strictly excludes 'test' and 'unknown' directories.
        """
        logger.info("Loading training data exclusively from: %s", TRAIN_DIR)
        return self._load_from_folder(TRAIN_DIR)

    def load_testing_data(self):
        """
        Yields (person_id, image_path, BGR_image) ONLY from dataset/test.
        """
        logger.info("Loading testing data exclusively from: %s", TEST_DIR)
        return self._load_from_folder(TEST_DIR)

    def load_unknown_data(self):
        """
        Yields ('Unknown', image_path, BGR_image) ONLY from dataset/unknown.
        """
        logger.info("Loading unknown evaluation data exclusively from: %s", UNKNOWN_DIR)
        for fname in os.listdir(UNKNOWN_DIR):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(UNKNOWN_DIR, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    yield "Unknown", img_path, img

    def _load_from_folder(self, base_folder):
        for person_id in os.listdir(base_folder):
            person_dir = os.path.join(base_folder, person_id)
            if not os.path.isdir(person_dir):
                continue
                
            # Extra safety check to prevent directory leakage
            if person_id.lower() in ['test', 'unknown', 'train']:
                logger.warning(f"Skipping nested system directory: {person_dir}")
                continue

            for fname in os.listdir(person_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_dir, fname)
                    img = cv2.imread(img_path)
                    if img is not None:
                        yield person_id, img_path, img
