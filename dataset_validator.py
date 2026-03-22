"""
dataset_validator.py
====================
Validates the integrity, structure, and readability of the dataset directory.

Requirements met
----------------
1. Verify dataset/train and dataset/test structure.
2. Check that each person folder contains images.
3. Detect corrupted images.
4. Generate dataset statistics.

Usage
-----
    from dataset_validator import DatasetValidator
    
    validator = DatasetValidator()
    report = validator.validate()
    
    if report["is_valid"]:
        print("Dataset is healthy!")

Usage (CLI)
-----------
    python dataset_validator.py
"""

import os
import glob
import logging
from typing import Dict, Any, List

import cv2

from config import DATASET_DIR, TRAIN_DIR, TEST_DIR, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Valid image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

class DatasetValidator:
    """
    Scans the dataset directory to ensure structural integrity and image readability.
    """

    def __init__(self, dataset_root: str = DATASET_DIR) -> None:
        self.dataset_root = dataset_root
        self.train_dir = os.path.join(dataset_root, "train")
        self.test_dir = os.path.join(dataset_root, "test")

    def validate(self) -> Dict[str, Any]:
        """
        Run the full validation suite.
        
        Returns
        -------
        report : dict
            A dictionary containing boolean validity flags, statistics, and error messages.
        """
        logger.info("Starting dataset validation on '%s' ...", self.dataset_root)
        
        report: Dict[str, Any] = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "train": {"identities": 0, "images": 0},
                "test": {"identities": 0, "images": 0},
                "total_images": 0,
                "corrupted_images": 0
            }
        }
        
        # 1. Structure Check
        self._check_structure(report)
        
        # 2 & 3. Folder content and Image integrity Check
        if os.path.exists(self.train_dir):
            self._scan_directory("train", self.train_dir, report)
        
        if os.path.exists(self.test_dir):
            self._scan_directory("test", self.test_dir, report)

        # 4. Final summary checks
        if report["stats"]["train"]["identities"] == 0:
            self._add_error("Training directory contains no valid identities/folders.", report)

        if report["stats"]["corrupted_images"] > 0:
            self._add_error(f"Found {report['stats']['corrupted_images']} corrupted/unreadable images.", report)
            
        logger.info("Validation complete. Valid: %s | Errors: %d | Warnings: %d", 
                    report["is_valid"], len(report["errors"]), len(report["warnings"]))
        
        return report

    def _add_error(self, msg: str, report: dict) -> None:
        logger.error(msg)
        report["errors"].append(msg)
        report["is_valid"] = False
        
    def _add_warning(self, msg: str, report: dict) -> None:
        logger.warning(msg)
        report["warnings"].append(msg)

    def _check_structure(self, report: dict) -> None:
        """Verify the primary directories exist."""
        if not os.path.exists(self.dataset_root):
            self._add_error(f"Dataset root directory missing: {self.dataset_root}", report)
            return

        for name, path in [("Train", self.train_dir), ("Test", self.test_dir)]:
            if not os.path.exists(path):
                self._add_warning(f"{name} directory missing at {path}. System may fail to train/test.", report)

    def _scan_directory(self, split: str, root_path: str, report: dict) -> None:
        """Scan through a split directory (train or test), validating subfolders and images."""
        person_folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
        
        report["stats"][split]["identities"] = len(person_folders)
        
        if not person_folders:
            self._add_warning(f"No person folders found in {split} directory.", report)
            return
            
        for person_id in person_folders:
            folder_path = os.path.join(root_path, person_id)
            
            # Grab all files
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            
            # Filter to image extensions
            img_files = [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
            
            if not img_files:
                self._add_error(f"Empty person folder (no valid images): {split}/{person_id}", report)
                continue
                
            valid_img_count = 0
            
            # Test readability
            for img_name in img_files:
                img_path = os.path.join(folder_path, img_name)
                
                # Try opening with OpenCV
                frame = cv2.imread(img_path)
                
                if frame is None or frame.size == 0:
                    self._add_error(f"Corrupted or unreadable image: {img_path}", report)
                    report["stats"]["corrupted_images"] += 1
                else:
                    valid_img_count += 1
                    
            report["stats"][split]["images"] += valid_img_count
            report["stats"]["total_images"] += valid_img_count

def display_report(report: dict) -> None:
    """Pretty-print the validation report to the terminal."""
    print()
    print("═" * 50)
    print("  DATASET VALIDATION REPORT")
    print("═" * 50)
    
    status = "✅ PASSED" if report["is_valid"] else "❌ FAILED"
    print(f"  Overall Status : {status}")
    print("─" * 50)
    print("  STATISTICS")
    
    s = report["stats"]
    print(f"    Train Identities : {s['train']['identities']}")
    print(f"    Train Images     : {s['train']['images']}")
    print(f"    Test Identities  : {s['test']['identities']}")
    print(f"    Test Images      : {s['test']['images']}")
    print(f"    Total Read OK    : {s['total_images']}")
    print(f"    Corrupt Images   : {s['corrupted_images']}")
    
    if report["warnings"]:
        print("─" * 50)
        print("  WARNINGS")
        for w in report["warnings"]:
            print(f"    ! {w}")
            
    if report["errors"]:
        print("─" * 50)
        print("  ERRORS")
        for e in report["errors"]:
            print(f"    ✖ {e}")
            
    print("═" * 50)
    print()

if __name__ == "__main__":
    validator = DatasetValidator()
    report = validator.validate()
    display_report(report)
