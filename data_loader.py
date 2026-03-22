"""
data_loader.py
==============
Dataset loading, label assignment, and preprocessing for the
Mask-Aware Hybrid Person Identification System.

Responsibilities
----------------
1. Walk dataset/train/ and dataset/test/ — one sub-folder per identity.
2. Assign integer labels from folder names and maintain a bidirectional
   label ↔ person_id mapping.
3. Pre-process every image:
     a. Convert to grayscale (for LBPH / HOG) or keep BGR (for CNN).
     b. Resize to a fixed target size.
     c. Normalise pixel values to [0, 1] (float32).
4. Return NumPy arrays ready for model training / evaluation.
5. Provide a DataLoader class for programmatic use and a
   stand-alone CLI for quick sanity-checks.

Usage (module)
--------------
    from data_loader import DataLoader

    dl = DataLoader()
    X_train, y_train, label_map = dl.load_train()
    X_test,  y_test,  _         = dl.load_test(label_map=label_map)

Usage (CLI)
-----------
    python data_loader.py                    # summarise train + test splits
    python data_loader.py --split train      # only train split
    python data_loader.py --split test       # only test split
    python data_loader.py --vis              # show sample images (requires display)
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import TRAIN_DIR, TEST_DIR, FACE_SIZE, LOG_LEVEL

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# Supported image extensions
_SUPPORTED_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DatasetInfo:
    """Summary statistics for a loaded dataset split."""
    split: str                          # 'train' or 'test'
    num_samples: int = 0
    num_classes: int = 0
    image_shape: tuple = ()            # (H, W) or (H, W, C)
    label_map: dict[int, str] = field(default_factory=dict)   # int → person_id
    class_counts: dict[str, int] = field(default_factory=dict)  # person_id → count

    def __str__(self) -> str:
        lines = [
            f"── DatasetInfo ({self.split}) ────────────────────────────────",
            f"  Samples     : {self.num_samples}",
            f"  Classes     : {self.num_classes}",
            f"  Image shape : {self.image_shape}",
            f"  Per-class   :",
        ]
        for pid, cnt in sorted(self.class_counts.items()):
            lines.append(f"    {pid:<20s} → {cnt} images")
        lines.append("─" * 60)
        return "\n".join(lines)


# ── DataLoader ────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads and preprocesses images from the dataset directory structure.

    Parameters
    ----------
    train_dir : str
        Root directory for training images.
    test_dir : str
        Root directory for testing images.
    image_size : tuple[int, int]
        Target (width, height) for resizing.
    grayscale : bool
        If True, convert images to single-channel grayscale and normalise to
        [0, 1]. If False, keep BGR (3-channel) and normalise to [0, 1].
    """

    def __init__(
        self,
        train_dir: str = TRAIN_DIR,
        test_dir: str = TEST_DIR,
        image_size: tuple[int, int] = FACE_SIZE,
        grayscale: bool = True,
    ) -> None:
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_size = image_size   # (W, H) — OpenCV convention
        self.grayscale = grayscale

    # ── Public API ────────────────────────────────────────────────────────────

    def load_train(
        self,
        label_map: Optional[dict[int, str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
        """
        Load the training split.

        Parameters
        ----------
        label_map : dict, optional
            Existing int→person_id mapping. Pass ``None`` to build a new one
            from the folder names found in ``train_dir``.

        Returns
        -------
        X : np.ndarray  shape (N, H, W) or (N, H, W, 3)  dtype float32
        y : np.ndarray  shape (N,)  dtype int32
        label_map : dict[int, str]  int → person_id
        """
        return self._load_split(self.train_dir, split_name="train", label_map=label_map)

    def load_test(
        self,
        label_map: Optional[dict[int, str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
        """
        Load the test split.

        Parameters
        ----------
        label_map : dict, optional
            Pass the label_map returned by :meth:`load_train` so that numeric
            labels are consistent between splits.

        Returns
        -------
        X : np.ndarray  shape (N, H, W) or (N, H, W, 3)  dtype float32
        y : np.ndarray  shape (N,)  dtype int32
        label_map : dict[int, str]
        """
        return self._load_split(self.test_dir, split_name="test", label_map=label_map)

    def load_both(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        dict[int, str],
    ]:
        """
        Convenience: load train and test splits sharing one label_map.

        Returns
        -------
        X_train, y_train, X_test, y_test, label_map
        """
        X_train, y_train, label_map = self.load_train()
        X_test,  y_test,  _         = self.load_test(label_map=label_map)
        return X_train, y_train, X_test, y_test, label_map

    def dataset_info(self, split: str = "train") -> DatasetInfo:
        """
        Return a :class:`DatasetInfo` summary without loading pixel data.

        Parameters
        ----------
        split : str
            ``'train'`` or ``'test'``.
        """
        base = self.train_dir if split == "train" else self.test_dir
        return self._collect_info(base, split)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_split(
        self,
        base_dir: str,
        split_name: str,
        label_map: Optional[dict[int, str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
        """
        Core loading routine shared by :meth:`load_train` / :meth:`load_test`.
        """
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: '{base_dir}'"
            )

        # Build or validate label map
        person_ids = self._discover_persons(base_dir)
        if not person_ids:
            raise ValueError(
                f"No person sub-folders found in '{base_dir}'. "
                "Each sub-folder should contain images for one identity."
            )

        if label_map is None:
            # Build a fresh mapping: sort for reproducibility
            label_map = {idx: pid for idx, pid in enumerate(sorted(person_ids))}

        # Reverse map: person_id → int
        pid_to_int: dict[str, int] = {v: k for k, v in label_map.items()}

        images: list[np.ndarray] = []
        labels: list[int] = []
        skipped = 0

        for pid in sorted(person_ids):
            person_dir = os.path.join(base_dir, pid)
            img_paths = self._list_images(person_dir)

            if not img_paths:
                logger.warning("No valid images in '%s' — skipping.", person_dir)
                continue

            if pid not in pid_to_int:
                logger.warning(
                    "Person '%s' found in %s but not in label_map — skipping.",
                    pid, split_name,
                )
                continue

            label = pid_to_int[pid]

            for img_path in img_paths:
                img = self._read_and_preprocess(img_path)
                if img is None:
                    skipped += 1
                    continue
                images.append(img)
                labels.append(label)

        if not images:
            logger.error(
                "Zero images loaded from '%s'. "
                "Check that sub-folders contain supported image files (%s).",
                base_dir, ", ".join(_SUPPORTED_EXTS),
            )
            # Return empty arrays with proper dtype
            h, w = self.image_size[1], self.image_size[0]
            shape = (0, h, w) if self.grayscale else (0, h, w, 3)
            return np.empty(shape, dtype=np.float32), np.empty((0,), dtype=np.int32), label_map

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        logger.info(
            "[%s] Loaded %d images | %d classes | shape %s | skipped %d",
            split_name, len(X), len(label_map), X.shape, skipped,
        )
        return X, y, label_map

    def _read_and_preprocess(self, img_path: str) -> Optional[np.ndarray]:
        """
        Read one image from disk and apply:
          1. Colour conversion (grayscale or BGR)
          2. Resize to ``self.image_size``
          3. Histogram equalisation (grayscale only) for lighting invariance
          4. Normalise pixel values to [0, 1] float32
        """
        img: Optional[np.ndarray] = cv2.imread(img_path)
        if img is None:
            logger.warning("Could not read image — skipping: %s", img_path)
            return None

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize  (cv2.resize takes (width, height))
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

        if self.grayscale:
            # Histogram equalisation improves performance under varied lighting
            img = cv2.equalizeHist(img)

        # Normalise to [0.0, 1.0]
        img = img.astype(np.float32) / 255.0

        return img

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _discover_persons(base_dir: str) -> list[str]:
        """Return sorted list of sub-directory names (= person_ids)."""
        return sorted(
            name for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        )

    @staticmethod
    def _list_images(directory: str) -> list[str]:
        """Return full paths of all supported images in *directory*."""
        return sorted(
            os.path.join(directory, fname)
            for fname in os.listdir(directory)
            if Path(fname).suffix.lower() in _SUPPORTED_EXTS
        )

    def _collect_info(self, base_dir: str, split: str) -> DatasetInfo:
        """Collect summary metadata without loading pixel data."""
        info = DatasetInfo(split=split)
        if not os.path.isdir(base_dir):
            return info

        h, w = self.image_size[1], self.image_size[0]
        info.image_shape = (h, w) if self.grayscale else (h, w, 3)

        person_ids = self._discover_persons(base_dir)
        info.num_classes = len(person_ids)
        info.label_map = {i: pid for i, pid in enumerate(sorted(person_ids))}

        for idx, pid in enumerate(sorted(person_ids)):
            count = len(self._list_images(os.path.join(base_dir, pid)))
            info.class_counts[pid] = count
            info.num_samples += count

        return info


# ── Visualisation helper ──────────────────────────────────────────────────────

def visualise_samples(
    X: np.ndarray,
    y: np.ndarray,
    label_map: dict[int, str],
    n_per_class: int = 3,
    window_name: str = "Dataset Samples",
) -> None:
    """
    Display a grid of sample images using OpenCV.

    Parameters
    ----------
    X : np.ndarray  float32 in [0, 1]
    y : np.ndarray  int labels
    label_map : dict[int, str]
    n_per_class : int
        Number of sample images to show per identity.
    """
    classes = sorted(label_map.keys())
    rows = []

    for cls in classes:
        indices = np.where(y == cls)[0]
        if len(indices) == 0:
            continue
        picks = indices[:n_per_class]
        imgs = []
        for idx in picks:
            img = (X[idx] * 255).astype(np.uint8)
            # Convert grayscale → BGR for display grid
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Add person label
            cv2.putText(img, label_map[cls], (2, 12), 0, 0.35,
                        (0, 255, 0), 1, cv2.LINE_AA)
            imgs.append(img)
        # Pad row to n_per_class with blank images
        h, w = imgs[0].shape[:2]
        while len(imgs) < n_per_class:
            imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.hstack(imgs))

    if not rows:
        logger.warning("No images to display.")
        return

    # Pad all rows to the same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    grid = np.vstack(padded)
    cv2.imshow(window_name, grid)
    logger.info("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataLoader CLI – summarise or visualise dataset splits."
    )
    parser.add_argument(
        "--split", choices=["train", "test", "both"], default="both",
        help="Which split to process (default: both).",
    )
    parser.add_argument(
        "--grayscale", action="store_true", default=True,
        help="Load images as grayscale (default: True).",
    )
    parser.add_argument(
        "--color", dest="grayscale", action="store_false",
        help="Load images in BGR colour mode.",
    )
    parser.add_argument(
        "--size", nargs=2, type=int, default=list(FACE_SIZE),
        metavar=("W", "H"),
        help=f"Target image size (default: {FACE_SIZE[0]} {FACE_SIZE[1]}).",
    )
    parser.add_argument(
        "--vis", action="store_true",
        help="Show sample images in an OpenCV window.",
    )
    parser.add_argument(
        "--n-vis", type=int, default=3, metavar="N",
        help="Number of sample images to show per person (default: 3).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    target_size = tuple(args.size)  # (W, H)

    loader = DataLoader(grayscale=args.grayscale, image_size=target_size)

    splits_to_run = (
        ["train", "test"] if args.split == "both"
        else [args.split]
    )

    label_map: dict[int, str] = {}

    for split in splits_to_run:
        # Print info summary (no pixel loading)
        info = loader.dataset_info(split=split)
        print(info)

        # Load pixel data
        if split == "train":
            X, y, label_map = loader.load_train()
        else:
            X, y, _ = loader.load_test(label_map=label_map if label_map else None)

        print(f"  X_{split} dtype  : {X.dtype}")
        print(f"  X_{split} shape  : {X.shape}")
        print(f"  X_{split} range  : [{X.min():.3f}, {X.max():.3f}]")
        print(f"  y_{split} unique : {np.unique(y).tolist()}\n")

        if args.vis and len(X) > 0:
            visualise_samples(
                X, y, label_map,
                n_per_class=args.n_vis,
                window_name=f"Samples – {split}",
            )
