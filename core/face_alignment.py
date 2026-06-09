"""
face_alignment.py
=================
Face detection and alignment module using MTCNN.

MTCNN (Multi-task Cascaded Convolutional Networks) detects faces and 5 facial
landmarks (left eye, right eye, nose, mouth left, mouth right).

This module uses the eye landmarks to rotate (align) the face so that the eyes
are perfectly horizontal, and then scales/crops the face to a fixed output size.
This is highly recommended before extracting deep features (like MobileNetV2),
as it standardises the input distribution.

Requirements
------------
    pip install mtcnn

Usage
-----
    from face_alignment import FaceAligner

    aligner = FaceAligner(output_size=(224, 224))
    
    # Process a BGR OpenCV frame
    aligned_faces = aligner.align(frame)
    
    for face in aligned_faces:
        box = face["box"]              # original (x, y, w, h)
        crop = face["aligned_crop"]    # perfectly aligned 224x224 BGR image
        conf = face["confidence"]      # MTCNN confidence score
        
        # e.g. pass 'crop' directly to EmbeddingExtractor ...

Usage (CLI)
-----------
    python face_alignment.py --input test.jpg --show
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

import cv2
import numpy as np

from config import LOG_LEVEL

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


class FaceAligner:
    """
    Detects faces and aligns them based on MTCNN eye landmarks.

    Parameters
    ----------
    output_size : tuple[int, int]
        Desired (width, height) of the output aligned face crops.
    left_eye_pct : tuple[float, float]
        Where the left eye should be placed in the final crop, as a percentage
        of width and height. Default (0.35, 0.35) means 35% from the left edge 
        and 35% from the top edge. The right eye is automatically placed symmetrically.
    min_confidence : float
        Minimum MTCNN detection confidence score to keep a face.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = (224, 224),
        left_eye_pct: tuple[float, float] = (0.35, 0.35),
        min_confidence: float = 0.90,
    ) -> None:
        self.output_size = output_size
        self.left_eye_pct = left_eye_pct
        self.min_confidence = min_confidence

        try:
            from mtcnn import MTCNN
            logger.info("Initialising MTCNN detector … (may take a moment to load TF/Keras)")
            # Suppress excessive TF logging if possible
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            self.detector = MTCNN()
            logger.info("MTCNN detector ready.")
        except ImportError:
            logger.error(
                "MTCNN is not installed. Please install it to use face alignment:\n"
                "  pip install mtcnn\n"
            )
            raise

    def align(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in *image_bgr*, compute the affine transform to align the eyes,
        and yield cropped/normalised faces.

        Parameters
        ----------
        image_bgr : np.ndarray
            Original OpenCV BGR frame.

        Returns
        -------
        list of dict:
            [
                {
                    "box": (x, y, w, h),
                    "confidence": float,
                    "keypoints": dict,
                    "aligned_crop": np.ndarray (output_size[1] x output_size[0] BGR image)
                }, ...
            ]
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        # MTCNN expects RGB images
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # detect_faces returns [{'box': [x,y,w,h], 'confidence': 0.9, 'keypoints': {...}}, ...]
        results = self.detector.detect_faces(image_rgb)
        
        aligned_results = []
        for res in results:
            if res["confidence"] < self.min_confidence:
                continue

            # In MTCNN:
            # left_eye is on the left side of the image (the person's right eye anatomically)
            keypoints = res["keypoints"]
            left_eye = keypoints["left_eye"]
            right_eye = keypoints["right_eye"]
            
            # MTCNN bounding boxes can occasionally go out of bounds (negative)
            bx, by, bw, bh = res["box"]
            bx, by = max(0, bx), max(0, by)
            clean_box = (bx, by, bw, bh)

            # ── 1. Calculate rotation angle ──────────────────────────────────
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            
            # If dy > 0, the right eye is lower than the left.
            # arctan2(dy, dx) will be positive.
            # cv2.getRotationMatrix2D uses positive angles for counter-clockwise rotation,
            # which will lift the right eye up, making them level.
            angle = np.degrees(np.arctan2(dy, dx))

            # ── 2. Calculate scaling factor ──────────────────────────────────
            # Distance between eyes in the original image
            dist_original = np.sqrt(dx**2 + dy**2)
            
            # Desired distance between eyes in the output crop
            # e.g., if left eye is at 35% width, right eye must be at 65% width
            desired_right_eye_x = 1.0 - self.left_eye_pct[0]
            desired_dist = (desired_right_eye_x - self.left_eye_pct[0]) * self.output_size[0]
            
            # Scale to match the desired eye distance
            scale = desired_dist / max(1.0, dist_original)

            # ── 3. Calculate translation ─────────────────────────────────────
            # Center point between eyes in original image
            eyes_center = (
                float(left_eye[0] + right_eye[0]) / 2.0,
                float(left_eye[1] + right_eye[1]) / 2.0
            )
            
            # Get base rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
            
            # Adjust the translation component of the matrix (M[0,2] and M[1,2])
            # so that the eyes_center maps perfectly to the desired center of the output.
            target_center_x = self.output_size[0] * 0.5
            target_center_y = self.output_size[1] * self.left_eye_pct[1]
            
            M[0, 2] += (target_center_x - eyes_center[0])
            M[1, 2] += (target_center_y - eyes_center[1])

            # ── 4. Apply Affine Transform ────────────────────────────────────
            # Warp the original BGR image to the target size
            aligned_crop = cv2.warpAffine(
                image_bgr,
                M,
                self.output_size,
                flags=cv2.INTER_CUBIC
            )

            aligned_results.append({
                "box": clean_box,
                "confidence": res["confidence"],
                "keypoints": keypoints,
                "aligned_crop": aligned_crop
            })
            
        logger.debug("MTCNN found and aligned %d face(s).", len(aligned_results))
        return aligned_results


# ── CLI for testing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MTCNN Face Alignment.")
    parser.add_argument("--input", required=True, help="Path to input image.")
    parser.add_argument("--show", action="store_true", help="Display aligned outputs.")
    parser.add_argument("--size", type=int, default=224, help="Output crop size (default 224).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    
    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    try:
        aligner = FaceAligner(output_size=(args.size, args.size))
    except ImportError:
        sys.exit(1)

    frame = cv2.imread(args.input)
    if frame is None:
        logger.error("Cannot read image.")
        sys.exit(1)

    print(f"\nProcessing '{args.input}' ...")
    results = aligner.align(frame)

    if not results:
        print("No faces detected.")
        sys.exit(0)

    print(f"\nFound {len(results)} face(s).")
    for i, res in enumerate(results, 1):
        box = res["box"]
        conf = res["confidence"]
        print(f"  Face {i} | Box {box} | Conf {conf:.3f}")
        
        if args.show:
            crop = res["aligned_crop"]
            # Draw original bounding box on the original frame
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {i}", (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show the separated aligned crop
            cv2.imshow(f"Aligned Face {i} ({args.size}x{args.size})", crop)

    if args.show:
        # Show original frame with boxes
        cv2.imshow("Original with MTCNN boxes", frame)
        print("\nPress any key on an image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
