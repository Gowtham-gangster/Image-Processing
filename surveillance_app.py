"""
surveillance_app.py
===================
Main entry point for the Mask-Aware Hybrid Person Identification System.

Supports three input modes (controlled via CLI arguments):
  --source image   <path>  – Analyse a single image file.
  --source video   <path>  – Analyse a video file.
  --source camera  <id>    – Live camera feed (default).

Usage examples
--------------
  python surveillance_app.py                             # webcam (default)
  python surveillance_app.py --source image input.jpg   # single image
  python surveillance_app.py --source video cctv.mp4    # video file
  python surveillance_app.py --source camera 0          # explicit webcam
  python surveillance_app.py --no-display               # headless mode
"""

import argparse
import logging
import sys
import time

import cv2

from config import CAMERA_SOURCE, LOG_LEVEL, LOGS_DIR, LOG_FILE
from database import PersonDatabase
from face_detector import FaceDetector
from mask_detector import MaskDetector
from person_identifier import PersonIdentifier
from utils import (
    draw_face_box, draw_status_bar, draw_attributes_panel, log_detection
)

# ── Logging ───────────────────────────────────────────────────────────────────
import os
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Main pipeline ─────────────────────────────────────────────────────────────

class SurveillanceSystem:
    """
    Orchestrates face detection → mask detection → identification → display.
    """

    def __init__(self):
        logger.info("Initialising Mask-Aware Hybrid Person Identification System …")
        self.db = PersonDatabase()
        self.detector = FaceDetector()
        self.mask_det = MaskDetector()
        self.identifier = PersonIdentifier(db=self.db)
        logger.info(
            "System ready | Detector: %s | Mask: %s | Recognizer: %s",
            self.detector.mode,
            self.mask_det.mode,
            "loaded" if self.identifier.is_ready() else "NOT loaded (run model_trainer.py)",
        )

    # ── Per-frame processing ──────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Run the full pipeline on a single BGR frame.
        Returns the annotated frame and a list of IdentificationResults.
        """
        boxes = self.detector.detect_faces(frame)
        results = []
        last_known_result = None

        for box in boxes:
            x, y, w, h = box
            # Guard against zero-size boxes
            if w <= 0 or h <= 0:
                continue

            face_roi = frame[y: y + h, x: x + w]
            if face_roi.size == 0:
                continue

            # Mask detection
            is_masked, mask_conf = self.mask_det.is_masked(face_roi)
            logger.debug("Mask: %s (conf=%.2f)", is_masked, mask_conf)

            # Identification
            result = self.identifier.identify(face_roi, is_masked=is_masked)
            results.append(result)

            # Annotate frame
            draw_face_box(frame, box, result)

            if result.is_known:
                last_known_result = result

            # Log to CSV
            log_detection(result)

        # Show attribute panel for last identified known person
        if last_known_result is not None:
            draw_attributes_panel(frame, last_known_result)

        # Status bar
        draw_status_bar(
            frame,
            self.detector.mode,
            self.mask_det.mode,
            len(boxes),
        )

        return frame, results

    # ── Input modes ───────────────────────────────────────────────────────────

    def run_image(self, path: str, display: bool = True) -> None:
        """Process a single image file."""
        frame = cv2.imread(path)
        if frame is None:
            logger.error("Could not read image: %s", path)
            return

        annotated, results = self.process_frame(frame)
        self._print_results(results)

        if display:
            cv2.imshow("Mask-Aware Identification – Image", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            out_path = path.replace(".", "_result.", 1)
            cv2.imwrite(out_path, annotated)
            logger.info("Saved annotated image → %s", out_path)

    def run_video(self, path: str, display: bool = True) -> None:
        """Process a video file frame-by-frame."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.error("Could not open video: %s", path)
            return
        self._run_capture(cap, window_title="Video", display=display)
        cap.release()

    def run_camera(self, source=CAMERA_SOURCE, display: bool = True) -> None:
        """Live camera / webcam feed."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Could not open camera source: %s", source)
            return
        logger.info("Camera stream started. Press 'q' to quit.")
        self._run_capture(cap, window_title="CCTV – Live", display=display)
        cap.release()

    def _run_capture(self, cap: cv2.VideoCapture,
                     window_title: str = "Stream",
                     display: bool = True) -> None:
        """Shared loop for video and camera sources."""
        fps_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Stream ended.")
                break

            frame_count += 1
            annotated, _ = self.process_frame(frame)

            # FPS overlay
            elapsed = time.time() - fps_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0] - 34),
                        0, 0.48, (100, 220, 100), 1, cv2.LINE_AA)

            if display:
                cv2.imshow(window_title, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User pressed 'q'. Exiting.")
                    break

        if display:
            cv2.destroyAllWindows()

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_results(results) -> None:
        print("\n── Detection Results ──────────────────────────────────────────")
        if not results:
            print("  No faces detected.")
        for i, r in enumerate(results, 1):
            print(f"  Face {i}: {r}")
        print("────────────────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Mask-Aware Hybrid Person Identification System"
    )
    parser.add_argument(
        "--source", choices=["image", "video", "camera"], default="camera",
        help="Input source type (default: camera)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to image/video file (required for image/video modes)",
    )
    parser.add_argument(
        "--camera-id", type=int, default=CAMERA_SOURCE,
        help=f"Camera device index (default: {CAMERA_SOURCE})",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run in headless mode (save output instead of displaying window)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    system = SurveillanceSystem()
    display = not args.no_display

    if args.source == "image":
        if not args.input:
            print("ERROR: --input <path> is required for image mode.")
            sys.exit(1)
        system.run_image(args.input, display=display)

    elif args.source == "video":
        if not args.input:
            print("ERROR: --input <path> is required for video mode.")
            sys.exit(1)
        system.run_video(args.input, display=display)

    else:  # camera
        system.run_camera(source=args.camera_id, display=display)
