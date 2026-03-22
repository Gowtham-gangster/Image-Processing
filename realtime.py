"""
realtime.py
===========
Real-time person identification from webcam or CCTV stream.

For every detected face the overlay shows:
  Name · Gender · Age · Phone · Address · Confidence score

Controls
--------
  q   – quit
  p   – pause / resume
  s   – save current frame to logs/snapshots/
  +/- – raise / lower recognition thresholds on the fly

Usage
-----
    python realtime.py                          # default webcam (index 0)
    python realtime.py --source 1              # second camera
    python realtime.py --source cctv.mp4       # video file
    python realtime.py --source rtsp://...     # RTSP / IP camera
    python realtime.py --clf-thresh 0.7        # stricter confidence gate
    python realtime.py --width 1280 --height 720
    python realtime.py --no-attr              # hide attribute panel
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time

import cv2
import numpy as np

from config import CAMERA_SOURCE, LOG_LEVEL, LOGS_DIR, COLOUR_KNOWN, COLOUR_UNKNOWN, COLOUR_MASKED, FONT
from recognition import RecognitionResult
from detection import Detector
from face_alignment import FaceAligner
from embedding_model import FaceNetEmbedder
from embedding_database import EmbeddingDatabase
from attributes_manager import AttributesManager
from unknown_detector import UnknownDetector
from person_tracker import PersonTracker
from liveness_detector import LivenessDetector
from alert_manager import AlertManager, ALERT_SPOOF

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

SNAPSHOT_DIR = os.path.join(LOGS_DIR, "snapshots")


# ── Overlay drawing ───────────────────────────────────────────────────────────

def _draw_face_overlay(
    frame: np.ndarray,
    result: RecognitionResult,
    show_attrs: bool = True,
) -> None:
    """
    Draw bounding box + inline label + side attribute panel for one result.
    All drawing is done in-place on *frame*.
    """
    x, y, w, h = result.bbox
    x2, y2 = x + w, y + h

    # ── Colour ────────────────────────────────────────────────────────────────
    colour = COLOUR_KNOWN if result.is_known else COLOUR_UNKNOWN

    # ── Bounding box ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x, y), (x2, y2), colour, 2)

    # ── Inline name tag above the box ─────────────────────────────────────────
    label = result.name if result.is_known else "Unknown Person"
    conf  = f"Conf: {result.clf_confidence:.0%}"

    tag_lines = [label, conf]
    tag_h     = 20
    tag_top   = max(y - len(tag_lines) * tag_h - 4, 0)
    overlay   = frame.copy()
    cv2.rectangle(overlay, (x, tag_top), (x2, y), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(tag_lines):
        cv2.putText(frame, line,
                    (x + 4, tag_top + (i + 1) * tag_h - 2),
                    FONT, 0.50, colour, 1, cv2.LINE_AA)

    # ── Attribute panel (right of bounding box) ────────────────────────────────
    if show_attrs and result.is_known:
        attrs = result.attributes
        lines = [
            f"Name    : {attrs.get('name',    'N/A')}",
            f"Gender  : {attrs.get('gender',  'N/A')}",
            f"Age     : {attrs.get('age',     'N/A')}",
            f"Phone   : {attrs.get('phone',   'N/A')}",
            f"Address : {attrs.get('address', 'N/A')}",
            f"Conf    : {result.clf_confidence:.1%}",
            f"Sim     : {result.embed_similarity:.1%}",
        ]
        _draw_attr_panel(frame, lines, anchor=(x, y), box_w=w, colour=colour)


def _draw_attr_panel(
    frame: np.ndarray,
    lines: list[str],
    anchor: tuple[int, int],
    box_w: int,
    colour: tuple[int, int, int],
    panel_width: int = 280,
    line_height: int = 20,
) -> None:
    """Draw a dark-background attribute panel to the right of the face box."""
    img_h, img_w = frame.shape[:2]
    ax, ay       = anchor
    panel_h      = len(lines) * line_height + 10

    # Prefer right of box; fall back to left if out of bounds
    px = ax + box_w + 8
    if px + panel_width > img_w:
        px = max(0, ax - panel_width - 8)

    py = ay
    if py + panel_h > img_h:
        py = max(0, img_h - panel_h)

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_width, py + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_width, py + panel_h), colour, 1)

    # Text
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (px + 8, py + (i + 1) * line_height),
            FONT, 0.44, (220, 220, 220), 1, cv2.LINE_AA,
        )


def _draw_hud(
    frame: np.ndarray,
    n_faces: int,
    fps: float,
    paused: bool,
    clf_thresh: float,
    sim_thresh: float,
    det_mode: str,
) -> None:
    """Draw the heads-up status bar at the bottom of the frame."""
    img_h, img_w = frame.shape[:2]
    bar_h = 28

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, img_h - bar_h), (img_w, img_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    ts     = datetime.datetime.now().strftime("%H:%M:%S")
    status = "PAUSED" if paused else "LIVE"
    text   = (
        f"  {ts}  |  {status}  |  FPS: {fps:.1f}  |  Faces: {n_faces}"
        f"  |  Detector: {det_mode.upper()}"
        f"  |  Clf≥{clf_thresh:.0%}  Sim≥{sim_thresh:.0%}"
        f"  |  [q] quit  [p] pause  [s] snap  [+/-] thresh"
    )
    cv2.putText(frame, text,
                (4, img_h - 8), FONT, 0.40, (160, 200, 160), 1, cv2.LINE_AA)


# ── Snapshot helper ───────────────────────────────────────────────────────────

def _save_snapshot(frame: np.ndarray) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SNAPSHOT_DIR, f"snap_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_realtime(
    source=CAMERA_SOURCE,
    clf_threshold: float = 0.55,
    sim_threshold: float = 0.50,
    dl_backend: str = "auto",
    det_backend: str = "auto",
    frame_width: int = 0,
    frame_height: int = 0,
    show_attrs: bool = True,
    detect_upper_body: bool = False,
) -> None:
    """
    Main real-time recognition loop.

    Parameters
    ----------
    source           : Camera index (int) or path/URL string.
    clf_threshold    : Min classifier confidence gate.
    sim_threshold    : Min cosine similarity gate.
    dl_backend       : 'auto', 'tf', or 'torch'.
    det_backend      : 'auto', 'dnn', or 'haar'.
    frame_width/height: Requested capture resolution (0 = camera default).
    show_attrs       : Show/hide attribute panel.
    detect_upper_body: Fallback to upper-body detection.
    """

    # ── Load system ───────────────────────────────────────────────────────────
    logger.info("Initialising recognition system …")
    detector = Detector(backend=det_backend)
    aligner = FaceAligner()
    embedder = FaceNetEmbedder()
    db = EmbeddingDatabase()
    attrs = AttributesManager()
    unknown_det = UnknownDetector(threshold=sim_threshold)
    tracker = PersonTracker(max_age=30)
    liveness_det = LivenessDetector()
    alert_mgr = AlertManager()

    # ── Open capture ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        sys.exit(1)

    if frame_width  > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_width)
    if frame_height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Stream opened: %dx%d  source=%s", actual_w, actual_h, source)
    print("\n  Real-time recognition started.")
    print("  Controls: [q] quit  [p] pause/resume  [s] snapshot  [+/-] thresholds\n")

    # ── State ─────────────────────────────────────────────────────────────────
    paused      = False
    fps         = 0.0
    frame_count = 0
    last_results: list[RecognitionResult] = []
    t_start     = time.time()
    last_frame: np.ndarray = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                logger.info("Stream ended or read failed.")
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()

        frame_count += 1
        elapsed      = time.time() - t_start
        fps          = frame_count / elapsed if elapsed > 0 else 0.0

        # ── Recognition ───────────────────────────────────────────────────────
        if not paused:
            last_results = []
            
            bboxes = []
            confidences = []
            det_mapping = []
            
            # --- HEURISTIC: Run heavy CNN detection only every 3rd frame ---
            if frame_count % 3 == 0:
                detections = detector.detect(frame)
                for det in detections:
                    bboxes.append(det.bbox)
                    confidences.append(det.confidence)
                    det_mapping.append(det)
            
            # Update tracker (if empty, DeepSORT automatically coasts via Kalman Filter)
            tracks = tracker.update_tracks(bboxes, confidences, frame)
            
            for index, track in enumerate(tracks):
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb(orig=True)
                x, y = int(ltrb[0]), int(ltrb[1])
                w, h = int(ltrb[2] - ltrb[0]), int(ltrb[3] - ltrb[1])
                box = (x, y, w, h)
                
                det_ref = None
                # Naive matching track to original detection crop to get the face crop
                # For DeepSORT we usually map back using original box proximity
                for det in det_mapping:
                    dx, dy, dw, dh = det.bbox
                    cx, cy = x + w/2, y + h/2
                    dcx, dcy = dx + dw/2, dy + dh/2
                    if ((cx-dcx)**2 + (cy-dcy)**2)**0.5 < max(w, h):
                        det_ref = det
                        break
                        
                if det_ref is None:
                    continue # Track active but face detection currently lost
                    
                # Cache hit: Person already identified
                if tracker.has_identity(track_id):
                    cached = tracker.get_identity(track_id)
                    display_name = cached["attributes"].get("name", cached["person_id"]) if cached["attributes"] else "Unknown Person"
                    res = RecognitionResult(
                        person_id=cached["person_id"],
                        name=f"[{track_id}] {display_name}",
                        is_known=(cached["person_id"] != "Unknown Person"),
                        clf_confidence=cached["confidence"],
                        embed_similarity=cached["confidence"],
                        bbox=box,
                        attributes=cached["attributes"] or {},
                        detection=det_ref
                    )
                    last_results.append(res)
                    continue

                # Cache miss: Run heavy recognition
                face_crop = det_ref.face_roi
                aligned_faces = aligner.align(face_crop)
                if not aligned_faces:
                    continue
                    
                best_face = max(aligned_faces, key=lambda f: f["confidence"])
                aligned_crop = best_face["aligned_crop"]
                
                # --- Anti-Spoofing Gate ---
                is_live, spoof_msg = liveness_det.check(aligned_crop)
                if not is_live:
                    # Trigger Alert
                    alert_mgr.send_alert(
                        ALERT_SPOOF, 
                        camera_id=str(args.source), 
                        person_id="SPOOF DETECTED",
                        confidence=0.0,
                        extra={"reason": spoof_msg}
                    )
                    
                    res = RecognitionResult(
                        person_id="SPOOF DETECTED",
                        name=f"[{track_id}] {spoof_msg}",
                        is_known=False,
                        clf_confidence=0.0,
                        embed_similarity=0.0,
                        bbox=box,
                        attributes={"_masked": "N/A", "_strategy": "Spoof"},
                        detection=det_ref
                    )
                    last_results.append(res)
                    continue

                try:
                    emb = embedder.extract(aligned_crop)
                    person_id, score = unknown_det.identify(emb, db)
                    person_attrs = attrs.get_attributes(person_id) if person_id != "Unknown Person" else {}
                    
                    # Update cache
                    tracker.assign_identity(track_id, person_id, score, person_attrs)
                    
                    display_name = person_attrs.get("name", "Unknown Person") if person_attrs else "Unknown Person"
                    res = RecognitionResult(
                        person_id=person_id,
                        name=f"[{track_id}] {display_name}",
                        is_known=(person_id != "Unknown Person"),
                        clf_confidence=score,
                        embed_similarity=score,
                        bbox=box,
                        attributes=person_attrs or {},
                        detection=det_ref
                    )
                    last_results.append(res)
                except Exception as e:
                    logger.error("Recognition failed: %s", e)
                    continue

        # ── Draw overlays ─────────────────────────────────────────────────────
        for result in last_results:
            _draw_face_overlay(frame, result, show_attrs=show_attrs)

        _draw_hud(
            frame,
            n_faces    = len(last_results),
            fps        = fps,
            paused     = paused,
            clf_thresh = clf_threshold,
            sim_thresh = unknown_det.threshold,
            det_mode   = detector.backend,
        )

        cv2.imshow("Mask-Aware CCTV Recognition  –  press q to quit", frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            logger.info("User quit.")
            break

        elif key == ord("p"):
            paused = not paused
            logger.info("Paused: %s", paused)

        elif key == ord("s"):
            path = _save_snapshot(frame)
            logger.info("Snapshot saved → %s", path)
            print(f"  Snapshot saved → {path}")

        elif key == ord("+") or key == ord("="):
            clf_threshold = min(1.0, clf_threshold + 0.05)
            unknown_det.set_threshold(min(1.0, unknown_det.threshold + 0.05))
            logger.info("Thresholds raised: clf=%.2f sim=%.2f",
                        clf_threshold, unknown_det.threshold)

        elif key == ord("-"):
            clf_threshold = max(0.0, clf_threshold - 0.05)
            unknown_det.set_threshold(max(0.0, unknown_det.threshold - 0.05))
            logger.info("Thresholds lowered: clf=%.2f sim=%.2f",
                        clf_threshold, unknown_det.threshold)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Session ended after %d frames  (avg FPS: %.1f).", frame_count, fps)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time CCTV person recognition with attribute overlay."
    )
    parser.add_argument(
        "--source", default=None,
        help=(
            "Camera index, video path, or RTSP URL. "
            f"Default: webcam {CAMERA_SOURCE}."
        ),
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
        "--width",  type=int, default=0,
        help="Requested capture width  (0 = camera default).",
    )
    parser.add_argument(
        "--height", type=int, default=0,
        help="Requested capture height (0 = camera default).",
    )
    parser.add_argument(
        "--upper-body", action="store_true",
        help="Also run upper-body cascade as detection fallback.",
    )
    parser.add_argument(
        "--no-attr", action="store_true",
        help="Hide the attribute side-panel (shows name + confidence only).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Resolve source: int for camera index, str for file/URL
    src = args.source
    if src is None:
        src = CAMERA_SOURCE
    elif src.isdigit():
        src = int(src)

    run_realtime(
        source            = src,
        clf_threshold     = args.clf_thresh,
        sim_threshold     = args.sim_thresh,
        dl_backend        = args.dl_backend,
        det_backend       = args.det_backend,
        frame_width       = args.width,
        frame_height      = args.height,
        show_attrs        = not args.no_attr,
        detect_upper_body = args.upper_body,
    )
