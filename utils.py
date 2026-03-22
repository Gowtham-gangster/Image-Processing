"""
utils.py
========
Drawing helpers, overlay rendering, and miscellaneous utilities.
"""

import logging
import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    COLOUR_KNOWN, COLOUR_UNKNOWN, COLOUR_MASKED,
    FONT, LOGS_DIR,
)
from person_identifier import IdentificationResult

logger = logging.getLogger(__name__)


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_face_box(frame: np.ndarray,
                  bbox: tuple[int, int, int, int],
                  result: IdentificationResult) -> np.ndarray:
    """
    Draw bounding box and identification overlay on *frame*.

    Parameters
    ----------
    frame  : np.ndarray  BGR image (modified in-place and returned).
    bbox   : (x, y, w, h)
    result : IdentificationResult from person_identifier.

    Returns
    -------
    np.ndarray  – annotated frame.
    """
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    # Choose colour
    if result.is_masked:
        colour = COLOUR_MASKED
    elif result.is_known:
        colour = COLOUR_KNOWN
    else:
        colour = COLOUR_UNKNOWN

    # Box
    cv2.rectangle(frame, (x, y), (x2, y2), colour, 2)

    # Build label lines
    label_lines = _build_label_lines(result)

    # Background pill behind text
    line_h = 20
    panel_h = len(label_lines) * line_h + 6
    panel_y1 = max(y - panel_h, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, panel_y1), (x2, y), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Text
    for i, line in enumerate(label_lines):
        text_y = panel_y1 + (i + 1) * line_h - 2
        cv2.putText(
            frame, line,
            (x + 4, text_y),
            FONT, 0.47, colour, 1, cv2.LINE_AA,
        )

    return frame


def draw_status_bar(frame: np.ndarray,
                    detector_mode: str,
                    mask_mode: str,
                    n_faces: int) -> np.ndarray:
    """Draw a bottom status bar with system info."""
    h, w = frame.shape[:2]
    bar_h = 26
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    status = (
        f"  {ts}   |   Detector: {detector_mode.upper()}   |   "
        f"Mask model: {mask_mode.upper()}   |   Faces: {n_faces}"
    )
    cv2.putText(frame, status, (4, h - 7), FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def draw_attributes_panel(frame: np.ndarray,
                           result: IdentificationResult,
                           position: tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Draw an expanded attribute panel in the corner of the frame.
    Used when a known person is detected.
    """
    if not result.is_known:
        return frame

    attrs = result.attributes
    lines = [
        f"Name    : {result.name}",
        f"ID      : {result.person_id}",
        f"Gender  : {attrs.get('gender', 'N/A')}",
        f"Age     : {attrs.get('age', 'N/A')}",
        f"Phone   : {attrs.get('phone', 'N/A')}",
        f"Address : {attrs.get('address', 'N/A')}",
        f"Masked  : {'Yes' if result.is_masked else 'No'}",
        f"Conf    : {result.confidence:.1%}",
    ]

    line_h = 22
    panel_w = 340
    panel_h = len(lines) * line_h + 14
    px, py = position

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), COLOUR_KNOWN, 1)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (px + 8, py + (i + 1) * line_h),
            FONT, 0.47, (220, 220, 220), 1, cv2.LINE_AA,
        )

    return frame


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_detection(result: IdentificationResult) -> None:
    """Append a timestamped detection event to the log file."""
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    log_file = Path(LOGS_DIR) / "detections.csv"

    header = "timestamp,person_id,name,is_known,is_masked,confidence,distance\n"
    row = (
        f"{datetime.datetime.now().isoformat()},"
        f"{result.person_id or ''},"
        f"{result.name},"
        f"{result.is_known},"
        f"{result.is_masked},"
        f"{result.confidence:.4f},"
        f"{result.distance:.4f}\n"
    )

    if not log_file.exists():
        log_file.write_text(header)
    with open(log_file, "a") as f:
        f.write(row)


# ── Private ───────────────────────────────────────────────────────────────────

def _build_label_lines(result: IdentificationResult) -> list[str]:
    """Build the list of text lines to display above the bounding box."""
    mask_tag = " [MASKED]" if result.is_masked else ""
    if result.is_known:
        return [
            f"{result.name}{mask_tag}",
            f"ID: {result.person_id}  Conf: {result.confidence:.0%}",
        ]
    return [f"Unknown Person{mask_tag}"]
