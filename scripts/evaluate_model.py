"""
evaluate_model.py
=================
Formal evaluation of the trained person-identification model on dataset/test/.

Metrics produced
----------------
  • Overall accuracy
  • Per-class precision / recall / F1  (sklearn classification_report)
  • Confusion matrix  (text + optional heatmap PNG)

Output files (written to logs/)
--------------------------------
  evaluation_<timestamp>.txt    – full text report
  confusion_matrix_<timestamp>.png – heatmap (if matplotlib is available)

Usage
-----
    python evaluate_model.py
    python evaluate_model.py --clf-thresh 0.70 --sim-thresh 0.65
    python evaluate_model.py --no-plot       # skip heatmap
    python evaluate_model.py --save-report   # save .txt to logs/
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import TEST_DIR, LOGS_DIR, LOG_LEVEL
from recognition import Recognizer, RecognitionResult

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Data collection ───────────────────────────────────────────────────────────

def collect_predictions(
    rec: Recognizer,
    test_dir: str = TEST_DIR,
) -> tuple[list[str], list[str], list[float], list[float]]:
    """
    Run recognition on every image in *test_dir* and collect predictions.

    Returns
    -------
    y_true       : list[str]   – ground-truth person_id (folder name)
    y_pred       : list[str]   – predicted person_id, or 'Unknown'
    clf_confs    : list[float] – classifier confidence per sample
    sims         : list[float] – cosine similarity per sample
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: '{test_dir}'")

    person_dirs = sorted(
        p for p in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, p))
    )
    if not person_dirs:
        raise ValueError(f"No person sub-folders found in '{test_dir}'.")

    y_true:    list[str]   = []
    y_pred:    list[str]   = []
    clf_confs: list[float] = []
    sims:      list[float] = []
    skipped = 0

    total_imgs = sum(
        len([f for f in os.listdir(os.path.join(test_dir, p))
             if os.path.splitext(f)[1].lower() in IMG_EXTS])
        for p in person_dirs
    )
    logger.info("Evaluating %d images across %d identities …",
                total_imgs, len(person_dirs))

    for pid in person_dirs:
        person_dir = os.path.join(test_dir, pid)
        img_files  = sorted(
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
        for fname in img_files:
            img_path = os.path.join(person_dir, fname)
            frame    = cv2.imread(img_path)
            if frame is None:
                logger.warning("Cannot read '%s' — skipping.", img_path)
                skipped += 1
                continue

            results = rec.recognize(frame)

            if not results:
                pred_id  = "Unknown"
                clf_conf = 0.0
                sim      = 0.0
            else:
                best     = results[0]           # highest confidence first
                pred_id  = best.person_id if best.is_known else "Unknown"
                clf_conf = best.clf_confidence
                sim      = best.embed_similarity

            y_true.append(pid)
            y_pred.append(pred_id)
            clf_confs.append(clf_conf)
            sims.append(sim)

            status = "✔" if pred_id == pid else "✘"
            logger.debug("%s  true=%-12s  pred=%-12s  clf=%.2f  sim=%.2f",
                         status, pid, pred_id, clf_conf, sim)

    if skipped:
        logger.warning("%d image(s) skipped (unreadable).", skipped)
    return y_true, y_pred, clf_confs, sims


# ── Report builders ───────────────────────────────────────────────────────────

def build_report(
    y_true: list[str],
    y_pred: list[str],
    clf_confs: list[float],
    sims: list[float],
    clf_threshold: float,
    sim_threshold: float,
) -> str:
    """
    Build the full text evaluation report.

    Returns the report as a single string (also printed to stdout).
    """
    classes = sorted(set(y_true) | (set(y_pred) - {"Unknown"}))

    # Remap Unknown predictions to a consistent label
    y_pred_clean = [p if p != "Unknown" else "__Unknown__" for p in y_pred]
    y_true_clean = list(y_true)
    all_labels = classes + (["__Unknown__"] if "__Unknown__" in y_pred_clean else [])

    accuracy = accuracy_score(y_true_clean, y_pred_clean)
    n_unknown = y_pred_clean.count("__Unknown__")
    n_total   = len(y_true)

    lines = []
    sep   = "═" * 65

    lines += [
        "",
        sep,
        "  EVALUATION REPORT — Mask-Aware Person Identification System",
        f"  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        f"  Test images    : {n_total}",
        f"  Identities     : {len(classes)}",
        f"  CLF threshold  : {clf_threshold:.2f}",
        f"  SIM threshold  : {sim_threshold:.2f}",
        f"  Unknown outputs: {n_unknown} ({n_unknown/n_total:.1%})",
        "─" * 65,
        f"  Overall Accuracy : {accuracy:.4f}  ({accuracy:.2%})",
        f"  Avg CLF conf     : {np.mean(clf_confs):.4f}",
        f"  Avg Sim score    : {np.mean(sims):.4f}",
        "─" * 65,
        "  CLASSIFICATION REPORT",
        "─" * 65,
    ]

    report = classification_report(
        y_true_clean, y_pred_clean,
        labels=all_labels,
        zero_division=0,
    )
    lines.append(report)

    # ── Per-class accuracy ────────────────────────────────────────────────────
    lines += ["─" * 65, "  PER-IDENTITY ACCURACY", "─" * 65]
    lines.append(f"  {'Identity':<20} {'Total':>6} {'Correct':>8} {'Accuracy':>10}")
    lines.append("  " + "·" * 50)
    for pid in sorted(set(y_true)):
        idxs    = [i for i, t in enumerate(y_true) if t == pid]
        total   = len(idxs)
        correct = sum(1 for i in idxs if y_pred[i] == pid)
        acc_p   = correct / total if total else 0
        lines.append(f"  {pid:<20} {total:>6} {correct:>8} {acc_p:>10.1%}")
    lines.append(sep)
    lines.append("")

    report_str = "\n".join(lines)
    print(report_str)
    return report_str


def build_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    save_path: str | None = None,
    show: bool = True,
) -> np.ndarray:
    """
    Compute and optionally display / save the confusion matrix.

    Returns
    -------
    cm : np.ndarray  (N × N)
    """
    classes = sorted(set(y_true) | (set(y_pred) - {"Unknown"}))
    y_pred_clean = [p if p != "Unknown" else "__Unknown__" for p in y_pred]
    all_labels   = classes + (["__Unknown__"] if "__Unknown__" in y_pred_clean else [])

    cm = confusion_matrix(y_true, y_pred_clean, labels=all_labels)

    # ── Text confusion matrix ──────────────────────────────────────────────────
    col_w = max(len(lb) for lb in all_labels) + 2
    header = "  " + "".ljust(col_w) + "".join(lb.center(col_w) for lb in all_labels)
    print("\n  CONFUSION MATRIX")
    print("  (rows = true, cols = predicted)\n")
    print(header)
    print("  " + "─" * len(header.rstrip()))
    for i, lb in enumerate(all_labels):
        row = "  " + lb.ljust(col_w) + "".join(str(v).center(col_w) for v in cm[i])
        print(row)
    print()

    # ── Heatmap via matplotlib ─────────────────────────────────────────────────
    if save_path or show:
        try:
            import matplotlib                          # type: ignore
            matplotlib.use("Agg")                      # headless-safe backend
            import matplotlib.pyplot as plt           # type: ignore

            fig, ax = plt.subplots(figsize=(max(6, len(all_labels)), max(5, len(all_labels))))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
            disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation="vertical")
            ax.set_title("Confusion Matrix — Person Identification", fontsize=12, pad=14)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info("Confusion matrix saved → %s", save_path)
                print(f"  Confusion matrix heatmap → {save_path}")

            if show:
                plt.show()
            plt.close(fig)

        except ImportError:
            logger.warning("matplotlib not installed — heatmap skipped.")
            print("  (Install matplotlib to generate the heatmap PNG.)")

    return cm


# ── Entry point ───────────────────────────────────────────────────────────────

def evaluate(
    test_dir: str = TEST_DIR,
    clf_threshold: float = 0.55,
    sim_threshold: float = 0.50,
    dl_backend: str = "auto",
    det_backend: str = "auto",
    show_plot: bool = True,
    save_report: bool = False,
) -> dict:
    """
    Full evaluation pipeline.

    Returns
    -------
    dict with keys: accuracy, y_true, y_pred, clf_confs, sims, cm
    """
    # ── Load recogniser ───────────────────────────────────────────────────────
    logger.info("Loading recognition system …")
    try:
        rec = Recognizer(
            clf_threshold=clf_threshold,
            sim_threshold=sim_threshold,
            dl_backend=dl_backend,
            detect_backend=det_backend,
        )
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        print("  Run  python train_model.py  first.\n", file=sys.stderr)
        sys.exit(1)

    # ── Collect predictions ───────────────────────────────────────────────────
    y_true, y_pred, clf_confs, sims = collect_predictions(rec, test_dir)

    if not y_true:
        print("\n  No test images found — nothing to evaluate.\n")
        sys.exit(0)

    # ── Build report ──────────────────────────────────────────────────────────
    report_str = build_report(y_true, y_pred, clf_confs, sims,
                               clf_threshold, sim_threshold)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    os.makedirs(LOGS_DIR, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path  = os.path.join(LOGS_DIR, f"confusion_matrix_{ts}.png") if save_report else None
    cm       = build_confusion_matrix(y_true, y_pred,
                                       save_path=cm_path, show=show_plot)

    # ── Save text report ──────────────────────────────────────────────────────
    if save_report:
        rpt_path = os.path.join(LOGS_DIR, f"evaluation_{ts}.txt")
        with open(rpt_path, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"  Text report saved → {rpt_path}")

    accuracy = accuracy_score(y_true, [p if p != "Unknown" else "__Unknown__"
                                        for p in y_pred])
    return {
        "accuracy":   accuracy,
        "y_true":     y_true,
        "y_pred":     y_pred,
        "clf_confs":  clf_confs,
        "sims":       sims,
        "cm":         cm,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on dataset/test/ with sklearn metrics."
    )
    parser.add_argument(
        "--test-dir", default=TEST_DIR,
        help=f"Root of test split (default: {TEST_DIR}).",
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
        help="Embedding backend (default: auto).",
    )
    parser.add_argument(
        "--det-backend", choices=["auto", "dnn", "haar"], default="auto",
        help="Detector backend (default: auto).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Do not display the matplotlib confusion matrix window.",
    )
    parser.add_argument(
        "--save-report", action="store_true",
        help="Save evaluation .txt and confusion matrix .png to logs/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        test_dir      = args.test_dir,
        clf_threshold = args.clf_thresh,
        sim_threshold = args.sim_thresh,
        dl_backend    = args.dl_backend,
        det_backend   = args.det_backend,
        show_plot     = not args.no_plot,
        save_report   = args.save_report,
    )
