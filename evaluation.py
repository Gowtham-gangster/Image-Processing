"""
evaluation.py
=============
Evaluates the CNN Embedding face recognition system using dataset/test images.

Requirements met
----------------
1. Test model on dataset/test images.
2. Calculate accuracy.
3. Generate confusion matrix.
4. Output classification report.

Dependencies
------------
scikit-learn, numpy, matplotlib (optional for plots), OpenCV, MTCNN

Usage
-----
    from evaluation import EmbeddingEvaluator
    
    evaluator = EmbeddingEvaluator(threshold=0.65)
    metrics = evaluator.evaluate()
    
    print(f"Accuracy: {metrics['accuracy']:.2%}")

Usage (CLI)
-----------
    python evaluation.py
    python evaluation.py --thresh 0.70 --no-plot
"""

import os
import argparse
import logging
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import TEST_DIR, LOGS_DIR, LOG_LEVEL
from detection import Detector
from face_alignment import FaceAligner
from embedding_model import FaceNetEmbedder
from embedding_database import EmbeddingDatabase
from attributes_manager import AttributesManager
from unknown_detector import UnknownDetector

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

class EmbeddingEvaluator:
    """
    Runs face alignment, embedding extraction, and unknown-gate matching on the test/ directory.
    Uses sklearn to generate accuracy scores, classification txt reports, and confusion matrices.
    """

    def __init__(self, test_dir: str = TEST_DIR, threshold: float = 0.65) -> None:
        self.test_dir = test_dir
        self.threshold = threshold
        
        logger.info("Initializing Evaluation subsystems...")
        self.detector = Detector()
        self.aligner = FaceAligner()
        self.embedder = FaceNetEmbedder()
        self.db = EmbeddingDatabase()
        self.attrs = AttributesManager()
        
        if self.db.get_faiss_engine().index.ntotal == 0:
            logger.warning(
                "Embedding Database is empty! "
                "The evaluation will predict 'Unknown Person' for all faces. "
                "Did you run the training/enrollment script first?"
            )
            
        self.detector = UnknownDetector(threshold=self.threshold)
        
    def evaluate(self, show_plot: bool = True) -> dict:
        """
        Walks the test directory and generates all sklearn metrics.
        
        Returns
        -------
        dict
            Contains accuracy (float), classification report (str), y_true (list), y_pred (list)
        """
        logger.info("Running evaluation on %s with threshold %.2f", self.test_dir, self.threshold)
        
        if not os.path.exists(self.test_dir):
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
            
        y_true = []
        y_pred = []
        skipped = 0
        total_predictions = 0

        # Discover all subfolders (which act as ground truth labels)
        person_folders = [f for f in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, f))]
        
        for true_id in person_folders:
            folder_path = os.path.join(self.test_dir, true_id)
            for fname in os.listdir(folder_path):
                if os.path.splitext(fname)[1].lower() not in VALID_EXTENSIONS:
                    continue
                    
                img_path = os.path.join(folder_path, fname)
                frame = cv2.imread(img_path)
                
                if frame is None:
                    logger.warning("Failed to read test image: %s", img_path)
                    skipped += 1
                    continue
                    
                # 1. Align the face
                aligned_faces = self.aligner.align(frame)
                
                if not aligned_faces:
                    # If MTCNN couldn't find a face, record as a total detection failure
                    y_true.append(true_id)
                    y_pred.append("No_Face_Detected")
                    continue
                    
                # Usually test images have 1 tightly cropped face. Take the most confident hit.
                best_face = max(aligned_faces, key=lambda f: f["confidence"])
                aligned_crop = best_face["aligned_crop"]
                
                # 2. Extract Embedding
                try:
                    embedding = self.embedder.extract(aligned_crop)
                except Exception as e:
                    logger.warning("Feature extraction failed for %s: %s", img_path, e)
                    y_true.append(true_id)
                    y_pred.append("Extraction_Error")
                    continue
                    
                # 3. Predict Identity
                predicted_label, _ = self.detector.identify(embedding, self.db)
                
                # Record result
                y_true.append(true_id)
                y_pred.append(predicted_label)
                total_predictions += 1
                
        if total_predictions == 0:
            logger.error("No valid test images processed!")
            return {"accuracy": 0.0}
            
        # 4. Sklearn calculations
        accuracy = accuracy_score(y_true, y_pred)
        
        # Format "Unknown Person", "No_Face_Detected", etc. neatly in reports
        labels = sorted(set(y_true) | set(y_pred))
        
        report = classification_report(y_true, y_pred, zero_division=0)
        
        self._print_report(accuracy, report, skipped)
        self._build_confusion_matrix(y_true, y_pred, labels, show_plot)
        
        return {
            "accuracy": accuracy,
            "report_str": report,
            "y_true": y_true,
            "y_pred": y_pred
        }
        
    def _print_report(self, accuracy: float, report_str: str, skipped: int) -> None:
        """Renders the text report to the terminal."""
        print()
        print("═" * 60)
        print("  EMBEDDING SYSTEM EVALUATION REPORT")
        print("═" * 60)
        print(f"  Similarity Threshold : {self.threshold:.2f}")
        print(f"  Images Skipped (I/O) : {skipped}")
        print("─" * 60)
        print(f"  Overall Accuracy     : {accuracy:.4f}  ({accuracy:.2%})")
        print("─" * 60)
        print("  CLASSIFICATION REPORT")
        print("─" * 60)
        print(report_str)
        print("═" * 60)
        print()
        
    def _build_confusion_matrix(self, y_true: list, y_pred: list, labels: list, show_plot: bool) -> None:
        """Constructs text CM and optional matplotlib heatmap."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        print("  CONFUSION MATRIX")
        print("  (rows = true, cols = predicted)\n")
        
        col_w = max(len(lb) for lb in labels) + 2
        header = "  " + "".ljust(col_w) + "".join(lb.center(col_w) for lb in labels)
        print(header)
        print("  " + "─" * len(header.rstrip()))
        for i, lb in enumerate(labels):
            row = "  " + lb.ljust(col_w) + "".join(str(v).center(col_w) for v in cm[i])
            print(row)
        print()

        if show_plot:
            try:
                import matplotlib.pyplot as plt # type: ignore
                fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels))))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation="vertical")
                ax.set_title(f"Confusion Matrix (Embedding Similarity ≥ {self.threshold})")
                plt.tight_layout()
                
                os.makedirs(LOGS_DIR, exist_ok=True)
                plot_path = os.path.join(LOGS_DIR, "embedding_confusion_matrix.png")
                plt.savefig(plot_path)
                logger.info("Confusion matrix heatmap saved to %s", plot_path)
                
                plt.show()
                plt.close(fig)
            except ImportError:
                logger.warning("matplotlib is not installed. Text-only confusion matrix used.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 128-D CNN Embedding accuracy with sklearn.")
    parser.add_argument("--test-dir", default=TEST_DIR, help=f"Path to test split (default: {TEST_DIR}).")
    parser.add_argument(
        "--thresh", type=float, default=0.65,
        help="Cosine similarity threshold (default: 0.65)."
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the matplotlib heatmap display.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    
    try:
        evaluator = EmbeddingEvaluator(test_dir=args.test_dir, threshold=args.thresh)
        evaluator.evaluate(show_plot=not args.no_plot)
    except Exception as e:
        logger.error("Evaluation aborted: %s", e)
