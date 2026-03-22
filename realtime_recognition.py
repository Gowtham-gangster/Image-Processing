"""
realtime_recognition.py
=======================
Mode 2 - Recognition Mode (Strict Multi-Modal Fusion)
"""

import os
import cv2
import faiss
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path

from config import CAMERA_SOURCE, LOG_LEVEL, COLOUR_KNOWN, COLOUR_UNKNOWN, FONT
from face_alignment import FaceAligner
from yolo_person_detector import YoloPersonDetector
from body_feature_extractor import BodyFeatureExtractor
from attribute_extractor import AttributeExtractor
from feature_extractor import FeatureExtractor
from attributes_manager import AttributesManager

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class RealtimeRecognition:
    def __init__(self, camera_idx: int = CAMERA_SOURCE, threshold: float = 0.6, test_dir: str = None) -> None:
        self.camera_idx = camera_idx
        self.threshold = threshold
        self.test_dir = test_dir
        
        # Paths for Multi-Modal Models
        face_idx_path = "embeddings/faiss_index.index"
        body_idx_path = "embeddings/body_faiss.index"
        attr_idx_path = "embeddings/attr_faiss.index"
        multi_labels_path = "embeddings/multi_labels.pkl"
        legacy_labels_path = "embeddings/labels.pkl"

        if not os.path.exists(face_idx_path) or not os.path.exists(multi_labels_path):
            logger.error("CRITICAL: Multi-modal models missing! Please run train_embeddings.py first.")
            raise RuntimeError("Missing FAISS indices or multi_labels.pkl.")
            
        logger.info("Loading Multi-Modal FAISS indices...")
        self.face_faiss = faiss.read_index(face_idx_path)
        self.body_faiss = faiss.read_index(body_idx_path) if os.path.exists(body_idx_path) else None
        self.attr_faiss = faiss.read_index(attr_idx_path) if os.path.exists(attr_idx_path) else None
        
        with open(multi_labels_path, "rb") as f:
            self.multi_labels = pickle.load(f)
            
        self.label_map = {}
        if os.path.exists(legacy_labels_path):
            with open(legacy_labels_path, "rb") as f:
                self.label_map = pickle.load(f)
        
        # Initialize Components
        self.aligner = FaceAligner(min_confidence=0.4)
        self.yolo_detector = YoloPersonDetector(aligner=self.aligner)
        self.face_extractor = FeatureExtractor()
        self.body_extractor = BodyFeatureExtractor()
        self.attr_extractor = AttributeExtractor()
        self.attributes = AttributesManager()
        
    def run(self) -> None:
        if self.test_dir:
            self._process_test_directory()
        else:
            self._process_camera()
            
    def _process_test_directory(self) -> None:
        logger.info(f"Mode 2 - Recognition active on test folder: {self.test_dir}")
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for img_path in Path(self.test_dir).rglob(ext):
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                display_frame = frame.copy()
                self._analyze_frame(frame, display_frame)
                
                cv2.imshow("Mode 2 - Real Time Analysis", display_frame)
                if cv2.waitKey(0) & 0xFF == ord('q'): 
                    break
        cv2.destroyAllWindows()
        
    def _process_camera(self) -> None:
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_idx}")
            return
            
        logger.info("Mode 2 - Recognition active on webcam. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display_frame = frame.copy()
            self._analyze_frame(frame, display_frame)
            cv2.imshow("Mode 2 - Real Time Analysis", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def _analyze_frame(self, frame: np.ndarray, display_frame: np.ndarray) -> None:
        if self.face_faiss is None:
            return

        # 1. Detection Phase (YOLO as primary)
        results = self.yolo_detector.detect(frame)
        if not results:
            return

        f_labels_map = self.multi_labels.get("face_labels", [])
        b_labels_map = self.multi_labels.get("body_labels", [])
        a_labels_map = self.multi_labels.get("attr_labels", [])

        for res in results:
            person_bbox = res.get("person_bbox")
            if person_bbox is None: continue
            px, py, pw, ph = person_bbox
            
            body_crop = res.get("body_crop")
            face_crop = res.get("face_crop")
            
            # 2. Embedding & Search Phase
            scores = {} # pid -> {F, B, A} scores
            
            def search(idx, extractor, crop):
                if idx is None or crop is None or crop.size == 0:
                    return None, None
                emb = extractor.extract(crop, masked=False) if extractor == self.face_extractor else extractor.extract(crop)
                norm = np.linalg.norm(emb)
                if norm > 0: emb = emb / norm
                q_vec = np.array([emb], dtype=np.float32)
                dists, inds = idx.search(q_vec, 5)
                return dists[0], inds[0]

            # Body Search
            b_d, b_i = search(self.body_faiss, self.body_extractor, body_crop)
            if b_i is not None:
                for d, i in zip(b_d, b_i):
                    if i != -1 and i < len(b_labels_map):
                        pid_int = b_labels_map[i]
                        pid = self.label_map.get(pid_int, str(pid_int))
                        scr = max(0.0, 1.0 - (d / 2.0))
                        if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                        scores[pid]["B"] = max(scores[pid]["B"], scr)

            # Attribute Search
            a_d, a_i = search(self.attr_faiss, self.attr_extractor, body_crop)
            if a_i is not None:
                for d, i in zip(a_d, a_i):
                    if i != -1 and i < len(a_labels_map):
                        pid_int = a_labels_map[i]
                        pid = self.label_map.get(pid_int, str(pid_int))
                        scr = max(0.0, 1.0 - (d / 2.0))
                        if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                        scores[pid]["A"] = max(scores[pid]["A"], scr)

            # Face Search
            has_face = False
            if face_crop is not None and face_crop.size > 0:
                has_face = True
                f_d, f_i = search(self.face_faiss, self.face_extractor, face_crop)
                if f_i is not None:
                    for d, i in zip(f_d, f_i):
                        if i != -1 and i < len(f_labels_map):
                            pid_int = f_labels_map[i]
                            pid = self.label_map.get(pid_int, str(pid_int))
                            scr = max(0.0, 1.0 - (d / 2.0))
                            if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                            scores[pid]["F"] = max(scores[pid]["F"], scr)

            # 3. Fusion Phase
            best_id = "Unknown Person"
            best_score = 0.0
            mode_str = "N/A"
            
            for pid, s in scores.items():
                if has_face:
                    fused = (s["F"] * 0.5) + (s["B"] * 0.3) + (s["A"] * 0.2)
                    current_mode = "Face+Body+Attr"
                else:
                    fused = (s["B"] * 0.7) + (s["A"] * 0.3)
                    current_mode = "Body+Attr"
                
                if fused > best_score:
                    best_score = fused
                    best_id = pid
                    mode_str = current_mode
            
            # Apply Threshold
            final_id = best_id if best_score >= self.threshold else "Unknown Person"
            is_known = final_id != "Unknown Person"
            
            attrs = self.attributes.get_attributes(final_id) if is_known else {}
            if attrs is None: attrs = {}
            
            self._draw_overlay(display_frame, px, py, pw, ph, final_id, best_score, is_known, attrs, mode_str if is_known else "N/A")

    def _draw_overlay(self, frame: np.ndarray, x: int, y: int, w: int, h: int, person_id: str, conf: float, is_known: bool, attrs: dict, mode: str) -> None:
        colour = COLOUR_KNOWN if is_known else COLOUR_UNKNOWN
        cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
        
        if is_known:
            display_lines = [
                f"ID: {person_id}",
                f"Name: {attrs.get('name', 'N/A')}",
                f"Mode: {mode}",
                f"Score: {conf:.2f}",
                f"Age/Addr: {attrs.get('age', 'N/A')} | {attrs.get('address', 'N/A')[:10]}...",
                "------------------------------"
            ]
        else:
            display_lines = [
                "Unknown Person",
                "------------------------------"
            ]
            
        # Draw background box for text
        text_h = len(display_lines) * 15
        box_y = max(0, y - text_h - 10)
        cv2.rectangle(frame, (x, box_y), (x + w, y), (0, 0, 0), -1)
        
        for i, line in enumerate(display_lines):
            text_y = box_y + 15 + (i * 15)
            cv2.putText(frame, line, (x + 5, text_y), FONT, 0.4, colour, 1, cv2.LINE_AA)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=CAMERA_SOURCE, help="Webcam device index.")
    parser.add_argument("--test_dir", type=str, default=None, help="Process explicit test directory images instead of camera.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for recognition.")
    args = parser.parse_args()
    
    app = RealtimeRecognition(camera_idx=args.camera, threshold=args.threshold, test_dir=args.test_dir)
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception("Runtime error occurred")
