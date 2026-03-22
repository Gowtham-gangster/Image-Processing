"""
realtime_recognition.py
=======================
Mode 2 - Recognition Mode (Strict Separation)
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
from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
from attributes_manager import AttributesManager

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class RealtimeRecognition:
    def __init__(self, camera_idx: int = CAMERA_SOURCE, threshold: float = 0.6, test_dir: str = None) -> None:
        self.camera_idx = camera_idx
        self.threshold = threshold
        self.test_dir = test_dir
        
        if not (os.path.exists("embeddings/faiss_index.index") and os.path.exists("embeddings/labels.pkl")):
            logger.error("CRITICAL: Models missing! Recognition should NOT run without embeddings and FAISS index.")
            raise RuntimeError("Please run train_embeddings.py to generate models before starting realtime recognition.")
            
        logger.info("Loading FAISS ...")
        self.faiss_idx = faiss.read_index("embeddings/faiss_index.index")
        
        with open("embeddings/labels.pkl", "rb") as f:
            self.label_map = pickle.load(f)
            
        self.detector = FaceDetector()
        self.extractor = FeatureExtractor()
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
        boxes = self.detector.detect_faces(frame)
        for (x, y, w, h) in boxes:
            if w < 10 or h < 10:
                continue
                
            face_crop = frame[y:y+h, x:x+w]
            emb = self.extractor.extract(face_crop, masked=False)
            
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                
            q_vec = np.array([emb], dtype=np.float32)
            distances, indices = self.faiss_idx.search(q_vec, 1)
            
            distance = distances[0][0]
            idx = indices[0][0]
            
            person_id = "Unknown Person"
            if distance < self.threshold and idx != -1:
                person_id = self.label_map.get(idx, "Unknown Person")
                
            decision = "Known" if person_id != "Unknown Person" else "Unknown"
            
            print(f"Detected person: {person_id}")
            print(f"Distance: {distance:.4f}")
            print(f"Result: {decision}")
            
            sim_score = max(0.0, 1.0 - ((distance**2) / 2.0))
            
            is_known = decision == "Known"
            attrs = self.attributes.get_attributes(person_id) if is_known else {}
            if attrs is None: attrs = {}
            
            self._draw_overlay(display_frame, x, y, w, h, person_id, sim_score, is_known, attrs)

    def _draw_overlay(self, frame: np.ndarray, x: int, y: int, w: int, h: int, person_id: str, conf: float, is_known: bool, attrs: dict) -> None:
        colour = COLOUR_KNOWN if is_known else COLOUR_UNKNOWN
        cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
        
        if is_known:
            display_lines = [
                f"Name: {attrs.get('name', person_id)}",
                f"Age: {attrs.get('age', 'N/A')}",
                f"Phone: {attrs.get('phone', 'N/A')}",
                f"Address: {attrs.get('address', 'N/A')}",
                f"Confidence Score: {conf:.2f}",
                "------------------------------"
            ]
        else:
            display_lines = [
                "--------",
                "Unknown Person",
                f"Confidence Score: {conf:.2f}",
                "------------------------------"
            ]
            
        total_height = len(display_lines) * 20
        start_y = max(20, y - total_height - 5)
        
        for i, line in enumerate(display_lines):
            text_y = start_y + (i * 20)
            cv2.putText(frame, line, (x, text_y), FONT, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, text_y), FONT, 0.45, colour, 1, cv2.LINE_AA)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=CAMERA_SOURCE, help="Webcam device index.")
    parser.add_argument("--test_dir", type=str, default=None, help="Process explicit test directory images instead of camera.")
    args = parser.parse_args()
    
    app = RealtimeRecognition(camera_idx=args.camera, test_dir=args.test_dir)
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
