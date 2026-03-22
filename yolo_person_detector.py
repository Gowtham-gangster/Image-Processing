"""
yolo_person_detector.py
=======================
Ultra-precise Person Detection using YOLOv8 to guarantee correct pairing
of a physical body bounding box to the facial crop found within it.

Requirements met
----------------
1. Use YOLOv8 pretrained weights to detect persons in CCTV frames.
2. For each detected person bounding box, extract the body region.
3. Run face detection inside the person bounding box.
4. Pair each detected face with the correct person body.
5. Output a combined detection object.

Usage
-----
    from yolo_person_detector import YoloPersonDetector
    from face_alignment import FaceAligner
    
    aligner = FaceAligner()
    detector = YoloPersonDetector(aligner=aligner)
    
    results = detector.detect(frame)
    for res in results:
        print(res["person_bbox"]) # [x, y, w, h] of Body
        print(res["face_bbox"])   # None or [x, y, w, h] of Face (relative to full frame)
        # res["body_crop"] and res["face_crop"] are the respective numpy arrays
"""

import cv2
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from config import LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class YoloPersonDetector:
    """
    Leverages state-of-the-art YOLOv8 to detect complete persons, then explicitly 
    searches for faces *inside* those specific coordinate bounds to perfectly pair 
    body features with facial features, even in crowded CCTV scenes.
    """

    def __init__(self, aligner: Any, model_name: str = "yolov8n.pt", conf_threshold: float = 0.5) -> None:
        """
        Parameters
        ----------
        aligner : FaceAligner
            An initialized instance of FaceAligner (MTCNN) to find faces inside YOLO bodies.
        model_name : str
            The Ultralytics model string. Will auto-download if missing. 'yolov8n.pt' is nano (fastest).
        conf_threshold : float
            Minimum YOLO confidence to keep a person bounding box.
        """
        self.aligner = aligner
        self.conf_threshold = conf_threshold
        
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8 model '%s' ...", model_name)
            self.model = YOLO(model_name)
            # Suppress verbose UL logging on every frame
            import logging as ul_logging
            ul_logging.getLogger("ultralytics").setLevel(ul_logging.WARNING)
            logger.info("YOLOv8 successfully loaded.")
        except ImportError:
            logger.error("The 'ultralytics' package is required. Run: pip install ultralytics")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects persons, extracts bodies, and pairs them with sub-detected faces.
        
        Returns
        -------
        list of dict:
            [
                {
                    "person_bbox": [x, y, w, h],
                    "person_conf": float,
                    "body_crop": np.ndarray,
                    "face_bbox": [x, y, w, h] or None,
                    "face_crop": np.ndarray or None
                }, ...
            ]
        """
        if frame is None or frame.size == 0:
            return []
            
        fh, fw = frame.shape[:2]
        final_results = []
        
        # 1. Run YOLOv8 on the frame, restricting detection purely to Class 0 (person)
        results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return []
            
        boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        confs = results[0].boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            
            # Clamp to frame just in case
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(fw, x2)
            y2 = min(fh, y2)
            
            bw = x2 - x1
            bh = y2 - y1
            
            if bw < 10 or bh < 10:
                continue
                
            # 2. Extract strictly the Upper Body Region (top 45%)
            y2_upper = min(y2, y1 + int(bh * 0.45))
            body_crop = frame[y1:y2_upper, x1:x2]
            
            # 3. Super-scale 1.5x and run Face Detection
            upscaled_body = cv2.resize(body_crop, None, fx=1.5, fy=1.5)
            face_results = self.aligner.align(upscaled_body)
            
            face_bbox = None
            face_crop = None
            face_conf = 0.0
            
            if face_results:
                best_face = max(face_results, key=lambda f: f["box"][2] * f["box"][3])
                
                # Rescale face bounds back down to 1.0x native body indices
                rel_fx = int(best_face["box"][0] / 1.5)
                rel_fy = int(best_face["box"][1] / 1.5)
                fw_f   = int(best_face["box"][2] / 1.5)
                fh_f   = int(best_face["box"][3] / 1.5)
                
                abs_fx = x1 + rel_fx
                abs_fy = y1 + rel_fy
                
                abs_fx = max(0, abs_fx)
                abs_fy = max(0, abs_fy)
                fw_f = min(fw - abs_fx, fw_f)
                fh_f = min(fh - abs_fy, fh_f)
                
                face_bbox = [abs_fx, abs_fy, fw_f, fh_f]
                face_crop = best_face["aligned_crop"]
                face_conf = best_face["confidence"]
                
            final_results.append({
                "person_bbox": [x1, y1, bw, bh],
                "person_conf": float(conf),
                "body_crop": body_crop,
                "face_bbox": face_bbox,
                "face_crop": face_crop,
                "face_conf": float(face_conf)
            })
            
        return final_results
