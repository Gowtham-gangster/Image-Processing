import os
import cv2
import logging
import numpy as np
from config import TRAIN_DIR
from embedding_model import get_embedder
from body_feature_extractor import BodyFeatureExtractor
from database import DatabaseManager
from face_alignment import FaceAligner
from yolo_person_detector import YoloPersonDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_database():
    db = DatabaseManager()
    face_embedder = get_embedder()
    body_embedder = BodyFeatureExtractor()
    aligner = FaceAligner()
    yolo_detector = YoloPersonDetector(aligner=aligner)

    logger.info("Clearing existing embeddings to rebuild from scratch...")
    db.clear_face_embeddings()
    db.clear_body_embeddings()

    persons_added = set()

    for person_id in os.listdir(TRAIN_DIR):
        person_dir = os.path.join(TRAIN_DIR, person_id)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            logger.info(f"Processing {img_path}")
            
            # Using YOLO to get both face and body crops where possible
            results = yolo_detector.detect(img)
            
            if not results:
                # Fallback to just MTCNN if YOLO misses
                aligned_faces = aligner.align(img)
                for face in aligned_faces:
                    try:
                        emb = face_embedder.extract(face["aligned_crop"])
                        db.add_face_embedding(person_id, emb)
                        persons_added.add(person_id)
                    except Exception as e:
                        logger.error(f"Fallback Face Embed Error {img_path}: {e}")
                continue
                
            for res in results:
                body_crop = res.get("body_crop")
                face_crop = res.get("face_crop")
                
                if face_crop is not None and face_crop.size > 0:
                    try:
                        emb = face_embedder.extract(face_crop)
                        db.add_face_embedding(person_id, emb)
                        persons_added.add(person_id)
                    except Exception as e:
                        logger.error(f"Face Embed Error {img_path}: {e}")
                        
                if body_crop is not None and body_crop.size > 0:
                    try:
                        emb = body_embedder.extract(body_crop)
                        db.add_body_embedding(person_id, emb)
                        persons_added.add(person_id)
                    except Exception as e:
                        logger.error(f"Body Embed Error {img_path}: {e}")

    logger.info(f"Successfully populated embeddings for {len(persons_added)} persons.")

if __name__ == "__main__":
    populate_database()
