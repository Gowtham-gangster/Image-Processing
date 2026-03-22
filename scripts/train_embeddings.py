import os
import faiss
import pickle
import logging
import numpy as np
import cv2

from dataset_loader import DatasetLoader
from face_alignment import FaceAligner
from feature_extractor import FeatureExtractor
from body_feature_extractor import BodyFeatureExtractor
from attribute_extractor import AttributeExtractor
from yolo_person_detector import YoloPersonDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training_pipeline() -> None:
    loader = DatasetLoader()
    aligner = FaceAligner(min_confidence=0.40)
    extractor = FeatureExtractor()
    body_extractor = BodyFeatureExtractor()
    attr_extractor = AttributeExtractor()
    yolo_detector = YoloPersonDetector(aligner=aligner)
    
    face_embeddings = []
    face_labels_int = []
    
    body_embeddings = []
    body_labels_int = []
    
    attr_embeddings = []
    attr_labels_int = []
    
    person_to_int = {}
    label_map = {}
    current_label_id = 0
    total_images_processed = 0
    
    logger.info("Step 1: Loading dataset from dataset/train only")
    for person_id, img_path, img_bgr in loader.load_training_data():
        if person_id not in person_to_int:
            person_to_int[person_id] = current_label_id
            label_map[current_label_id] = person_id
            current_label_id += 1
            
        label_int = person_to_int[person_id]
        print(f"Processing person: {person_id} -> {img_path}")
        
        face_crop = None
        body_crop = None
        
        # 1. Global Face Check (1.5x Scale)
        upscaled = cv2.resize(img_bgr, None, fx=1.5, fy=1.5)
        faces = aligner.align(upscaled)
        if faces:
            best_face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
            face_crop = best_face["aligned_crop"]
            
        # 2. YOLO Body Extract
        yolo_res = yolo_detector.detect(img_bgr)
        if yolo_res:
            best_body = max(yolo_res, key=lambda x: x["person_bbox"][2] * x["person_bbox"][3])
            body_crop = best_body["body_crop"]
            if face_crop is None and best_body.get("face_crop") is not None:
                face_crop = best_body["face_crop"]
                
        # If Body crop fails totally, use full image as the "body" for features
        if body_crop is None or body_crop.size == 0:
            body_crop = img_bgr

        # Store Face Embedding
        if face_crop is not None and face_crop.size > 0:
            f_emb = extractor.extract(face_crop, masked=False)
            norm = np.linalg.norm(f_emb)
            if norm > 0:
                f_emb = f_emb / norm
            face_embeddings.append(f_emb.flatten())
            face_labels_int.append(label_int)
            
        # Store Body Embedding
        b_emb = body_extractor.extract(body_crop)
        norm_b = np.linalg.norm(b_emb)
        if norm_b > 0: b_emb = b_emb / norm_b
        body_embeddings.append(b_emb.flatten())
        body_labels_int.append(label_int)
        
        # Store Attribute Embedding
        a_emb = attr_extractor.extract(body_crop)
        norm_a = np.linalg.norm(a_emb)
        if norm_a > 0: a_emb = a_emb / norm_a
        attr_embeddings.append(a_emb.flatten())
        attr_labels_int.append(label_int)
        
        total_images_processed += 1
        
    os.makedirs("embeddings", exist_ok=True)
        
    logger.info("Storing labels into embeddings/")
    with open("embeddings/labels.pkl", "wb") as f:
        pickle.dump(label_map, f)
        
    print(f"Number of persons: {len(person_to_int)}")
    print(f"Images processed: {total_images_processed}")
    
    # helper for generating flat indices cleanly
    def build_faiss(emb_list, name):
        if not emb_list:
            logger.warning(f"No embeddings generated for {name}.")
            return
        
        # Ensure all elements are 1D arrays of the same length
        filtered = [np.array(e).flatten() for e in emb_list]
        mat = np.array(filtered, dtype=np.float32)
        dim = mat.shape[1]
        
        idx = faiss.IndexFlatL2(dim)
        idx.add(mat)
        faiss.write_index(idx, f"embeddings/{name}.index")
        print(f"[{name}] Index Dimensions: {dim} -> Size: {idx.ntotal}")

    logger.info("Building Face FAISS index...")
    build_faiss(face_embeddings, "faiss_index")
    
    logger.info("Building Body FAISS index...")
    build_faiss(body_embeddings, "body_faiss")
    
    logger.info("Building Attribute FAISS index...")
    build_faiss(attr_embeddings, "attr_faiss")
    
    # Save the parallel label arrays
    with open("embeddings/multi_labels.pkl", "wb") as f:
        pickle.dump({
            "face_labels": face_labels_int,
            "body_labels": body_labels_int,
            "attr_labels": attr_labels_int
        }, f)
        
    logger.info("Multi-Modal Pipeline Complete.")

if __name__ == "__main__":
    run_training_pipeline()
