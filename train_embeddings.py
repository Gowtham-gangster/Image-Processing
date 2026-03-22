import os
import faiss
import pickle
import logging
import numpy as np

from dataset_loader import DatasetLoader
from face_detector import FaceDetector
from feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training_pipeline() -> None:
    loader = DatasetLoader()
    detector = FaceDetector()
    extractor = FeatureExtractor()
    
    embeddings = []
    labels_int = []
    
    person_to_int = {}
    label_map = {}
    current_label_id = 0
    
    person_counts = {}
    total_images_processed = 0
    
    logger.info("Step 1: Loading dataset from dataset/train only")
    for person_id, img_path, img_bgr in loader.load_training_data():
        print(f"Processing person: {person_id}")
        
        boxes = detector.detect_faces(img_bgr)
        if not boxes:
            continue
            
        x, y, w, h = boxes[0]
        face_crop = img_bgr[y:y+h, x:x+w]
        
        emb = extractor.extract(face_crop, masked=False)
        
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
            
        emb_vector = emb.flatten()
        
        if person_id not in person_to_int:
            person_to_int[person_id] = current_label_id
            label_map[current_label_id] = person_id
            current_label_id += 1
            
        person_counts[person_id] = person_counts.get(person_id, 0) + 1
        total_images_processed += 1
            
        embeddings.append(emb_vector)
        labels_int.append(person_to_int[person_id])
        print("Embedding stored")
        
    if not embeddings:
        logger.error("No faces found inside the dataset/train directory.")
        return
        
    os.makedirs("embeddings", exist_ok=True)
        
    logger.info("Storing embeddings and labels into embeddings/")
    with open("embeddings/embeddings.pkl", "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels_int}, f)
        
    print(f"Number of persons: {len(person_to_int)}")
    print(f"Number of images processed: {total_images_processed}")
    print(f"Number of embeddings stored: {len(embeddings)}")
        
    logger.info("Building FAISS index...")
    db_matrix = np.array(embeddings, dtype=np.float32)
    dim = db_matrix.shape[1]
    
    faiss_idx = faiss.IndexFlatL2(dim)
    faiss_idx.add(db_matrix)
    faiss.write_index(faiss_idx, "embeddings/faiss_index.index")
    print(f"FAISS index size: {faiss_idx.ntotal}")
    
    with open("embeddings/labels.pkl", "wb") as f:
        pickle.dump(label_map, f)
        
    logger.info("Pipeline Complete. Model files saved efficiently to embeddings/ directory.")

if __name__ == "__main__":
    run_training_pipeline()
