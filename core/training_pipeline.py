"""
training_pipeline.py
====================
Build FAISS indexes from dataset/train. Cloud-lite safe (no TensorFlow/ResNet50).
"""

from __future__ import annotations

import logging
import os
import pickle

import cv2
import faiss
import numpy as np

from config import CLOUD_LITE, ROOT_DIR
from dataset_loader import DatasetLoader
from feature_extractor import FeatureExtractor
from attribute_extractor import AttributeExtractor

logger = logging.getLogger(__name__)


def _data_root() -> str:
    return os.environ.get("DATA_DIR", ROOT_DIR)


def _embeddings_dir() -> str:
    path = os.path.join(_data_root(), "embeddings")
    os.makedirs(path, exist_ok=True)
    return path


def run_training_pipeline() -> dict:
    """
    Train face + attribute FAISS indexes from dataset/train only.
    In cloud lite mode skips ResNet50 body embeddings.
    """
    from face_alignment import create_aligner
    from yolo_person_detector import YoloPersonDetector, preload_yolo_model

    loader = DatasetLoader()
    train_dir = os.path.join(_data_root(), "dataset", "train")
    if not os.path.isdir(train_dir) or not os.listdir(train_dir):
        raise FileNotFoundError(
            f"No training images at {train_dir}. "
            "Upload dataset/train to your Railway volume (DATA_DIR/dataset/train)."
        )

    preload_yolo_model()
    aligner = create_aligner(min_confidence=0.40)
    yolo_detector = YoloPersonDetector(aligner=aligner)
    extractor = FeatureExtractor()
    attr_extractor = AttributeExtractor()
    body_extractor = None
    if not CLOUD_LITE:
        from body_feature_extractor import BodyFeatureExtractor
        body_extractor = BodyFeatureExtractor()

    face_embeddings: list[np.ndarray] = []
    face_labels_int: list[int] = []
    body_embeddings: list[np.ndarray] = []
    body_labels_int: list[int] = []
    attr_embeddings: list[np.ndarray] = []
    attr_labels_int: list[int] = []
    person_to_int: dict[str, int] = {}
    label_map: dict[int, str] = {}
    current_label_id = 0
    total_images_processed = 0

    logger.info("Training from dataset/train at %s (cloud_lite=%s)", train_dir, CLOUD_LITE)

    for person_id, img_path, img_bgr in loader.load_training_data():
        if person_id not in person_to_int:
            person_to_int[person_id] = current_label_id
            label_map[current_label_id] = person_id
            current_label_id += 1

        label_int = person_to_int[person_id]
        face_crop = None
        body_crop = None

        upscaled = cv2.resize(img_bgr, None, fx=1.5, fy=1.5)
        faces = aligner.align(upscaled)
        if faces:
            best_face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
            face_crop = best_face["aligned_crop"]

        yolo_res = yolo_detector.detect(img_bgr)
        if yolo_res:
            best_body = max(yolo_res, key=lambda x: x["person_bbox"][2] * x["person_bbox"][3])
            body_crop = best_body["body_crop"]
            if face_crop is None and best_body.get("face_crop") is not None:
                face_crop = best_body["face_crop"]

        if body_crop is None or body_crop.size == 0:
            body_crop = img_bgr

        if face_crop is not None and face_crop.size > 0:
            f_emb = extractor.extract(face_crop, masked=False)
            norm = np.linalg.norm(f_emb)
            if norm > 0:
                f_emb = f_emb / norm
            face_embeddings.append(f_emb.flatten())
            face_labels_int.append(label_int)

        if body_extractor is not None:
            b_emb = body_extractor.extract(body_crop)
            norm_b = np.linalg.norm(b_emb)
            if norm_b > 0:
                b_emb = b_emb / norm_b
            body_embeddings.append(b_emb.flatten())
            body_labels_int.append(label_int)

        a_emb = attr_extractor.extract(body_crop)
        norm_a = np.linalg.norm(a_emb)
        if norm_a > 0:
            a_emb = a_emb / norm_a
        attr_embeddings.append(a_emb.flatten())
        attr_labels_int.append(label_int)

        total_images_processed += 1

    emb_dir = _embeddings_dir()

    with open(os.path.join(emb_dir, "labels.pkl"), "wb") as f:
        pickle.dump(label_map, f)

    def build_faiss(emb_list: list[np.ndarray], name: str) -> int:
        if not emb_list:
            logger.warning("No embeddings generated for %s.", name)
            return 0
        mat = np.array([np.array(e).flatten() for e in emb_list], dtype=np.float32)
        idx = faiss.IndexFlatL2(mat.shape[1])
        idx.add(mat)
        faiss.write_index(idx, os.path.join(emb_dir, f"{name}.index"))
        return idx.ntotal

    face_count = build_faiss(face_embeddings, "faiss_index")
    body_count = build_faiss(body_embeddings, "body_faiss") if body_embeddings else 0
    attr_count = build_faiss(attr_embeddings, "attr_faiss")

    with open(os.path.join(emb_dir, "multi_labels.pkl"), "wb") as f:
        pickle.dump({
            "face_labels": face_labels_int,
            "body_labels": body_labels_int,
            "attr_labels": attr_labels_int,
        }, f)

    return {
        "persons": len(person_to_int),
        "images_processed": total_images_processed,
        "face_index_size": face_count,
        "body_index_size": body_count,
        "attr_index_size": attr_count,
        "embeddings_dir": emb_dir,
        "cloud_lite": CLOUD_LITE,
    }
