"""
body_feature_extractor.py
=========================
Feature extraction for upper-body/clothing appearance using a pretrained CNN.

This module uses ResNet50 (pretrained on ImageNet) to extract high-level 
visual features (e.g., clothing color, texture, shape) from body crops.
These features are highly useful as a secondary biometric when the face 
is heavily masked or occluded.

Outputs a 2048-dimensional L2-normalized embedding vector.

Usage
-----
    from body_feature_extractor import BodyFeatureExtractor
    import cv2

    extractor = BodyFeatureExtractor()
    body_image = cv2.imread("upper_body.jpg")

    # Outputs a (2048,) numpy float32 array
    vector = extractor.extract(body_image)
"""

import os
import logging
import numpy as np
import cv2

from config import LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class BodyFeatureExtractor:
    """
    Extracts 2048-Dimensional embeddings from body images using ResNet50.
    """

    def __init__(self) -> None:
        logger.info("Initializing BodyFeatureExtractor (ResNet50)...")
        try:
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet50 import preprocess_input
            
            # Load ResNet50 without the classification head, using average pooling
            # This yields a 2048-D vector per image.
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input
            self.target_size = (224, 224)
            logger.info("ResNet50 model loaded successfully.")
            
        except ImportError:
            logger.error(
                "TensorFlow/Keras is required for BodyFeatureExtractor.\n"
                "Install it via: pip install tensorflow"
            )
            raise

    def extract(self, body_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a BGR body crop into a 2048-D L2-normalized embedding.

        Parameters
        ----------
        body_bgr : np.ndarray
            OpenCV BGR image array containing the upper body.

        Returns
        -------
        embedding : np.ndarray
            A 1D numpy array of shape (2048,) containing float32 values.
        """
        if body_bgr is None or body_bgr.size == 0:
            raise ValueError("Empty image provided to body feature extractor.")

        # Resize to ResNet50 expected input size
        body_resized = cv2.resize(body_bgr, self.target_size)
        
        # Convert BGR to RGB
        body_rgb = cv2.cvtColor(body_resized, cv2.COLOR_BGR2RGB)
        
        # Expand dimensions to create a batch of 1: (1, 224, 224, 3)
        img_arr = np.expand_dims(body_rgb, axis=0).astype(np.float32)
        
        # Apply ResNet50 specific preprocessing (e.g., mean subtraction)
        img_preprocessed = self.preprocess_input(img_arr)
        
        # Forward pass
        embedding = self.model.predict(img_preprocessed, verbose=0)
        
        # Flatten to 1D (2048,)
        embedding = embedding.flatten()
        
        # L2 Normalize the vector to ensure proper cosine similarity behaviour
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.astype(np.float32)

if __name__ == "__main__":
    # Quick test
    try:
        extractor = BodyFeatureExtractor()
        dummy_img = np.zeros((300, 150, 3), dtype=np.uint8)
        vec = extractor.extract(dummy_img)
        print(f"Generated body embedding shape: {vec.shape}")
        print(f"Embedding L2 norm: {np.linalg.norm(vec):.4f}")
    except ImportError:
        pass
