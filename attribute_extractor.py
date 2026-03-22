"""
attribute_extractor.py
======================
Extracts physical 'attributes' (clothing color histograms and structural HOG features)
from an upper-body image crop, generating a normalized 1D embedding vector.

This operates independently of deep learning backends, leveraging classical
computer vision (HSV Spaces + HOG gradients) to serve as a fast biometric
fallback during masked or occluded facial recognition events.

Usage
-----
    from attribute_extractor import AttributeExtractor
    import cv2
    
    extractor = AttributeExtractor()
    body_img = cv2.imread("body.jpg")
    attr_vec = extractor.extract(body_img)
"""

import cv2
import numpy as np
import logging
from config import LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class AttributeExtractor:
    """
    Generates a fixed-size normalized vector representing clothing color distribution 
    (HSV Hue/Sat Histograms) and structural textures (HOG grids).
    """
    
    def __init__(self, target_size=(128, 128), bins=(16, 16)) -> None:
        self.target_size = target_size
        self.bins = bins
        
        # Initialize OpenCV HOG Descriptor
        win_size = (128, 128)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    def extract(self, body_bgr: np.ndarray) -> np.ndarray:
        """
        Calculates and concatenates HSV histograms and HOG descriptors.
        
        Parameters
        ----------
        body_bgr : np.ndarray
            OpenCV BGR upper-body crop.
            
        Returns
        -------
        embedding : np.ndarray
            L2-normalized 1D float32 array.
        """
        if body_bgr is None or body_bgr.size == 0:
            # Fallback zero-vector if empty
            # HOG len = 1764, Hist len = 16*16 = 256. Total = 2020.
            return np.zeros((2020,), dtype=np.float32)
            
        img = cv2.resize(body_bgr, self.target_size)
        
        # 1. Color Attributes (HSV Histogram)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Compute histogram on Hue and Saturation
        hist = cv2.calcHist([hsv], [0, 1], None, self.bins, [0, 180, 0, 256])
        # Flatten and normalize locally
        cv2.normalize(hist, hist)
        hist_features = hist.flatten()
        
        # 2. Structural/Texture Attributes (HOG)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = self.hog.compute(gray)
        if hog_features is not None:
            hog_features = hog_features.flatten()
        else:
            hog_features = np.zeros((1764,), dtype=np.float32)
            
        # 3. Concatenate and L2 Normalize Globally
        embedding = np.concatenate([hist_features, hog_features], axis=0).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

if __name__ == "__main__":
    extractor = AttributeExtractor()
    dummy = np.zeros((200, 100, 3), dtype=np.uint8)
    vec = extractor.extract(dummy)
    print(f"Attribute Embedding Shape: {vec.shape}")
    print(f"L2 Norm: {np.linalg.norm(vec):.4f}")
