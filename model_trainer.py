"""
model_trainer.py
================
[DEPRECATED] Legacy LBPH Trainer.

The LBPH (Local Binary Pattern Histogram) architecture has been deprecated in favor of the Advanced Hybrid Pipeline using FaceNet + SVM/KNN and DeepSORT Tracking.

This script now redirects to `train_model.py`.

Usage
-----
    python model_trainer.py
"""

import logging
from config import LOG_LEVEL
from train_model import train_classifier

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

def train(*args, **kwargs) -> None:
    logger.warning("LBPH is deprecated. Redirecting to Deep Learning Multi-Modal Pipeline (train_model.py).")
    train_classifier()

if __name__ == "__main__":
    train()
