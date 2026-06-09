"""
config.py
=========
Central configuration for Mask-Aware Hybrid Person Identification System.
All paths, thresholds, and model parameters are defined here.
"""

import os

# ── Project root ────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset paths ───────────────────────────────────────────────────────────
DATASET_DIR       = os.path.join(ROOT_DIR, "dataset")
TRAIN_DIR         = os.path.join(DATASET_DIR, "train")
TEST_DIR          = os.path.join(DATASET_DIR, "test")
IS_VERCEL         = os.environ.get("VERCEL") == "1"
PERSONS_DB        = os.path.join("/tmp", "persons.db") if IS_VERCEL else os.path.join(DATASET_DIR, "persons.db")


# ── Model / asset paths ─────────────────────────────────────────────────────
MODELS_DIR        = os.path.join(ROOT_DIR, "models")
RECOGNIZER_PATH   = os.path.join(MODELS_DIR, "recognizer.yml")   # LBPH model
LABEL_MAP_PATH    = os.path.join(MODELS_DIR, "label_map.pkl")    # id → person_id

# OpenCV DNN face detector (download separately – see README)
FACE_PROTO        = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_MODEL        = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Mask detector model (Keras / TF SavedModel format)
MASK_MODEL_PATH   = os.path.join(MODELS_DIR, "mask_detector.model")

# ── ONNX / TensorRT optimised models ─────────────────────────────────
ONNX_DIR             = os.path.join(MODELS_DIR, "onnx")
ONNX_EMBEDDER_PATH   = os.path.join(ONNX_DIR, "openface_embedder.onnx")
ONNX_MASK_PATH       = os.path.join(ONNX_DIR, "mask_detector.onnx")
ONNX_DETECTOR_PATH   = os.path.join(ONNX_DIR, "ssd_face_detector.onnx")

# TensorRT compiled engines (generated on the target device e.g. Jetson Nano)
TRT_DIR              = os.path.join(MODELS_DIR, "trt")
TRT_EMBEDDER_ENGINE  = os.path.join(TRT_DIR, "openface_embedder.engine")
TRT_MASK_ENGINE      = os.path.join(TRT_DIR, "mask_detector.engine")
TRT_DETECTOR_ENGINE  = os.path.join(TRT_DIR, "ssd_face_detector.engine")

# Inference backend selection — set to "onnx" or "trt" for edge deployment
# "original" uses cv2.dnn / Keras (development default)
INFERENCE_BACKEND    = os.environ.get("MASKAWARE_BACKEND", "original")

# Logs
LOGS_DIR          = os.path.join("/tmp", "logs") if IS_VERCEL else os.path.join(ROOT_DIR, "logs")
LOG_FILE          = os.path.join(LOGS_DIR, "surveillance.log")

# ── Events / Alerts ──────────────────────────────────────────────────────────
EVENTS_DB         = os.path.join(LOGS_DIR, "events.db")
ALERT_CONFIG_PATH = os.path.join("/tmp", "alerts_config.json") if IS_VERCEL else os.path.join(ROOT_DIR, "alerts_config.json")

# ── Detection / recognition parameters ─────────────────────────────────────
FACE_CONF_THRESHOLD   = 0.5    # DNN face detector confidence threshold
MASK_CONF_THRESHOLD   = 0.6    # Mask classifier confidence threshold
RECOGNIZER_THRESHOLD  = 80     # LBPH distance threshold (lower = stricter)

# Image pre-processing
FACE_SIZE             = (100, 100)   # Resize faces before LBPH training
DNN_INPUT_SIZE        = (300, 300)   # DNN blob size

# Camera / video source  (0 = default webcam, or path to video file)
CAMERA_SOURCE         = 0
FRAME_SCALE           = 1.0          # Scale factor for display window

# ── Display colours (BGR) ───────────────────────────────────────────────────
COLOUR_KNOWN    = (0, 255, 0)     # Green  – identified person
COLOUR_UNKNOWN  = (0, 0, 255)     # Red    – unknown person
COLOUR_MASKED   = (255, 165, 0)   # Orange – masked person
FONT            = 0               # cv2.FONT_HERSHEY_SIMPLEX

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"   # DEBUG | INFO | WARNING | ERROR

# ── Ensure output directories exist ─────────────────────────────────────────
for _d in [MODELS_DIR, LOGS_DIR]:
    try:
        os.makedirs(_d, exist_ok=True)
    except OSError:
        pass
