# Mask-Aware Hybrid Person Identification System for CCTV Surveillance

A Python-based machine learning system that detects, identifies, and profiles
individuals from CCTV footage — even when faces are masked or partially occluded.

---

## 📁 Project Structure

```
MaskAwareHybridID/
│
├── api/                      ← Vercel serverless entrypoint (index.py)
├── dataset/                  ← Training images, test images, and persons.csv
├── models/                   ← Saved models (DNN weights, LBPH, ONNX engines)
├── logs/                     ← Detections, logs, events, and metrics
├── scripts/                  ← Training, evaluation, and DB migration tools
├── tests/                    ← API and System integration tests
├── dashboard/                ← React dashboard frontend
├── Dockerfile                ← Docker deployment config
├── docker-compose.yml        ← Local multi-container arrangement
├── vercel.json               ← Vercel deployment config (API routing)
├── requirements.txt          ← Core Python dependencies
├── SURVEILLANCE_APPS.py      ← (realtime.py, surveillance_app.py, enrollment_gui.py)
└── CORE_ENGINE.py            ← (detection.py, mask_detector.py, extraction.py, etc.)
```

---

## ⚙️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### Note on Cloud Deployment
Due to the massive size of machine learning dependencies (`tensorflow`, `ultralytics`, `opencv` sum up to ~9GB), **Serverless platforms like Vercel or AWS Lambda are inherently incompatible**.
You **must** deploy this backend using **Docker** on platforms like Render, Railway, or standard Virtual Machines. A `Dockerfile` and `docker-compose.yml` are provided in the repo.

> **Note:** `tensorflow` is optional. If not installed, the mask detector
> automatically falls back to a skin-tone heuristic.

### 2. Download DNN Face Detector Weights (Recommended)

Place these two files in the `models/` directory:

| File | URL |
|------|-----|
| `deploy.prototxt` | [OpenCV GitHub](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt) |
| `res10_300x300_ssd_iter_140000.caffemodel` | [OpenCV 3rd Party](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel) |

> Without these files, the system automatically uses the **Haar Cascade** fallback.

---

## 🚀 Quickstart

### Step 1 — Add Training Images

```
dataset/train/
    person1/
        img001.jpg
        img002.jpg
        ...
    person2/
        img001.jpg
        ...
```

> Each sub-folder name **must match** a `person_id` in `dataset/persons.csv`.

### Step 2 — Update persons.csv

```csv
person_id,name,gender,age,phone,address
person1,Arjun Sharma,Male,28,+91-9876543210,"42 MG Road, Bengaluru"
person2,Priya Nair,Female,34,+91-9765432109,"15 Anna Salai, Chennai"
```

### Step 3 — Train the Model

```bash
python model_trainer.py
```

This creates `models/recognizer.yml` and `models/label_map.pkl`.

### Step 4 — Run Surveillance

```bash
# Live webcam
python realtime.py

# Second camera
python realtime.py --source 1

# Video file
python realtime.py --source cctv.mp4

# Classic surveillance app (image / video / camera)
python surveillance_app.py
python surveillance_app.py --source image --input path/to/image.jpg
python surveillance_app.py --source video --input path/to/cctv.mp4

# Headless (no window)
python surveillance_app.py --source video --input cctv.mp4 --no-display
```

### Step 5 — Evaluate Accuracy

```bash
# LBPH-based evaluation (fast, uses older pipeline)
python evaluate.py

# Full sklearn evaluation with confusion matrix (MobileNetV2 pipeline)
python evaluate_model.py

# Embedding-based evaluator (MTCNN + FAISS + FaceNet)
python evaluation.py

# With custom thresholds
python evaluate_model.py --clf-thresh 0.70 --sim-thresh 0.65 --no-plot
python evaluation.py --thresh 0.70 --no-plot
```

---

## 🖥️ CLI Entry Points

| Script | Purpose |
|--------|---------|
| `realtime.py` | Live webcam / CCTV recognition with HUD overlay |
| `surveillance_app.py` | Image / video / camera surveillance app |
| `scripts/train_model.py` | Train SVM or KNN classifier on embeddings |
| `scripts/model_trainer.py` | Train LBPH recognizer (simpler pipeline) |
| `scripts/evaluate.py` | LBPH accuracy evaluation on `dataset/test/` |
| `scripts/evaluate_model.py` | Full sklearn metrics + confusion matrix |
| `scripts/evaluation.py` | CNN embedding evaluator (MTCNN + FAISS) |
| `tests/test_system.py` | End-to-end test with per-image output |
| `enrollment_gui.py` | Tkinter GUI for enrolling new identities |
| `scripts/embeddings.py` | Extract and store MobileNetV2 embeddings |
| `detection.py` | Standalone face / body detector demo |
| `recognition.py` | End-to-end recognition pipeline demo |
| `attributes.py` | Query person attributes from persons.csv |

---

## 🧠 System Architecture

```
Input (Image / Video / Camera)
         │
         ▼
┌─────────────────────┐
│   Face Detection     │  OpenCV DNN → YOLO → Haar Cascade fallback
└────────┬────────────┘
         │  (x, y, w, h)
         ▼
┌─────────────────────┐
│  Face Alignment      │  MTCNN → affine warp to 224×224
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Mask Detection     │  Keras model → Skin-tone heuristic fallback
└────────┬────────────┘
         │  (is_masked, confidence)
         ▼
┌─────────────────────┐
│ Feature Extraction   │  MobileNetV2 / FaceNet 128-D embeddings
│                      │  Fallback: HOG + LBP (LBPH pipeline)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  FAISS / SVM / KNN  │  Cosine similarity / classifier predict
└────────┬────────────┘
         │ person_id, confidence
         ▼
┌─────────────────────┐
│   Database Lookup   │  persons.csv → name, age, gender, phone, address
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Display / Log     │  Annotated frame + detections.csv
└─────────────────────┘
```

---

## 🎛️ Configuration

All tunable parameters are in **`config.py`**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FACE_CONF_THRESHOLD` | `0.5` | DNN detector minimum confidence |
| `MASK_CONF_THRESHOLD` | `0.6` | Mask classifier minimum confidence |
| `RECOGNIZER_THRESHOLD` | `80` | LBPH distance cutoff (lower = stricter) |
| `FACE_SIZE` | `(100, 100)` | Resize target for recognition |
| `CAMERA_SOURCE` | `0` | Webcam index |

---

## 📊 Output

- **Known person detected** → Green bounding box + attribute panel (name, ID,
  gender, age, phone, address).
- **Unknown person detected** → Red bounding box + "Unknown Person" label.
- **Masked person** → Orange bounding box + `[MASKED]` label.
- All detections are appended to `logs/detections.csv` automatically.

---

## 🔧 Extending the System

| Goal | Files to modify |
|------|----------------|
| Add a new person | Add images to `dataset/train/<id>/`, add row to `persons.csv`, re-run `model_trainer.py` |
| Swap face detector | Edit `face_detector.py`, or subclass `FaceDetector` |
| Use a custom mask model | Train a Keras binary classifier and save to `models/mask_detector.model` |
| Use deep embeddings | Extend `feature_extractor.py` with a MobileNet/FaceNet backbone |
| REST API | Wrap `SurveillanceSystem.process_frame()` in a Flask/FastAPI endpoint |
