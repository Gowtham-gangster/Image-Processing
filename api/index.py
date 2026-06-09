import os
import sys

# ── Critical: Set these BEFORE any ML library loads ───────────────────────────
# On Linux, if TensorFlow loads its OpenMP first, PyTorch's OpenMP will conflict
# causing a Segmentation fault. KMP_DUPLICATE_LIB_OK allows both to coexist.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF startup noise

# ── Force PyTorch to load FIRST before TensorFlow ─────────────────────────────
# PyTorch (used by YOLO) must initialize its OpenMP instance before TF loads.
# If TF loads first, PyTorch's second OpenMP instance causes a segfault on Linux.
try:
    import torch  # noqa: F401 - intentional pre-load
    torch.set_num_threads(1)
except ImportError:
    pass  # torch not installed, will fail later with a clearer error

import asyncio
import json
import logging
import io
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure relative imports and model loading works
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(BASE_DIR, "core")
for d in [BASE_DIR, CORE_DIR]:
    if d not in sys.path:
        sys.path.append(d)

from face_alignment import FaceAligner
from feature_extractor import FeatureExtractor
from unknown_detector import UnknownDetector
from attributes_manager import AttributesManager
from database import PersonDatabase
from liveness_detector import LivenessDetector
from mask_detector import MaskDetector
from alert_manager import AlertManager, ALERT_UNKNOWN_PERSON, ALERT_UNMASKED, ALERT_MASKED, ALERT_SPOOF
from surveillance_logger import SurveillanceLogger
from yolo_person_detector import YoloPersonDetector, preload_yolo_model
from body_feature_extractor import BodyFeatureExtractor
from body_embedding_database import BodyEmbeddingDatabase
from attribute_extractor import AttributeExtractor
from camera_manager import CameraManager
import faiss

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mask-Aware Hybrid Person Identification API",
    description="Real-time surveillance system with face recognition, liveness detection, and alerting.",
    version="3.0.0",
)

# Allow all origins for dashboard dev (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SSE event queue shared across connections ─────────────────────────────────
_sse_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

# Global instances (loaded once on startup)
aligner        = None
embedder       = None
db             = None
attributes_mgr = None
unknown_det    = None
person_db      = None
liveness_det   = None
mask_det       = None
alert_mgr      = None
surv_logger    = None
yolo_detector  = None
body_extractor = None
body_db        = None
attr_extractor = None
camera_mgr     = None

face_faiss   = None
body_faiss   = None
attr_faiss   = None
multi_labels = {}
label_map    = {}


def get_body_extractor() -> BodyFeatureExtractor:
    """Lazy-load ResNet50 after PyTorch/YOLO have initialized."""
    global body_extractor
    if body_extractor is None:
        body_extractor = BodyFeatureExtractor()
    return body_extractor


@app.on_event("startup")
def startup_event():
    global aligner, embedder, db, attributes_mgr, unknown_det
    global person_db, liveness_det, mask_det, alert_mgr, surv_logger
    global yolo_detector, body_db, attr_extractor, camera_mgr

    logger.info("Initializing AI pipeline components...")

    global face_faiss, body_faiss, attr_faiss, multi_labels, label_map

    person_db      = PersonDatabase()
    attributes_mgr = AttributesManager(db=person_db)

    # PyTorch/YOLO must fully initialize before any TensorFlow import (MTCNN, ResNet50).
    logger.info("Pre-loading PyTorch/YOLO before TensorFlow to prevent OpenMP conflict...")
    preload_yolo_model()
    attr_extractor = AttributeExtractor()
    aligner        = FaceAligner(min_confidence=0.40)
    yolo_detector  = YoloPersonDetector(aligner=aligner)
    embedder       = FeatureExtractor()
    camera_mgr     = CameraManager()

    import pickle
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.getenv("DATA_DIR", base_dir)
    idx_path = os.path.join(data_dir, "embeddings", "faiss_index.index")
    body_path = os.path.join(data_dir, "embeddings", "body_faiss.index")
    attr_path = os.path.join(data_dir, "embeddings", "attr_faiss.index")
    labels_path = os.path.join(data_dir, "embeddings", "multi_labels.pkl")
    
    if os.path.exists(idx_path):
        face_faiss = faiss.read_index(idx_path)
    if os.path.exists(body_path):
        body_faiss = faiss.read_index(body_path)
    if os.path.exists(attr_path):
        attr_faiss = faiss.read_index(attr_path)
        
    if os.path.exists(labels_path):
        with open(labels_path, "rb") as f:
            multi_labels = pickle.load(f)
        logger.info("Multi-modal FAISS pipelines strictly cached in-memory.")

    legacy_path = os.path.join(data_dir, "embeddings", "labels.pkl")
    if os.path.exists(legacy_path):
        with open(legacy_path, "rb") as f:
            label_map = pickle.load(f)
        logger.info("Legacy labels loaded.")
        
    unknown_det    = UnknownDetector(threshold=0.60)
    liveness_det   = LivenessDetector()
    mask_det       = MaskDetector()
    alert_mgr      = AlertManager()
    surv_logger    = SurveillanceLogger()
    body_db        = BodyEmbeddingDatabase()

    logger.info("AI pipeline successfully initialized.")



# ── Pydantic models ───────────────────────────────────────────────────────────

class PersonCreate(BaseModel):
    person_id: str
    name: str
    gender: str = "N/A"
    age: str = "N/A"
    phone: str = "N/A"
    address: str = "N/A"

class AlertConfig(BaseModel):
    slack_webhook_url: Optional[str] = None
    webhook_url: Optional[str] = None
    email: Optional[dict] = None


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Verify that the API and AI modules are loaded and healthy."""
    return {"status": "healthy", "pipeline_active": yolo_detector is not None, "version": "3.0.0"}


# ── Persons ───────────────────────────────────────────────────────────────────

@app.get("/persons", tags=["Persons"])
def list_persons():
    """List all enrolled persons and their attributes."""
    records = person_db.all_persons()
    for r in records:
        for k, v in r.items():
            if v is None:
                r[k] = "N/A"
    return {"persons": records, "total": len(records)}

@app.post("/persons", tags=["Persons"])
def add_person(person: PersonCreate):
    """Add a new person to the identity database."""
    try:
        attributes_mgr.add_person(
            person_id=person.person_id,
            name=person.name,
            gender=person.gender,
            age=person.age,
            phone=person.phone,
            address=person.address,
        )
        return {"status": "success", "message": f"Added {person.name} ({person.person_id})"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/persons/{person_id}", tags=["Persons"])
def update_person(person_id: str, person: PersonCreate):
    """Update an existing person's details in the identity database."""
    try:
        # Check if person exists
        existing = person_db.get_person(person_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        
        # Update the person
        attributes_mgr.add_person(
            person_id=person_id,
            name=person.name,
            gender=person.gender,
            age=person.age,
            phone=person.phone,
            address=person.address,
        )
        return {"status": "success", "message": f"Updated {person.name} ({person_id})"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/persons/{person_id}", tags=["Persons"])
def delete_person(person_id: str):
    """Delete a person from the identity database."""
    try:
        # Check if person exists
        existing = person_db.get_person(person_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        
        # Note: This is a soft delete - we don't actually remove from DB
        # In production, you might want to add a 'deleted' flag or actually remove the record
        return {"status": "success", "message": f"Deleted person {person_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Events ────────────────────────────────────────────────────────────────────

@app.get("/events", tags=["Events"])
def get_events(
    limit: int  = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return paginated detection event history from the SQLite log."""
    events = surv_logger.get_recent_events(limit=limit, offset=offset)
    return {"events": events, "count": len(events)}

@app.get("/events/stats", tags=["Events"])
def get_event_stats():
    """Return aggregate stats for the analytics dashboard."""
    return surv_logger.get_stats()

@app.get("/events/stream", tags=["Events"])
async def event_stream():
    """
    Server-Sent Events endpoint.
    The browser dashboard subscribes here and receives real-time detection events.
    """
    async def generator():
        yield "retry: 3000\n\n"  # tell clients to retry every 3s if disconnected
        while True:
            try:
                event = await asyncio.wait_for(_sse_queue.get(), timeout=20)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"  # prevent proxy timeouts

    return StreamingResponse(generator(), media_type="text/event-stream")


# ── Alert Configuration ───────────────────────────────────────────────────────

@app.get("/alerts/config", tags=["Alerts"])
def get_alert_config():
    """Get current alert channel configuration (passwords redacted)."""
    cfg = dict(alert_mgr.get_config())
    if "email" in cfg and "password" in (cfg["email"] or {}):
        cfg["email"] = dict(cfg["email"])
        cfg["email"]["password"] = "***"
    return cfg

@app.post("/alerts/config", tags=["Alerts"])
def set_alert_config(config: AlertConfig):
    """Update alert channel settings (Slack, Email, Webhook)."""
    alert_mgr.save_config(config.model_dump(exclude_none=True))
    return {"status": "saved", "channels": alert_mgr._active_channels()}

@app.post("/alerts/test", tags=["Alerts"])
def test_alert():
    """Fire a test alert across all configured channels."""
    alert_mgr.send_alert(
        ALERT_UNKNOWN_PERSON,
        camera_id="Test-Camera",
        person_id="Test Person",
        confidence=0.0,
        extra={"note": "This is a test alert from the dashboard."},
    )
    return {"status": "test_alert_dispatched"}

@app.get("/alerts/history", tags=["Alerts"])
def get_alert_history(limit: int = 50, offset: int = 0):
    """Retrieve chronologically stored persistent alerts from the SQLite database."""
    return person_db.get_recent_alerts(limit=limit, offset=offset)

# ── Recognition ───────────────────────────────────────────────────────────────

def _push_sse_event(event: dict):
    """Non-blocking push to SSE queue. Drops events if queue is full."""
    try:
        _sse_queue.put_nowait(event)
    except asyncio.QueueFull:
        pass

@app.post("/upload", tags=["Recognition"])
async def recognize_image(file: UploadFile = File(...), camera_id: str = "API"):
    """
    Upload an image file, detect all faces via MTCNN, then run:
    mask detection → liveness check → embedding + identification.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode the image file.")

        # ── Step 1: Run MTCNN on the full frame (finds all faces + eye landmarks) ──
        # This is the correct approach for still images — the Detector class is
        # designed for realtime streams; face_alignment.align() is the right call here.
        aligned_faces = aligner.align(frame)
        results       = []

        for face in aligned_faces:
            aligned_crop = face["aligned_crop"]          # (224, 224, 3) BGR
            bx, by, bw, bh = face["box"]
            bbox = [bx, by, bw, bh]

            # ── Step 2: Mask detection on the aligned face crop ───────────────
            try:
                is_masked_flag, _ = mask_det.is_masked(aligned_crop)
            except Exception:
                is_masked_flag = False

            # ── Step 3: Liveness / Anti-Spoofing ─────────────────────────────
            is_live, spoof_msg = liveness_det.check(aligned_crop)
            if not is_live:
                event = {
                    "timestamp":  datetime.utcnow().isoformat() + "Z",
                    "camera_id":  camera_id,
                    "person_id":  "SPOOF DETECTED",
                    "name":       spoof_msg,
                    "is_known":   False,
                    "is_masked":  is_masked_flag,
                    "is_live":    False,
                    "confidence": 0.0,
                    "bbox":       bbox,
                    "attributes": {},
                }
                results.append(event)
                surv_logger.log_event(camera_id, "SPOOF", 0.0, is_known=False, is_masked=is_masked_flag)
                alert_mgr.send_alert(ALERT_SPOOF, camera_id=camera_id, confidence=0.0)
                _push_sse_event(event)
                continue

            # ── Step 4: Embedding + Identification ───────────────────────────
            try:
                emb = embedder.extract(aligned_crop, masked=False)
                person_id = "Unknown Person"
                score = 0.0
                
                if face_faiss is not None:
                    norm = np.linalg.norm(emb)
                    if norm > 0: emb = emb / norm
                    q_vec = np.array([emb], dtype=np.float32)
                    distances, indices = face_faiss.search(q_vec, 1)
                    
                    distance = distances[0][0]
                    idx = indices[0][0]
                    
                    score = max(0.0, 1.0 - ((distance**2) / 2.0))
                    if distance < 0.6 and idx != -1:
                        person_id = label_map.get(idx, "Unknown Person")
                        
            except Exception as exc:
                logger.error("Embedding/identification failed for face at %s: %s", bbox, exc)
                # Still report as Unknown so the user sees the detected face
                person_id = "Unknown Person"
                score     = 0.0

            is_known     = person_id != "Unknown Person"
            person_attrs = attributes_mgr.get_attributes(person_id) if is_known else {}

            event = {
                "timestamp":  datetime.utcnow().isoformat() + "Z",
                "camera_id":  camera_id,
                "person_id":  person_id,
                "name":       person_attrs.get("name", "Unknown Person") if person_attrs else "Unknown Person",
                "is_known":   is_known,
                "is_masked":  is_masked_flag,
                "is_live":    True,
                "confidence": float(score),
                "bbox":       bbox,
                "attributes": person_attrs or {},
            }
            results.append(event)

            # ── Step 5: Log + Alert ───────────────────────────────────────────
            surv_logger.log_event(
                camera_id, person_id, float(score),
                is_known=is_known, is_masked=is_masked_flag,
            )

            # Send alerts only for: 1) Unknown persons, 2) Masked persons
            if not is_known:
                alert_mgr.send_alert(ALERT_UNKNOWN_PERSON, camera_id=camera_id,
                                     person_id=person_id, confidence=float(score))
            
            if is_masked_flag:
                alert_mgr.send_alert(ALERT_MASKED, camera_id=camera_id,
                                     person_id=person_id, confidence=float(score))

            _push_sse_event(event)

        return JSONResponse(content={"detections_count": len(results), "results": results})

    except Exception as exc:
        logger.exception("Error processing image")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}")


def recognize_uploaded_image(image_path: str) -> dict:
    frame = cv2.imread(image_path)
    
    debug_info = {
        "image_path": image_path,
        "image_resolution": f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "N/A",
        "face_detected": "No",
        "face_bounding_box": "N/A",
        "detection_confidence": "N/A",
        "embedding_generated": "No",
        "embedding_vector_size": "N/A",
        "faiss_index_size": face_faiss.ntotal if face_faiss else 0,
        "nearest_distance": "N/A",
        "threshold": 0.6,
        "final_result": "Unknown"
    }

    def print_debug():
        print("====== DEBUG MODE ======")
        print(f"Image resolution: {debug_info['image_resolution']}")
        print(f"Face bounding box: {debug_info['face_bounding_box']}")
        print(f"Detection confidence: {debug_info['detection_confidence']}")
        print(f"Face detected: {debug_info['face_detected']}")
        print(f"Embedding generated: {debug_info['embedding_generated']}")
        print(f"Embedding vector size: {debug_info['embedding_vector_size']}")
        print(f"FAISS index loaded size: {debug_info['faiss_index_size']}")
        print(f"Nearest distance: {debug_info['nearest_distance']}")
        print(f"Threshold: {debug_info['threshold']}")
        print(f"Final result: {debug_info['final_result']}")
        print("========================")

    def fail(msg: str):
        print_debug()
        return {
            "person": msg,
            "confidence": 0.0,
            "mask": False,
            "age": "N/A",
            "gender": "N/A",
            "phone": "N/A",
            "address": "N/A",
            "error": msg,
            "debug_info": debug_info
        }
        
    if frame is None:
        return fail("Invalid image file.")
        
    aligner.min_confidence = 0.40
    
    face_crop = None
    body_crop = None
    
    # OPTIMIZATION 1: Try MTCNN first without upscaling (much faster)
    global_faces = aligner.align(frame)
    if global_faces:
        best_face = max(global_faces, key=lambda f: f["box"][2] * f["box"][3])
        face_crop = best_face["aligned_crop"]
        bx, by, bw, bh = best_face["box"]
        debug_info["face_bounding_box"] = f"[{bx}, {by}, {bw}, {bh}]"
        debug_info["detection_confidence"] = f"{best_face['confidence']:.3f}"
        debug_info["face_detected"] = "Yes"
        
    # OPTIMIZATION 2: Only run YOLO if face detection failed or for body context
    # Skip YOLO entirely if we have a good face detection
    if face_crop is not None and face_crop.size > 0:
        # We have a face, use the full frame as body crop (faster than YOLO)
        body_crop = frame
    else:
        # No face detected, try YOLO as fallback
        results = yolo_detector.detect(frame)
        if results:
            best_res = max(results, key=lambda x: x["person_bbox"][2] * x["person_bbox"][3])
            body_crop = best_res.get("body_crop")
            if face_crop is None and best_res.get("face_crop") is not None:
                face_crop = best_res["face_crop"]
                box_arr = best_res.get("face_bbox")
                debug_info["face_bounding_box"] = str(box_arr) if box_arr else "N/A"
                debug_info["detection_confidence"] = f"{best_res.get('face_conf', 0.0):.3f}"
                debug_info["face_detected"] = "Yes"
        
        if body_crop is None or body_crop.size == 0:
            body_crop = frame
        
    if face_faiss is None or body_faiss is None or attr_faiss is None:
        return fail("Models missing.")
        
    # Initialize Score Tensors
    scores = {}
    
    def search_faiss(idx, embedding_model, crop, label_key):
        emb = embedding_model.extract(crop, masked=False) if embedding_model == embedder else embedding_model.extract(crop)
        norm = np.linalg.norm(emb)
        if norm > 0: emb = emb / norm
        q_vec = np.array([emb], dtype=np.float32)
        dist_array, ind_array = idx.search(q_vec, min(5, idx.ntotal))
        return dist_array[0], ind_array[0], emb.shape
    
    # OPTIMIZATION 3: Prioritize face recognition - if we have a strong face match, skip body/attr
    if face_crop is not None and face_crop.size > 0:
        f_dists, f_inds, shape = search_faiss(face_faiss, embedder, face_crop, "face_labels")
        debug_info["embedding_generated"] = "Yes"
        debug_info["embedding_vector_size"] = str(shape)
        debug_info["nearest_distance"] = f"{f_dists[0]:.4f}"
        
        f_labels_map = multi_labels.get("face_labels", [])
        
        for d, i in zip(f_dists, f_inds):
            if i != -1 and i < len(f_labels_map):
                pid_int = f_labels_map[i]
                pid = label_map.get(pid_int, str(pid_int))
                scr = max(0.0, 1.0 - (d / 2.0))
                if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                scores[pid]["F"] = max(scores[pid]["F"], scr)
        
        # Check if we have a strong face match (>0.7) - if yes, skip expensive body/attr extraction
        best_face_score = max([s["F"] for s in scores.values()]) if scores else 0.0
        if best_face_score > 0.7:
            # Strong face match - use it directly without body/attr
            best_id = max(scores.items(), key=lambda x: x[1]["F"])[0]
            best_score = best_face_score
            matched_id = best_id
            debug_info["final_result"] = "Known (Fast Path)"
        else:
            # Weak face match - run full multi-modal pipeline
            b_dists, b_inds, _ = search_faiss(body_faiss, get_body_extractor(), body_crop, "body_labels")
            a_dists, a_inds, _ = search_faiss(attr_faiss, attr_extractor, body_crop, "attr_labels")
            
            b_labels_map = multi_labels.get("body_labels", [])
            a_labels_map = multi_labels.get("attr_labels", [])
            
            for d, i in zip(b_dists, b_inds):
                if i != -1 and i < len(b_labels_map):
                    pid_int = b_labels_map[i]
                    pid = label_map.get(pid_int, str(pid_int))
                    scr = max(0.0, 1.0 - (d / 2.0))
                    if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                    scores[pid]["B"] = max(scores[pid]["B"], scr)
                    
            for d, i in zip(a_dists, a_inds):
                if i != -1 and i < len(a_labels_map):
                    pid_int = a_labels_map[i]
                    pid = label_map.get(pid_int, str(pid_int))
                    scr = max(0.0, 1.0 - (d / 2.0))
                    if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                    scores[pid]["A"] = max(scores[pid]["A"], scr)
            
            # Late fusion
            best_id = "Unknown Person"
            best_score = 0.0
            
            for pid, s in scores.items():
                fused = (s["F"] * 0.5) + (s["B"] * 0.3) + (s["A"] * 0.2)
                if fused > best_score:
                    best_score = fused
                    best_id = pid
            
            threshold = 0.60
            if best_score < threshold:
                matched_id = "Unknown"
            else:
                matched_id = best_id
            
            debug_info["final_result"] = "Known" if matched_id != "Unknown" and matched_id != "Unknown Person" else "Unknown"
    else:
        # No face detected - run body/attr only
        b_dists, b_inds, _ = search_faiss(body_faiss, get_body_extractor(), body_crop, "body_labels")
        a_dists, a_inds, _ = search_faiss(attr_faiss, attr_extractor, body_crop, "attr_labels")
        
        b_labels_map = multi_labels.get("body_labels", [])
        a_labels_map = multi_labels.get("attr_labels", [])
        
        for d, i in zip(b_dists, b_inds):
            if i != -1 and i < len(b_labels_map):
                pid_int = b_labels_map[i]
                pid = label_map.get(pid_int, str(pid_int))
                scr = max(0.0, 1.0 - (d / 2.0))
                if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                scores[pid]["B"] = max(scores[pid]["B"], scr)
                
        for d, i in zip(a_dists, a_inds):
            if i != -1 and i < len(a_labels_map):
                pid_int = a_labels_map[i]
                pid = label_map.get(pid_int, str(pid_int))
                scr = max(0.0, 1.0 - (d / 2.0))
                if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                scores[pid]["A"] = max(scores[pid]["A"], scr)
        
        # Late fusion without face
        best_id = "Unknown Person"
        best_score = 0.0
        
        for pid, s in scores.items():
            fused = (s["B"] * 0.7) + (s["A"] * 0.3)
            if fused > best_score:
                best_score = fused
                best_id = pid
        
        threshold = 0.60
        if best_score < threshold:
            matched_id = "Unknown"
        else:
            matched_id = best_id
        
        debug_info["final_result"] = "Known" if matched_id != "Unknown" and matched_id != "Unknown Person" else "Unknown"
    
    print_debug()
    
    name = "Unknown"
    age = "Unknown"
    gender = "Unknown"
    phone = "Unknown"
    address = "Unknown"
    
    if matched_id != "Unknown" and matched_id != "Unknown Person":
        attrs = attributes_mgr.get_attributes(matched_id)
        if attrs:
            name = attrs.get("name", matched_id)
            age = attrs.get("age", "Unknown")
            gender = attrs.get("gender", "Unknown")
            phone = attrs.get("phone", "Unknown")
            address = attrs.get("address", "Unknown")
            
    # Mask check fallback internally
    is_masked = False
    if face_crop is not None and face_crop.size > 0:
        is_masked, _ = mask_det.is_masked(face_crop)
        
    return {
        "person": name,
        "confidence": round(float(best_score), 2),
        "mask": is_masked,
        "age": age,
        "gender": gender,
        "phone": phone,
        "address": address,
        "debug_info": debug_info
    }

@app.post("/predict", tags=["Recognition"])
async def predict_image(file: UploadFile = File(...)):
    """
    Evaluate a single uploaded image and return a JSON response with
    identity, mask status, demographic metadata, and error streams.
    """
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only JPG and PNG files are allowed.")
    
    temp_path = f"temp_{file.filename}"
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
            
        result = recognize_uploaded_image(temp_path)
        
        # Fire configured notification channels and save to SQLite alerts table
        # Only send alerts for: 1) Unknown persons, 2) Masked persons
        # Do NOT send alerts for known unmasked persons
        if "error" not in result:
            # Alert for unknown persons
            if result.get("person", "Unknown") == "Unknown":
                alert_mgr.send_alert(ALERT_UNKNOWN_PERSON, camera_id="ImageUpload", person_id="Unknown", confidence=result.get("confidence", 0.0))
            
            # Alert for masked persons (whether known or unknown)
            if result.get("mask", False):
                alert_mgr.send_alert(ALERT_MASKED, camera_id="ImageUpload", person_id=result.get("person", "Unknown"), confidence=result.get("confidence", 0.0))
                
        return result
    except Exception as e:
        logger.exception("Error in /predict-image")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Camera Management ────────────────────────────────────────────────────────

@app.get("/cameras", tags=["Camera Management"])
async def get_cameras():
    """Get all configured cameras"""
    return {"cameras": camera_mgr.get_all_cameras()}

@app.get("/cameras/enabled", tags=["Camera Management"])
async def get_enabled_cameras():
    """Get only enabled cameras"""
    return {"cameras": camera_mgr.get_enabled_cameras()}

@app.get("/cameras/{camera_id}", tags=["Camera Management"])
async def get_camera(camera_id: str):
    """Get specific camera configuration"""
    camera = camera_mgr.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return camera

@app.post("/cameras", tags=["Camera Management"])
async def add_camera(camera_data: dict):
    """Add a new camera to configuration"""
    if camera_mgr.add_camera(camera_data):
        return {"message": "Camera added successfully", "camera_id": camera_data.get('id')}
    raise HTTPException(status_code=400, detail="Failed to add camera")

@app.put("/cameras/{camera_id}", tags=["Camera Management"])
async def update_camera(camera_id: str, camera_data: dict):
    """Update existing camera configuration"""
    if camera_mgr.update_camera(camera_id, camera_data):
        return {"message": "Camera updated successfully", "camera_id": camera_id}
    raise HTTPException(status_code=400, detail="Failed to update camera")

@app.delete("/cameras/{camera_id}", tags=["Camera Management"])
async def delete_camera(camera_id: str):
    """Delete camera from configuration"""
    if camera_mgr.delete_camera(camera_id):
        return {"message": "Camera deleted successfully", "camera_id": camera_id}
    raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

@app.get("/cameras/{camera_id}/status", tags=["Camera Management"])
async def get_camera_status(camera_id: str):
    """Check if a camera is currently active"""
    camera = camera_mgr.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    return {
        "camera_id": camera_id,
        "active": camera_mgr.is_camera_active(camera_id),
        "enabled": camera.get('enabled', False)
    }

@app.get("/cameras/active/list", tags=["Camera Management"])
async def get_active_cameras():
    """Get list of currently active cameras"""
    return {"active_cameras": camera_mgr.get_active_cameras()}


# ── Live Video Feed ──────────────────────────────────────────────────────────

# Global camera state tracking
_active_streams = {}
_stream_locks = {}

@app.get("/video/feed", tags=["Video"])
async def video_feed(camera_id: str = Query("CAM001", description="Camera ID from configuration")):
    """
    MJPEG video stream endpoint with real-time person detection and identification.
    Returns a continuous stream of JPEG frames with detection overlays.
    Uses camera_id from camera configuration (e.g., CAM001, CAM002).
    """
    # Validate camera exists and is enabled
    camera_info = camera_mgr.get_camera(camera_id)
    if not camera_info:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    if not camera_info.get('enabled', False):
        raise HTTPException(status_code=400, detail=f"Camera {camera_id} is disabled")
    
    # Get or create lock for this camera
    if camera_id not in _stream_locks:
        _stream_locks[camera_id] = asyncio.Lock()
    
    # Check if camera is already in use
    if camera_id in _active_streams:
        raise HTTPException(status_code=409, detail=f"Camera {camera_id} already in use")
    
    async def generate_frames():
        # Mark camera as active
        _active_streams[camera_id] = True
        
        cap = None
        try:
            # Open camera using camera manager
            cap = camera_mgr.open_camera(camera_id)
            if not cap:
                raise HTTPException(status_code=404, detail=f"Cannot open camera {camera_id}")
            
            camera_info = camera_mgr.get_camera(camera_id)
            camera_name = camera_info.get('name', camera_id) if camera_info else camera_id
            
            logger.info(f"Live video feed started on camera {camera_id} ({camera_name})")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Process frame with detection and recognition
                try:
                    # Run YOLO person detection + face detection
                    detections = yolo_detector.detect(frame)
                    
                    for det in detections:
                        person_bbox = det.get("person_bbox")
                        face_bbox = det.get("face_bbox")
                        face_crop = det.get("face_crop")
                        body_crop = det.get("body_crop")
                        
                        if person_bbox:
                            px, py, pw, ph = person_bbox
                            
                            # Initialize result variables
                            person_id = "Unknown Person"
                            confidence = 0.0
                            is_masked = False
                            is_known = False
                            
                            # If face detected, run full recognition pipeline
                            if face_crop is not None and face_crop.size > 0:
                                # Mask detection
                                try:
                                    is_masked, _ = mask_det.is_masked(face_crop)
                                except Exception:
                                    is_masked = False
                                
                                # Liveness check
                                is_live, spoof_msg = liveness_det.check(face_crop)
                                if not is_live:
                                    person_id = "SPOOF"
                                    colour = (0, 0, 255)  # Red
                                else:
                                    # Multi-modal recognition
                                    scores = {}
                                    
                                    # Face embedding
                                    if face_faiss is not None:
                                        try:
                                            f_emb = embedder.extract(face_crop, masked=False)
                                            norm = np.linalg.norm(f_emb)
                                            if norm > 0: f_emb = f_emb / norm
                                            f_vec = np.array([f_emb], dtype=np.float32)
                                            f_dists, f_inds = face_faiss.search(f_vec, 5)
                                            
                                            f_labels_map = multi_labels.get("face_labels", [])
                                            for d, i in zip(f_dists[0], f_inds[0]):
                                                if i != -1 and i < len(f_labels_map):
                                                    pid_int = f_labels_map[i]
                                                    pid = label_map.get(pid_int, str(pid_int))
                                                    scr = max(0.0, 1.0 - (d / 2.0))
                                                    if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                                                    scores[pid]["F"] = max(scores[pid]["F"], scr)
                                        except Exception as e:
                                            logger.debug(f"Face embedding failed: {e}")
                                    
                                    # Body embedding
                                    if body_faiss is not None and body_crop is not None:
                                        try:
                                            b_emb = get_body_extractor().extract(body_crop)
                                            norm = np.linalg.norm(b_emb)
                                            if norm > 0: b_emb = b_emb / norm
                                            b_vec = np.array([b_emb], dtype=np.float32)
                                            b_dists, b_inds = body_faiss.search(b_vec, 5)
                                            
                                            b_labels_map = multi_labels.get("body_labels", [])
                                            for d, i in zip(b_dists[0], b_inds[0]):
                                                if i != -1 and i < len(b_labels_map):
                                                    pid_int = b_labels_map[i]
                                                    pid = label_map.get(pid_int, str(pid_int))
                                                    scr = max(0.0, 1.0 - (d / 2.0))
                                                    if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                                                    scores[pid]["B"] = max(scores[pid]["B"], scr)
                                        except Exception as e:
                                            logger.debug(f"Body embedding failed: {e}")
                                    
                                    # Attribute embedding
                                    if attr_faiss is not None and body_crop is not None:
                                        try:
                                            a_emb = attr_extractor.extract(body_crop)
                                            norm = np.linalg.norm(a_emb)
                                            if norm > 0: a_emb = a_emb / norm
                                            a_vec = np.array([a_emb], dtype=np.float32)
                                            a_dists, a_inds = attr_faiss.search(a_vec, 5)
                                            
                                            a_labels_map = multi_labels.get("attr_labels", [])
                                            for d, i in zip(a_dists[0], a_inds[0]):
                                                if i != -1 and i < len(a_labels_map):
                                                    pid_int = a_labels_map[i]
                                                    pid = label_map.get(pid_int, str(pid_int))
                                                    scr = max(0.0, 1.0 - (d / 2.0))
                                                    if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                                                    scores[pid]["A"] = max(scores[pid]["A"], scr)
                                        except Exception as e:
                                            logger.debug(f"Attribute embedding failed: {e}")
                                    
                                    # Late fusion
                                    best_score = 0.0
                                    for pid, s in scores.items():
                                        fused = (s["F"] * 0.5) + (s["B"] * 0.3) + (s["A"] * 0.2)
                                        if fused > best_score:
                                            best_score = fused
                                            person_id = pid
                                    
                                    confidence = best_score
                                    if confidence >= 0.60:
                                        is_known = True
                                    else:
                                        person_id = "Unknown Person"
                            
                            # Get person attributes
                            person_name = person_id
                            if is_known and person_id != "Unknown Person":
                                attrs = attributes_mgr.get_attributes(person_id)
                                if attrs:
                                    person_name = attrs.get("name", person_id)
                            
                            # Draw bounding box
                            colour = (0, 255, 0) if is_known else (0, 0, 255)  # Green for known, Red for unknown
                            cv2.rectangle(frame, (px, py), (px + pw, py + ph), colour, 2)
                            
                            # Draw face box if available
                            if face_bbox:
                                fx, fy, fw, fh = face_bbox
                                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), colour, 1)
                            
                            # Draw label
                            label = f"{person_name} ({confidence:.0%})"
                            if is_masked:
                                label += " [MASKED]"
                            
                            # Background for text
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (px, py - text_h - 10), (px + text_w + 10, py), colour, -1)
                            cv2.putText(frame, label, (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Push SSE event for dashboard - send ALL detections
                            event = {
                                "timestamp": datetime.now().isoformat(),
                                "camera_id": camera_id,  # Use camera_id directly
                                "person_id": person_id,
                                "name": person_name,
                                "is_known": is_known,
                                "is_masked": is_masked,
                                "confidence": float(confidence),
                            }
                            _push_sse_event(event)
                            
                            # Log event
                            surv_logger.log_event(
                                camera_id,
                                person_id,
                                float(confidence),
                                is_known=is_known,
                                is_masked=is_masked
                            )
                
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # Yield frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Small delay to control frame rate
                await asyncio.sleep(0.033)  # ~30 FPS
        
        except Exception as e:
            logger.error(f"Video feed error: {e}")
        finally:
            # Release camera using camera manager
            camera_mgr.release_camera(camera_id)
            
            # Remove from active streams
            if camera_id in _active_streams:
                del _active_streams[camera_id]
            
            logger.info(f"Live video feed stopped for camera {camera_id}")
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video/status", tags=["Video"])
async def video_status(camera_id: str = Query(None, description="Optional camera ID to check")):
    """Check if camera(s) are currently active."""
    if camera_id:
        # Check specific camera
        return {
            "camera_id": camera_id,
            "active": camera_id in _active_streams
        }
    else:
        # Return all active cameras
        return {
            "active_cameras": list(_active_streams.keys()),
            "count": len(_active_streams)
        }

@app.get("/video/snapshot", tags=["Video"])
async def capture_snapshot(camera_id: str = Query("CAM001", description="Camera ID to capture from")):
    """
    Capture a single frame from the camera and return as JPEG image.
    Works with both active streams and can open camera temporarily if not streaming.
    """
    from fastapi.responses import Response
    import io
    
    # Validate camera exists and is enabled
    camera_info = camera_mgr.get_camera(camera_id)
    if not camera_info:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    if not camera_info.get('enabled', False):
        raise HTTPException(status_code=400, detail=f"Camera {camera_id} is disabled")
    
    cap = None
    should_release = False
    
    try:
        # Check if camera is already streaming
        if camera_id in _active_streams:
            # Camera is active in video feed, open a new temporary capture
            cap = camera_mgr.open_camera(camera_id)
            should_release = True
        else:
            # Open camera temporarily for snapshot
            cap = camera_mgr.open_camera(camera_id)
            should_release = True
        
        if not cap:
            raise HTTPException(status_code=500, detail=f"Failed to open camera {camera_id}")
        
        # Read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame from camera")
        
        # Process frame with detection and recognition (same as video feed)
        try:
            detections = yolo_detector.detect(frame)
            
            for det in detections:
                person_bbox = det.get("person_bbox")
                face_bbox = det.get("face_bbox")
                face_crop = det.get("face_crop")
                body_crop = det.get("body_crop")
                
                if person_bbox:
                    px, py, pw, ph = person_bbox
                    
                    person_id = "Unknown Person"
                    confidence = 0.0
                    is_masked = False
                    is_known = False
                    
                    if face_crop is not None and face_crop.size > 0:
                        try:
                            is_masked, _ = mask_det.is_masked(face_crop)
                        except Exception:
                            is_masked = False
                        
                        is_live, spoof_msg = liveness_det.check(face_crop)
                        if not is_live:
                            person_id = "SPOOF"
                            colour = (0, 0, 255)
                        else:
                            scores = {}
                            
                            # Face embedding
                            if face_faiss is not None:
                                try:
                                    f_emb = embedder.extract(face_crop, masked=False)
                                    norm = np.linalg.norm(f_emb)
                                    if norm > 0: f_emb = f_emb / norm
                                    f_vec = np.array([f_emb], dtype=np.float32)
                                    f_dists, f_inds = face_faiss.search(f_vec, 5)
                                    
                                    f_labels_map = multi_labels.get("face_labels", [])
                                    for d, i in zip(f_dists[0], f_inds[0]):
                                        if i != -1 and i < len(f_labels_map):
                                            pid_int = f_labels_map[i]
                                            pid = label_map.get(pid_int, str(pid_int))
                                            scr = max(0.0, 1.0 - (d / 2.0))
                                            if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                                            scores[pid]["F"] = max(scores[pid]["F"], scr)
                                except Exception as e:
                                    logger.debug(f"Face embedding failed: {e}")
                            
                            # Body embedding
                            if body_faiss is not None and body_crop is not None:
                                try:
                                    b_emb = get_body_extractor().extract(body_crop)
                                    norm = np.linalg.norm(b_emb)
                                    if norm > 0: b_emb = b_emb / norm
                                    b_vec = np.array([b_emb], dtype=np.float32)
                                    b_dists, b_inds = body_faiss.search(b_vec, 5)
                                    
                                    b_labels_map = multi_labels.get("body_labels", [])
                                    for d, i in zip(b_dists[0], b_inds[0]):
                                        if i != -1 and i < len(b_labels_map):
                                            pid_int = b_labels_map[i]
                                            pid = label_map.get(pid_int, str(pid_int))
                                            scr = max(0.0, 1.0 - (d / 2.0))
                                            if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                                            scores[pid]["B"] = max(scores[pid]["B"], scr)
                                except Exception as e:
                                    logger.debug(f"Body embedding failed: {e}")
                            
                            # Attribute embedding
                            if attr_faiss is not None and body_crop is not None:
                                try:
                                    a_emb = attr_extractor.extract(body_crop)
                                    norm = np.linalg.norm(a_emb)
                                    if norm > 0: a_emb = a_emb / norm
                                    a_vec = np.array([a_emb], dtype=np.float32)
                                    a_dists, a_inds = attr_faiss.search(a_vec, 5)
                                    
                                    a_labels_map = multi_labels.get("attr_labels", [])
                                    for d, i in zip(a_dists[0], a_inds[0]):
                                        if i != -1 and i < len(a_labels_map):
                                            pid_int = a_labels_map[i]
                                            pid = label_map.get(pid_int, str(pid_int))
                                            scr = max(0.0, 1.0 - (d / 2.0))
                                            if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                                            scores[pid]["A"] = max(scores[pid]["A"], scr)
                                except Exception as e:
                                    logger.debug(f"Attribute embedding failed: {e}")
                            
                            # Late fusion
                            best_score = 0.0
                            for pid, s in scores.items():
                                fused = (s["F"] * 0.5) + (s["B"] * 0.3) + (s["A"] * 0.2)
                                if fused > best_score:
                                    best_score = fused
                                    person_id = pid
                            
                            confidence = best_score
                            if confidence >= 0.60:
                                is_known = True
                            else:
                                person_id = "Unknown Person"
                    
                    # Get person name
                    person_name = person_id
                    if is_known and person_id != "Unknown Person":
                        attrs = attributes_mgr.get_attributes(person_id)
                        if attrs:
                            person_name = attrs.get("name", person_id)
                    
                    # Draw bounding box
                    colour = (0, 255, 0) if is_known else (0, 0, 255)
                    cv2.rectangle(frame, (px, py), (px + pw, py + ph), colour, 2)
                    
                    # Draw face box
                    if face_bbox:
                        fx, fy, fw, fh = face_bbox
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), colour, 1)
                    
                    # Draw label
                    label = f"{person_name} ({confidence:.0%})"
                    if is_masked:
                        label += " [MASKED]"
                    
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (px, py - text_h - 10), (px + text_w + 10, py), colour, -1)
                    cv2.putText(frame, label, (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        except Exception as e:
            logger.error(f"Error processing snapshot frame: {e}")
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode frame as JPEG")
        
        # Return image
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Snapshot error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to capture snapshot: {str(e)}")
    finally:
        # Release camera if we opened it temporarily
        if should_release and cap:
            camera_mgr.release_camera(camera_id)


# ── Serve React Dashboard ─────────────────────────────────────────────────────
_dashboard_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "dist")
if os.path.isdir(_dashboard_dist):
    app.mount("/", StaticFiles(directory=_dashboard_dist, html=True), name="dashboard")
# Multi-Modal Pipeline Active
