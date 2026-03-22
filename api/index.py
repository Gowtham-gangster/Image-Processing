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
import os
import sys

# Ensure relative imports and model loading works
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from detection import Detector
from face_alignment import FaceAligner
from feature_extractor import FeatureExtractor
from unknown_detector import UnknownDetector
from attributes_manager import AttributesManager
from database import PersonDatabase
from liveness_detector import LivenessDetector
from mask_detector import MaskDetector
from alert_manager import AlertManager, ALERT_UNKNOWN_PERSON, ALERT_UNMASKED, ALERT_SPOOF
from surveillance_logger import SurveillanceLogger
from yolo_person_detector import YoloPersonDetector
from body_feature_extractor import BodyFeatureExtractor
from body_embedding_database import BodyEmbeddingDatabase
from attribute_extractor import AttributeExtractor
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
detector       = None
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

face_faiss   = None
body_faiss   = None
attr_faiss   = None
multi_labels = {}
label_map    = {}


@app.on_event("startup")
def startup_event():
    global detector, aligner, embedder, db, attributes_mgr, unknown_det
    global person_db, liveness_det, mask_det, alert_mgr, surv_logger
    global yolo_detector, body_extractor, body_db, attr_extractor

    logger.info("Initializing AI pipeline components...")

    global face_faiss, body_faiss, attr_faiss, multi_labels, label_map
    
    person_db      = PersonDatabase()
    attributes_mgr = AttributesManager(db=person_db)
    detector       = Detector(backend="auto")
    aligner        = FaceAligner(min_confidence=0.40)
    embedder       = FeatureExtractor()
    
    import pickle
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    idx_path = os.path.join(base_dir, "embeddings", "faiss_index.index")
    body_path = os.path.join(base_dir, "embeddings", "body_faiss.index")
    attr_path = os.path.join(base_dir, "embeddings", "attr_faiss.index")
    labels_path = os.path.join(base_dir, "embeddings", "multi_labels.pkl")
    
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

    legacy_path = os.path.join(base_dir, "embeddings", "labels.pkl")
    if os.path.exists(legacy_path):
        with open(legacy_path, "rb") as f:
            label_map = pickle.load(f)
        logger.info("Legacy labels loaded.")
        
    unknown_det    = UnknownDetector(threshold=0.60)
    liveness_det   = LivenessDetector()
    mask_det       = MaskDetector()
    alert_mgr      = AlertManager()
    surv_logger    = SurveillanceLogger()
    
    yolo_detector  = YoloPersonDetector(aligner=aligner)
    body_extractor = BodyFeatureExtractor()
    body_db        = BodyEmbeddingDatabase()
    attr_extractor = AttributeExtractor()

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
    return {"status": "healthy", "pipeline_active": detector is not None, "version": "3.0.0"}


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
    return db.get_recent_alerts(limit=limit, offset=offset)

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
                
                if faiss_idx is not None:
                    norm = np.linalg.norm(emb)
                    if norm > 0: emb = emb / norm
                    q_vec = np.array([emb], dtype=np.float32)
                    distances, indices = faiss_idx.search(q_vec, 1)
                    
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

            if not is_known:
                alert_mgr.send_alert(ALERT_UNKNOWN_PERSON, camera_id=camera_id,
                                     person_id=person_id, confidence=float(score))
            elif is_masked_flag:
                alert_mgr.send_alert(ALERT_UNMASKED, camera_id=camera_id,
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
    
    # 1. Direct MTCNN detection with 1.5x upscaling globally
    upscaled_frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
    global_faces = aligner.align(upscaled_frame)
    if global_faces:
        best_face = max(global_faces, key=lambda f: f["box"][2] * f["box"][3])
        face_crop = best_face["aligned_crop"]
        bx, by, bw, bh = best_face["box"]
        debug_info["face_bounding_box"] = f"[{int(bx/1.5)}, {int(by/1.5)}, {int(bw/1.5)}, {int(bh/1.5)}]"
        debug_info["detection_confidence"] = f"{best_face['confidence']:.3f}"
        
    # YOLO Body extraction for fallback or hybrid context
    results = yolo_detector.detect(frame)
    if results:
        best_res = max(results, key=lambda x: x["person_bbox"][2] * x["person_bbox"][3])
        body_crop = best_res.get("body_crop")
        if face_crop is None and best_res.get("face_crop") is not None:
            face_crop = best_res["face_crop"]
            box_arr = best_res.get("face_bbox")
            debug_info["face_bounding_box"] = str(box_arr) if box_arr else "N/A"
            debug_info["detection_confidence"] = f"{best_res.get('face_conf', 0.0):.3f}"
            
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
        
    # --- Body + Attribute Computations (Always Running) ---
    b_dists, b_inds, _ = search_faiss(body_faiss, body_extractor, body_crop, "body_labels")
    a_dists, a_inds, _ = search_faiss(attr_faiss, attr_extractor, body_crop, "attr_labels")
    
    b_labels_map = multi_labels.get("body_labels", [])
    a_labels_map = multi_labels.get("attr_labels", [])
    f_labels_map = multi_labels.get("face_labels", [])
    
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
            
    # --- Face Computations (Optional) ---
    if face_crop is not None and face_crop.size > 0:
        debug_info["face_detected"] = "Yes"
        f_dists, f_inds, shape = search_faiss(face_faiss, embedder, face_crop, "face_labels")
        debug_info["embedding_generated"] = "Yes"
        debug_info["embedding_vector_size"] = str(shape)
        debug_info["nearest_distance"] = f"{f_dists[0]:.4f}"
        
        for d, i in zip(f_dists, f_inds):
            if i != -1 and i < len(f_labels_map):
                pid_int = f_labels_map[i]
                pid = label_map.get(pid_int, str(pid_int))
                scr = max(0.0, 1.0 - (d / 2.0))
                if pid not in scores: scores[pid] = {"F": 0, "B": 0, "A": 0}
                scores[pid]["F"] = max(scores[pid]["F"], scr)
                
    # --- LATE FUSION ---
    best_id = "Unknown Person"
    best_score = 0.0
    
    for pid, s in scores.items():
        if debug_info["face_detected"] == "Yes":
            # 0.5 Face + 0.3 Body + 0.2 Attribute
            fused = (s["F"] * 0.5) + (s["B"] * 0.3) + (s["A"] * 0.2)
        else:
            # 0.7 Body + 0.3 Attribute (Ignoring Face)
            fused = (s["B"] * 0.7) + (s["A"] * 0.3)
            
        if fused > best_score:
            best_score = fused
            best_id = pid
            
    # Apply Threshold
    # When faces are omitted, the maximum body threshold matching requires stricter bounds
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
        if "error" not in result:
            if not result.get("mask", False):
                alert_mgr.send_alert(ALERT_UNMASKED, camera_id="ImageUpload", person_id=result.get("person", "Unknown"), confidence=result.get("confidence", 0.0))
                
            if result.get("person", "Unknown") == "Unknown":
                alert_mgr.send_alert(ALERT_UNKNOWN_PERSON, camera_id="ImageUpload", person_id="Unknown", confidence=result.get("confidence", 0.0))
                
        return result
    except Exception as e:
        logger.exception("Error in /predict-image")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Serve React Dashboard ─────────────────────────────────────────────────────
_dashboard_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "dist")
if os.path.isdir(_dashboard_dist):
    app.mount("/", StaticFiles(directory=_dashboard_dist, html=True), name="dashboard")
# Multi-Modal Pipeline Active
