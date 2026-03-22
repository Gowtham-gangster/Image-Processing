"""
enrollment_gui.py
=================
Graphical Wizard (Tkinter) for enrolling new identities into the Hybrid Identification System.
Bridges the physical OpenCV Webcam directly to the FAISS index and CSV rows.

Running
-------
    python enrollment_gui.py
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import logging

from config import CAMERA_SOURCE, LOG_LEVEL
from face_alignment import FaceAligner
from yolo_person_detector import YoloPersonDetector
from embedding_model import FaceNetEmbedder
from body_feature_extractor import BodyFeatureExtractor
from embedding_database import EmbeddingDatabase
from body_embedding_database import BodyEmbeddingDatabase
from attributes_manager import AttributesManager

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class EnrollmentWizard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hybrid ID System - Fast Enrollment Wizard")
        self.root.geometry("900x550")
        
        # --- Load Deep Learning Backends ---
        logger.info("Spinning up inference core...")
        self.aligner = FaceAligner()
        self.yolo = YoloPersonDetector(aligner=self.aligner)
        self.face_cnn = FaceNetEmbedder()
        self.body_cnn = BodyFeatureExtractor()
        
        # --- Load Databases ---
        self.faiss_db = EmbeddingDatabase()
        self.body_db = BodyEmbeddingDatabase()
        self.attrs_db = AttributesManager()
        
        # --- Camera State ---
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)
        self.current_frame = None
        self.is_running = True
        
        self._build_ui()
        self._video_loop()

    def _build_ui(self):
        # 1. Video Frame (Left)
        self.video_frame = tk.Frame(self.root, width=640, height=480, bg="black")
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()
        
        # 2. Controls & Form (Right)
        self.form_frame = tk.Frame(self.root)
        self.form_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(self.form_frame, text="New Person Registration", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Form Fields
        self.fields = {}
        labels = [
            ("Person ID (e.g. p005):", "person_id"),
            ("Full Name:", "name"),
            ("Gender:", "gender"),
            ("Age:", "age"),
            ("Phone:", "phone"),
            ("Address:", "address")
        ]
        
        for text, key in labels:
            row = tk.Frame(self.form_frame)
            row.pack(fill=tk.X, pady=5)
            tk.Label(row, text=text, width=18, anchor="w").pack(side=tk.LEFT)
            entry = ttk.Entry(row)
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.fields[key] = entry
            
        # Register Button
        self.btn_capture = ttk.Button(self.form_frame, text="Capture & Enroll Identity", command=self.enroll, style='Action.TButton')
        self.btn_capture.pack(pady=25, fill=tk.X, ipady=10)
        
    def _video_loop(self):
        if not self.is_running:
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            
            # Convert BGR to RGB for Tkinter display
            cv2frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk # Keep exact reference
            self.video_label.configure(image=imgtk)
            
        self.root.after(15, self._video_loop)

    def enroll(self):
        if self.current_frame is None:
            messagebox.showerror("Error", "Camera feeds not established yet.")
            return
            
        # 1. Validate Form
        pid = self.fields["person_id"].get().strip()
        if not pid:
            messagebox.showerror("Error", "Person ID is strictly required.")
            return
            
        if pid in self.attrs_db:
            res = messagebox.askyesno("Warning", f"ID '{pid}' already exists. Overwrite attributes & add another face to their FAISS profile?")
            if not res:
                return
                
        # 2. Extract Biometrics
        logger.info("Attempting extraction on frame...")
        yolo_results = self.yolo.detect(self.current_frame)
        
        if not yolo_results:
            messagebox.showerror("Error", "No person detected in the frame. Please stand in front of the camera.")
            return
            
        # Get largest person in frame
        best_person = max(yolo_results, key=lambda r: r["person_bbox"][2] * r["person_bbox"][3])
        
        face_crop = best_person.get("face_crop")
        body_crop = best_person.get("body_crop")
        
        if face_crop is None:
            messagebox.showerror("Error", "Body detected, but MTCNN could not find the facial features. Please turn around and look at the camera.")
            return
            
        # 3. Generate CNN Embeddings
        try:
            face_emb = self.face_cnn.extract(face_crop)
            body_emb = self.body_cnn.extract(body_crop)
            
            # 4. Save to Vectors
            self.faiss_db.add_embedding(pid, face_emb)
            self.faiss_db.save()
            
            self.body_db.add_embedding(pid, body_emb)
            self.body_db.save()
            
            # 5. Save Attributes to DB
            self.attrs_db.add_person(
                person_id=pid,
                name=self.fields["name"].get(),
                age=self.fields["age"].get(),
                gender=self.fields.get("gender").get() if "gender" in self.fields else "",
                phone=self.fields["phone"].get(),
                address=self.fields["address"].get()
            )
            
            messagebox.showinfo("Success", f"{pid} successfully enrolled into FAISS and ResNet50 databases!")
            
            # Clear form
            for entry in self.fields.values():
                entry.delete(0, tk.END)
                
        except Exception as e:
            logger.error("Enrollment crashed: %s", e)
            messagebox.showerror("Enrollment Error", str(e))

    def on_close(self):
        self.is_running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    
    # Optional styling
    style = ttk.Style()
    style.configure('Action.TButton', font=('Helvetica', 11, 'bold'))
    
    app = EnrollmentWizard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
