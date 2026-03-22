"""
person_tracker.py
=================
Temporal person tracking across video frames to reduce recognition load.

Requirements met
----------------
1. Track detected persons across continuous frames.
2. Use DeepSORT tracking algorithm.
3. Assign consistent Track IDs to persons.
4. Reduce repeated recognition calls by caching identity to Track ID.

Dependencies
------------
pip install deep-sort-realtime

Usage
-----
    import cv2
    from person_tracker import PersonTracker
    
    tracker = PersonTracker(max_age=30)
    
    # 1. Provide a list of detections: [ [x, y, w, h], confidence, class_id ]
    # 2. Extract bounding boxes from your face/body detector
    detections = [ ([10, 10, 50, 50], 0.9, 0) ]
    
    tracks = tracker.update_tracks(detections, frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb() # Left, Top, Right, Bottom
        
        # If identity not yet recognized for this track, run FaceNet/ResNet
        if not tracker.has_identity(track_id):
            person_id = "run_your_recognition_here()"
            tracker.assign_identity(track_id, person_id)
            
        print(f"Track {track_id} is {tracker.get_identity(track_id)}")
"""

import logging
from typing import List, Tuple, Any, Dict, Optional
import numpy as np

from config import LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

class PersonTracker:
    """
    Wraps the DeepSORT algorithm to assign stable Track IDs to bounding boxes.
    Maintains an identity cache to prevent continuously re-running expensive CNN 
    recognizers (FaceNet/ResNet50) on every single frame.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3) -> None:
        """
        Parameters
        ----------
        max_age : int
            Maximum number of missed misses before a track is deleted.
        n_init : int
            Number of consecutive detections before the track is confirmed.
        """
        logger.info("Initializing PersonTracker (DeepSORT)...")
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            
            # Initialize DeepSORT
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                nms_max_overlap=1.0, # Disable internal NMS since our Detector handles it
                max_cosine_distance=0.2, # Threshold for cosine distance
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet", # Use built-in lightweight embedder for tracking boxes
                half=True,
                bgr=True,
                embedder_gpu=False
            )
            logger.info("DeepSORT loaded successfully.")
            
        except ImportError:
            logger.error(
                "The 'deep-sort-realtime' package is missing.\n"
                "Please run: pip install deep-sort-realtime"
            )
            raise
            
        # Cache for recognized identities: { track_id : (person_id, confidence, dict_attrs) }
        self._identity_cache: Dict[str, dict] = {}

    def update_tracks(self, bboxes: List[Tuple[int, int, int, int]], confidences: List[float], frame: np.ndarray) -> List[Any]:
        """
        Updates the DeepSORT tracker with the latest bounding boxes.
        
        Parameters
        ----------
        bboxes : list
            List of [x, y, w, h] integer lists.
        confidences : list
            List of float confidence scores from the detector.
        frame : np.ndarray
            The original BGR frame (DeepSORT extracts visual features from it).
            
        Returns
        -------
        tracks : list
            List of Track objects from deep_sort_realtime.
        """
        # Format detections for deep_sort_realtime: [ ([x,y,w,h], confidence, detection_class) ]
        ds_detections = []
        for box, conf in zip(bboxes, confidences):
            # Class ID is '0' for person/face
            ds_detections.append((list(box), conf, 0))
            
        # Update tracker
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        
        # Cleanup cache for stale/deleted tracks (to prevent memory leaks across long running CCTV)
        active_track_ids = {str(t.track_id) for t in tracks if t.is_confirmed()}
        stale_ids = [tid for tid in self._identity_cache.keys() if tid not in active_track_ids]
        
        # Note: We give stale tracks a "grace period" before deletion if they briefly occlude. 
        # DeepSORT handles the visual re-ID, so we just clear cache when DeepSORT drops the ID permanently.
        # But for simplicity, we rely on the tracked items. DeepSORT deletes tracks internally after max_age frames.
        # So we can periodically clean out self._identity_cache manually here.
        # For now, we only delete ones that DeepSORT formally deleted:
        
        # Filter deleted tracks internally by DeepSORT
        current_valid_tracks = [t.track_id for t in self.tracker.tracker.tracks]
        for cached_tid in list(self._identity_cache.keys()):
            if cached_tid not in current_valid_tracks:
                del self._identity_cache[cached_tid]

        return tracks

    def has_identity(self, track_id: str) -> bool:
        """Check if we already ran expensive facial recognition on this Track ID."""
        return str(track_id) in self._identity_cache

    def assign_identity(self, track_id: str, person_id: str, confidence: float, attributes: dict = None) -> None:
        """
        Cache the identity for a given Track ID so we don't need to recognize them again.
        
        Parameters
        ----------
        track_id : str
            The DeepSORT internal ID.
        person_id : str
            The identified user ("person1").
        confidence : float
            The cosine similarity score (e.g. 0.85).
        attributes : dict, optional
            The fetched dataset attributes mapping.
        """
        self._identity_cache[str(track_id)] = {
            "person_id": person_id,
            "confidence": confidence,
            "attributes": attributes
        }

    def get_identity(self, track_id: str) -> Optional[dict]:
        """
        Fetch the cached identity payload for a Track ID.
        Returns None if not yet cached.
        """
        return self._identity_cache.get(str(track_id))

    def force_re_recognition(self, track_id: str) -> None:
        """Deletes the cache for a track, forcing the CCTV loop to scan their face again."""
        if str(track_id) in self._identity_cache:
            del self._identity_cache[str(track_id)]


if __name__ == "__main__":
    # Smoke Test
    try:
        tracker = PersonTracker()
        print("PersonTracker + DeepSORT Initialized Successfully.")
    except Exception as e:
        print(f"Init failed: {e}")
