"""
Camera Manager for handling multiple camera sources including:
- Local webcams (temporary solution)
- RTSP streams from CCTV cameras
- Camera configuration and management
"""

import json
import os
import cv2
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, config_file: str = "camera_config.json"):
        self.config_file = config_file
        self.cameras = {}
        self.active_captures = {}
        self.load_config()
    
    def load_config(self):
        """Load camera configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.cameras = {cam['id']: cam for cam in config.get('cameras', [])}
                logger.info(f"Loaded {len(self.cameras)} cameras from config")
            else:
                logger.warning(f"Camera config file not found: {self.config_file}")
                self.cameras = {}
        except Exception as e:
            logger.error(f"Error loading camera config: {e}")
            self.cameras = {}
    
    def save_config(self):
        """Save camera configuration to JSON file"""
        try:
            config = {'cameras': list(self.cameras.values())}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved {len(self.cameras)} cameras to config")
            return True
        except Exception as e:
            logger.error(f"Error saving camera config: {e}")
            return False
    
    def get_all_cameras(self) -> List[Dict]:
        """Get list of all configured cameras"""
        return list(self.cameras.values())
    
    def get_camera(self, camera_id: str) -> Optional[Dict]:
        """Get camera configuration by ID"""
        return self.cameras.get(camera_id)
    
    def get_enabled_cameras(self) -> List[Dict]:
        """Get list of enabled cameras"""
        return [cam for cam in self.cameras.values() if cam.get('enabled', False)]
    
    def add_camera(self, camera_data: Dict) -> bool:
        """Add a new camera to configuration"""
        try:
            camera_id = camera_data.get('id')
            if not camera_id:
                logger.error("Camera ID is required")
                return False
            
            if camera_id in self.cameras:
                logger.error(f"Camera {camera_id} already exists")
                return False
            
            self.cameras[camera_id] = camera_data
            self.save_config()
            logger.info(f"Added camera: {camera_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding camera: {e}")
            return False
    
    def update_camera(self, camera_id: str, camera_data: Dict) -> bool:
        """Update existing camera configuration"""
        try:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False
            
            # Preserve the ID
            camera_data['id'] = camera_id
            self.cameras[camera_id] = camera_data
            self.save_config()
            logger.info(f"Updated camera: {camera_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating camera: {e}")
            return False
    
    def delete_camera(self, camera_id: str) -> bool:
        """Delete camera from configuration"""
        try:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return False
            
            # Close capture if active
            if camera_id in self.active_captures:
                self.release_camera(camera_id)
            
            del self.cameras[camera_id]
            self.save_config()
            logger.info(f"Deleted camera: {camera_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting camera: {e}")
            return False
    
    def get_camera_source(self, camera_id: str):
        """Get the OpenCV-compatible source for a camera"""
        camera = self.get_camera(camera_id)
        if not camera:
            return None
        
        source = camera.get('source')
        camera_type = camera.get('type', 'local')
        
        # For local cameras, source should be an integer
        if camera_type == 'local':
            try:
                return int(source)
            except (ValueError, TypeError):
                return 0
        
        # For RTSP/HTTP streams, return the URL string
        return source
    
    def open_camera(self, camera_id: str) -> Optional[cv2.VideoCapture]:
        """Open a camera and return VideoCapture object"""
        try:
            if camera_id in self.active_captures:
                logger.warning(f"Camera {camera_id} already open")
                return self.active_captures[camera_id]
            
            source = self.get_camera_source(camera_id)
            if source is None:
                logger.error(f"Camera {camera_id} not found or invalid source")
                return None
            
            camera = self.get_camera(camera_id)
            camera_type = camera.get('type', 'local')
            
            # Create VideoCapture with appropriate settings
            cap = cv2.VideoCapture(source)
            
            # For RTSP streams, set additional options
            if camera_type in ['rtsp', 'http']:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  # 10 second read timeout
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id} with source {source}")
                logger.error(f"Possible issues: RTSP not enabled, wrong credentials, camera offline, or network issue")
                return None
            
            # Try to read a test frame to verify connection
            ret, _ = cap.read()
            if not ret:
                logger.error(f"Camera {camera_id} opened but cannot read frames")
                cap.release()
                return None
            
            self.active_captures[camera_id] = cap
            logger.info(f"Opened camera: {camera_id} (type: {camera_type})")
            return cap
        except Exception as e:
            logger.error(f"Error opening camera {camera_id}: {e}")
            return None
    
    def release_camera(self, camera_id: str):
        """Release a camera capture"""
        try:
            if camera_id in self.active_captures:
                self.active_captures[camera_id].release()
                del self.active_captures[camera_id]
                logger.info(f"Released camera: {camera_id}")
        except Exception as e:
            logger.error(f"Error releasing camera {camera_id}: {e}")
    
    def release_all_cameras(self):
        """Release all active camera captures"""
        for camera_id in list(self.active_captures.keys()):
            self.release_camera(camera_id)
    
    def is_camera_active(self, camera_id: str) -> bool:
        """Check if a camera is currently active"""
        return camera_id in self.active_captures
    
    def get_active_cameras(self) -> List[str]:
        """Get list of currently active camera IDs"""
        return list(self.active_captures.keys())
