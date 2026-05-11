"""
AetherNav Perception Module
YOLOv8 wrapper + Mock detector for instant demo
+ Depth estimation hooks (MiDaS ready)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import random
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ObstacleDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get('detector_model', 'mock')
        self.conf_threshold = config.get('confidence_threshold', 0.55)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.class_names = config.get('classes', ["wall", "chair", "table", "person", "door", "obstacle", "dynamic"])
        
        if self.model_name != "mock" and YOLO_AVAILABLE:
            try:
                self.model = YOLO(f"{self.model_name}.pt")
                self.model.to(self.device)
                print(f"✅ Loaded YOLOv8 model: {self.model_name} on {self.device}")
            except Exception as e:
                print(f"⚠️ Failed to load YOLO: {e}. Falling back to mock mode.")
                self.model_name = "mock"
        else:
            print("ℹ️ Running in MOCK perception mode (realistic simulation)")

    def detect(self, image: np.ndarray, drone_state: Optional[Dict] = None) -> List[Dict]:
        """
        Main detection entry point.
        Returns list of detections with bbox, class, confidence, distance estimate.
        """
        if self.model_name == "mock" or self.model is None:
            return self._mock_detect(drone_state)
        
        # Real YOLO inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else "unknown"
            
            # Estimate distance from bbox size (heuristic)
            bbox_area = (x2 - x1) * (y2 - y1)
            estimated_dist = max(0.8, 12.0 / max(np.sqrt(bbox_area), 1))
            
            detections.append({
                'class': class_name,
                'confidence': round(conf, 3),
                'bbox': [x1, y1, x2, y2],
                'distance': round(estimated_dist, 2),
                'center': [(x1 + x2)//2, (y1 + y2)//2]
            })
        
        return detections

    def _mock_detect(self, drone_state: Optional[Dict] = None) -> List[Dict]:
        """High-fidelity mock detector that mimics real outputs"""
        detections = []
        num = random.randint(3, 8)
        
        for _ in range(num):
            cls = random.choice(self.class_names)
            conf = round(random.uniform(0.68, 0.97), 3)
            
            # Simulate realistic distribution
            if cls in ['wall', 'door']:
                dist = round(random.uniform(1.2, 5.8), 2)
                w, h = random.randint(80, 220), random.randint(120, 280)
            elif cls == 'person':
                dist = round(random.uniform(1.8, 4.5), 2)
                w, h = random.randint(35, 70), random.randint(90, 160)
            else:
                dist = round(random.uniform(0.9, 4.2), 2)
                w, h = random.randint(50, 140), random.randint(50, 140)
            
            x = random.randint(60, 420)
            y = random.randint(50, 260)
            
            detections.append({
                'class': cls,
                'confidence': conf,
                'bbox': [x, y, x + w, y + h],
                'distance': dist,
                'center': [x + w//2, y + h//2],
                'angle': round(random.uniform(-55, 55), 1)  # relative to heading
            })
        
        # Sort by distance
        detections.sort(key=lambda d: d['distance'])
        return detections[:6]  # limit for dashboard

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Placeholder for MiDaS depth. Returns normalized depth map."""
        if self.model_name == "mock":
            h, w = image.shape[:2] if len(image.shape) > 2 else (320, 480)
            depth = np.linspace(0.6, 5.2, w).reshape(1, -1)
            depth = np.repeat(depth, h, axis=0)
            depth += np.random.normal(0, 0.15, depth.shape)
            return np.clip(depth, 0.4, 6.0)
        # Real MiDaS would go here
        return np.zeros((320, 480)) + 2.5

class DepthEstimator:
    """MiDaS wrapper (stub for now)"""
    def __init__(self, model_type: str = "small"):
        self.model_type = model_type
        print(f"Depth estimator initialized ({model_type} mode)")

    def predict(self, image: np.ndarray) -> np.ndarray:
        # In production: load torch.hub model and run
        return np.random.uniform(0.5, 5.0, image.shape[:2]).astype(np.float32)