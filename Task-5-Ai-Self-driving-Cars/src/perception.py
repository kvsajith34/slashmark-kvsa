"""
High-level Perception module.
Currently focuses on lane detection but designed to be extended
(e.g., add vehicle detection, traffic light classification, road segmentation).
"""

import cv2
import numpy as np
from typing import Dict, Any
import logging

from lane_detector import LaneDetector
from utils import load_config

logger = logging.getLogger(__name__)


class RoadPerception:
    """
    Unified perception interface for the autonomous stack.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.lane_detector = LaneDetector(self.config)
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run full perception stack on a single frame.
        Returns rich dict with lanes, metrics, and debug images.
        """
        self.frame_count += 1

        # Lane detection (core perception for this task)
        lane_result = self.lane_detector.detect(frame)

        # Basic road mask (simple color-based for "basic perception")
        road_mask = self._basic_road_segmentation(frame)

        # Combine
        perception_output = {
            "timestamp": self.frame_count,
            "lanes": lane_result,
            "road_mask": road_mask,
            "vehicle_detected": False,  # placeholder for future extension
            "traffic_light": None,      # placeholder
            "confidence": lane_result.get("confidence", 0.5),
        }

        return perception_output

    def _basic_road_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """Very simple HSV-based road mask (can be replaced with DL segmentation)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Typical asphalt range (tunable)
        lower = np.array([0, 0, 40])
        upper = np.array([180, 60, 200])
        mask = cv2.inRange(hsv, lower, upper)
        # Dilate to fill gaps
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def draw_perception(self, frame: np.ndarray, perception: Dict[str, Any]) -> np.ndarray:
        """Overlay all perception outputs on frame."""
        vis = frame.copy()

        # Draw lanes
        if perception["lanes"]["left_fit"] is not None:
            vis = self.lane_detector.draw_lanes(vis, perception["lanes"])

        # Optional: show road mask as semi-transparent green
        road_mask = perception["road_mask"]
        if road_mask is not None:
            colored_mask = np.zeros_like(vis)
            colored_mask[:, :, 1] = road_mask  # green channel
            vis = cv2.addWeighted(vis, 0.85, colored_mask, 0.15, 0)

        return vis
