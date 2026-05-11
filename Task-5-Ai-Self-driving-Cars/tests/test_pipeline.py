"""
Basic smoke tests for Task 5 pipeline.
Run with: pytest tests/ -q
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import cv2
import pytest

from lane_detector import LaneDetector
from pid_controller import PIDController
from perception import RoadPerception
from control_simulator import AutonomousDrivingStack
from utils import load_config


def test_config_loads():
    cfg = load_config("config.yaml")
    assert "lane_detector" in cfg
    assert "pid_controller" in cfg


def test_lane_detector_smoke():
    cfg = load_config("config.yaml")
    detector = LaneDetector(cfg)
    dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    # Draw fake lane lines
    cv2.line(dummy_frame, (200, 540), (400, 300), (255, 255, 255), 8)
    cv2.line(dummy_frame, (760, 540), (560, 300), (255, 255, 255), 8)
    result = detector.detect(dummy_frame)
    assert result["left_fit"] is not None or result["right_fit"] is not None


def test_pid_controller():
    cfg = load_config("config.yaml")
    pid = PIDController(cfg)
    steer = pid.update(50.0)  # positive error
    assert -25.0 <= steer <= 25.0
    assert isinstance(steer, float)


def test_full_stack_smoke():
    stack = AutonomousDrivingStack("config.yaml")
    # Run short simulation
    metrics = stack.run_simulation(duration_sec=2.0)
    assert "avg_abs_cte_px" in metrics
    assert metrics["frames"] > 10


def test_image_pipeline():
    stack = AutonomousDrivingStack("config.yaml")
    dummy = np.zeros((540, 960, 3), dtype=np.uint8)
    perception = stack.perception.process_frame(dummy)
    assert "lanes" in perception
    assert "confidence" in perception
