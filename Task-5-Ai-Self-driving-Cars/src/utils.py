"""
Utility functions for the self-driving car stack.
Clean, reusable helpers for image processing, geometry, and visualization.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration with sensible defaults."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning("Config not found, using defaults")
        return get_default_config()


def get_default_config() -> dict:
    """Fallback defaults if config.yaml missing."""
    return {
        "lane_detector": {
            "canny_low": 50,
            "canny_high": 150,
            "gaussian_kernel": 5,
            "rho": 2,
            "theta": np.pi / 180,
            "threshold": 15,
            "min_line_length": 40,
            "max_line_gap": 20,
            "roi_vertices": [[80, 540], [440, 320], [520, 320], [880, 540]],
            "use_perspective_transform": True,
            "smoothing_window": 5,
        },
        "pid_controller": {
            "kp": 0.9,
            "ki": 0.002,
            "kd": 0.3,
            "max_steer_deg": 25.0,
            "max_integral": 100.0,
            "derivative_filter_alpha": 0.2,
        },
        "pipeline": {
            "input_width": 960,
            "input_height": 540,
            "throttle": 0.65,
            "cte_threshold_m": 0.5,
            "temporal_smoothing": True,
            "metrics_window": 50,
        },
    }


def region_of_interest(img: np.ndarray, vertices: List[List[int]]) -> np.ndarray:
    """
    Apply polygonal mask to keep only the road region.
    vertices: list of [x, y] points in clockwise or counter-clockwise order.
    """
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def weighted_img(img: np.ndarray, initial_img: np.ndarray, α: float = 0.8, β: float = 1.0, λ: float = 0.0) -> np.ndarray:
    """Blend two images with weights (for overlaying detections)."""
    return cv2.addWeighted(initial_img, α, img, β, λ)


def perspective_transform(img: np.ndarray, src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute and apply perspective transform (bird's eye view).
    Returns warped image and inverse transform matrix.
    """
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    Minv = cv2.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, Minv


def calculate_curvature_and_offset(left_fit: np.ndarray, right_fit: np.ndarray, 
                                   img_shape: Tuple[int, int], 
                                   xm_per_pix: float = 3.7 / 700, 
                                   ym_per_pix: float = 30 / 720) -> Tuple[float, float, float]:
    """
    Calculate lane curvature (radius) and vehicle offset from center in meters.
    Assumes polynomial fit in pixel space: y = A x^2 + B x + C
    """
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    leftx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Fit in world space (meters)
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Curvature at bottom of image (closest to car)
    y_eval = img_shape[0] * ym_per_pix
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / np.abs(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1])**2)**1.5) / np.abs(2 * right_fit_cr[0])
    avg_curvature = (left_curverad + right_curverad) / 2

    # Vehicle offset from lane center
    lane_center = (leftx[-1] + rightx[-1]) / 2
    vehicle_center = img_shape[1] / 2
    offset_pixels = lane_center - vehicle_center
    offset_m = offset_pixels * xm_per_pix

    return avg_curvature, offset_m, offset_pixels


def draw_hud(frame: np.ndarray, cte: float, steer: float, speed: float = 25.0, 
             confidence: float = 0.92, frame_num: int = 0) -> np.ndarray:
    """Draw professional HUD overlay with key metrics."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"CTE: {cte:+.2f} px  |  Steer: {steer:+.1f}°", (20, 35), font, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Speed: {speed:.1f} mph  |  Conf: {confidence*100:.0f}%  |  Frame: {frame_num}", 
                (20, 65), font, 0.6, (255, 255, 255), 1)

    # Simple steering wheel indicator (circle + line)
    cx, cy = w - 120, 60
    cv2.circle(frame, (cx, cy), 35, (200, 200, 200), 2)
    angle_rad = np.deg2rad(steer)
    end_x = int(cx + 28 * np.sin(angle_rad))
    end_y = int(cy - 28 * np.cos(angle_rad))
    cv2.line(frame, (cx, cy), (end_x, end_y), (0, 255, 0), 3)

    return frame
