"""
Fresh Lane Detector implementation for Task 5.
Robust Canny + Hough pipeline with optional perspective transform and temporal smoothing.
Designed to be production-like and easy to extend.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from collections import deque
import logging

from utils import region_of_interest, perspective_transform, calculate_curvature_and_offset

logger = logging.getLogger(__name__)


class LaneDetector:
    def __init__(self, config: dict):
        self.cfg = config.get("lane_detector", {})
        self.smoothing_window = self.cfg.get("smoothing_window", 5)
        self.left_lane_history = deque(maxlen=self.smoothing_window)
        self.right_lane_history = deque(maxlen=self.smoothing_window)
        self.use_perspective = self.cfg.get("use_perspective_transform", True)

        # Perspective matrices (will be computed on first frame if needed)
        self.M = None
        self.Minv = None
        self.last_frame_shape = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply Gaussian blur."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = self.cfg.get("gaussian_kernel", 5)
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        return blurred

    def _edge_detection(self, img: np.ndarray) -> np.ndarray:
        """Canny edge detection with configurable thresholds."""
        low = self.cfg.get("canny_low", 50)
        high = self.cfg.get("canny_high", 150)
        return cv2.Canny(img, low, high)

    def _get_roi(self, img: np.ndarray) -> np.ndarray:
        """Apply region of interest mask."""
        h, w = img.shape[:2]
        # Scale vertices if they were defined for 960x540
        vertices = np.array(self.cfg.get("roi_vertices", [[80, 540], [440, 320], [520, 320], [880, 540]]))
        if h != 540 or w != 960:
            scale_x = w / 960.0
            scale_y = h / 540.0
            vertices = vertices * np.array([scale_x, scale_y])
        return region_of_interest(img, vertices.astype(int).tolist())

    def _hough_lines(self, edges: np.ndarray) -> np.ndarray:
        """Probabilistic Hough transform."""
        rho = self.cfg.get("rho", 2)
        theta = self.cfg.get("theta", np.pi / 180)
        threshold = self.cfg.get("threshold", 15)
        min_line_len = self.cfg.get("min_line_length", 40)
        max_line_gap = self.cfg.get("max_line_gap", 20)

        lines = cv2.HoughLinesP(
            edges, rho, theta, threshold,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap
        )
        return lines if lines is not None else np.array([])

    def _classify_and_fit_lines(self, lines: np.ndarray, img_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Separate left/right lines by slope, filter outliers, fit 2nd degree polynomial.
        Returns (left_fit, right_fit) or None if insufficient data.
        """
        if len(lines) == 0:
            return None, None

        left_points = []
        right_points = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Filter reasonable road slopes (avoid horizontal/vertical noise)
            if 0.4 < abs(slope) < 2.5:
                if slope < 0:  # left lane (negative slope in image coords)
                    left_points.extend([(x1, y1), (x2, y2)])
                else:
                    right_points.extend([(x1, y1), (x2, y2)])

        def fit_poly(points: List[Tuple[int, int]]) -> Optional[np.ndarray]:
            if len(points) < 10:
                return None
            pts = np.array(points)
            # RANSAC-like: fit and remove outliers iteratively (simple version)
            for _ in range(2):
                if len(pts) < 8:
                    return None
                fit = np.polyfit(pts[:, 1], pts[:, 0], 2)  # x = f(y)
                residuals = np.abs(pts[:, 0] - np.polyval(fit, pts[:, 1]))
                pts = pts[residuals < 30]  # keep inliers within 30 px
            return fit if len(pts) >= 8 else None

        left_fit = fit_poly(left_points)
        right_fit = fit_poly(right_points)

        # Temporal smoothing
        if left_fit is not None:
            self.left_lane_history.append(left_fit)
            left_fit = np.mean(self.left_lane_history, axis=0)
        if right_fit is not None:
            self.right_lane_history.append(right_fit)
            right_fit = np.mean(self.right_lane_history, axis=0)

        return left_fit, right_fit

    def detect(self, frame: np.ndarray) -> dict:
        """
        Main detection method.
        Returns dict with:
            - left_fit, right_fit (polynomial coeffs)
            - curvature (m), offset_m, offset_px
            - visualization images (edges, warped, etc.)
            - confidence (simple heuristic)
        """
        h, w = frame.shape[:2]
        self.last_frame_shape = (h, w)

        # 1. Preprocess & edges
        preprocessed = self._preprocess(frame)
        edges = self._edge_detection(preprocessed)
        masked_edges = self._get_roi(edges)

        # 2. Optional perspective warp for better polynomial fit
        if self.use_perspective and self.M is None:
            # Default source/destination points (tuned for typical dashcam)
            src = np.float32([[200, h], [w-200, h], [w//2-60, h//2+80], [w//2+60, h//2+80]])
            dst = np.float32([[300, h], [w-300, h], [300, 0], [w-300, 0]])
            self.M, self.Minv = cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)

        if self.use_perspective and self.M is not None:
            warped_edges = cv2.warpPerspective(masked_edges, self.M, (w, h))
            lines = self._hough_lines(warped_edges)
        else:
            lines = self._hough_lines(masked_edges)
            warped_edges = None

        # 3. Fit lanes
        left_fit, right_fit = self._classify_and_fit_lines(lines, (h, w))

        # 4. Compute metrics
        curvature, offset_m, offset_px = 0.0, 0.0, 0.0
        if left_fit is not None and right_fit is not None:
            curvature, offset_m, offset_px = calculate_curvature_and_offset(
                left_fit, right_fit, (h, w)
            )

        # 5. Confidence heuristic
        confidence = 0.5
        if left_fit is not None and right_fit is not None:
            confidence = min(0.95, 0.6 + 0.35 * (1 - abs(offset_m) / 1.5))

        result = {
            "left_fit": left_fit,
            "right_fit": right_fit,
            "curvature": curvature,
            "offset_m": offset_m,
            "offset_px": offset_px,
            "confidence": confidence,
            "edges": masked_edges,
            "warped_edges": warped_edges if self.use_perspective else None,
            "frame_shape": (h, w),
        }
        return result

    def draw_lanes(self, frame: np.ndarray, result: dict, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw detected lane lines and road area on original frame."""
        if result["left_fit"] is None or result["right_fit"] is None:
            return frame

        h, w = frame.shape[:2]
        left_fit = result["left_fit"]
        right_fit = result["right_fit"]

        ploty = np.linspace(0, h - 1, h)
        leftx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Create lane overlay
        lane_overlay = np.zeros_like(frame)
        pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(lane_overlay, np.int_([pts]), (0, 255, 0))
        cv2.polylines(lane_overlay, np.int32([pts_left]), isClosed=False, color=color, thickness=8)
        cv2.polylines(lane_overlay, np.int32([pts_right]), isClosed=False, color=color, thickness=8)

        # Blend
        result_frame = cv2.addWeighted(frame, 1, lane_overlay, 0.35, 0)

        # Add curvature text
        cv2.putText(result_frame, f"Curvature: {result['curvature']:.0f} m", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return result_frame
