"""
Full pipeline: Perception → Control Simulation → Visualization + Metrics.
This is the heart of the autonomous driving stack demo.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm
import os

from perception import RoadPerception
from pid_controller import PIDController
from utils import draw_hud, load_config

logger = logging.getLogger(__name__)


class AutonomousDrivingStack:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.perception = RoadPerception(config_path)
        self.pid = PIDController(self.config)
        self.metrics = {
            "cte_history": [],
            "steer_history": [],
            "confidence_history": [],
        }
        self.frame_idx = 0

    def process_video(self, input_path: str, output_path: str, show_preview: bool = False) -> Dict[str, float]:
        """
        Process entire video: detect lanes, run PID, annotate, save output.
        Returns evaluation metrics.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.config["pipeline"]["output_fps"]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logger.info(f"Processing {total_frames} frames from {input_path}...")

        pbar = tqdm(total=total_frames, desc="Processing video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_idx += 1

            # Perception
            perception = self.perception.process_frame(frame)

            # Control
            cte = perception["lanes"]["offset_px"]
            steer = self.pid.update(cte)

            # Record metrics
            self.metrics["cte_history"].append(cte)
            self.metrics["steer_history"].append(steer)
            self.metrics["confidence_history"].append(perception["confidence"])

            # Visualization
            vis_frame = self.perception.draw_perception(frame, perception)
            vis_frame = draw_hud(
                vis_frame,
                cte=cte,
                steer=steer,
                speed=self.config["pipeline"]["throttle"] * 50,  # fake speed
                confidence=perception["confidence"],
                frame_num=self.frame_idx
            )

            # Add steering command text
            cv2.putText(vis_frame, f"PID Steer: {steer:+.1f}°", (width - 280, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(vis_frame)

            if show_preview:
                cv2.imshow("Autonomous Driving Stack - Task 5", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Compute final metrics
        return self._compute_metrics()

    def run_simulation(self, duration_sec: float = 45.0) -> Dict[str, float]:
        """
        Pure simulation mode using synthetic track (no video).
        Useful for rapid PID tuning.
        """
        logger.info("Running pure control simulation...")
        self.pid.reset()
        self.metrics = {"cte_history": [], "steer_history": [], "confidence_history": []}

        dt = 0.05
        steps = int(duration_sec / dt)
        # Simple sinusoidal "lane" deviation for testing
        for i in range(steps):
            t = i * dt
            synthetic_cte = 80 * np.sin(0.3 * t) + 20 * np.sin(1.2 * t)  # realistic disturbance
            steer = self.pid.update(synthetic_cte, dt=dt)

            self.metrics["cte_history"].append(synthetic_cte)
            self.metrics["steer_history"].append(steer)
            self.metrics["confidence_history"].append(0.85)

        return self._compute_metrics()

    def _compute_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        ctes = np.array(self.metrics["cte_history"])
        steers = np.array(self.metrics["steer_history"])

        if len(ctes) == 0:
            return {"error": "No frames processed"}

        avg_cte = np.mean(np.abs(ctes))
        max_cte = np.max(np.abs(ctes))
        lane_keeping = np.mean(np.abs(ctes) < self.config["pipeline"]["cte_threshold_m"] * 40) * 100  # rough px conversion
        steer_smoothness = np.std(steers)

        report = {
            "frames": len(ctes),
            "avg_abs_cte_px": round(float(avg_cte), 2),
            "max_cte_px": round(float(max_cte), 2),
            "lane_keeping_pct": round(float(lane_keeping), 1),
            "steering_smoothness_deg": round(float(steer_smoothness), 2),
            "avg_confidence": round(float(np.mean(self.metrics["confidence_history"])), 3),
        }
        return report

    def print_report(self, metrics: Dict[str, float]):
        """Pretty print evaluation report."""
        print("\n" + "=" * 50)
        print("🚗 AUTONOMOUS DRIVING STACK — EVALUATION REPORT")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"{k:25s}: {v}")
        print("=" * 50 + "\n")
