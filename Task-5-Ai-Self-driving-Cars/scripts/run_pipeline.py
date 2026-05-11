#!/usr/bin/env python3
"""
Main entry point for Task 5 — AI Self-Driving Cars.
Flexible CLI for running the full autonomous driving stack.

Usage examples:
    python scripts/run_pipeline.py --input data/test_videos/solidWhiteRight.mp4 --output output/demo.mp4 --show-preview
    python scripts/run_pipeline.py --mode simulation --duration 60
    python scripts/run_pipeline.py --input data/test_images/solidYellowCurve.jpg --mode image
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from control_simulator import AutonomousDrivingStack
from utils import load_config, draw_hud

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("run_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 5: AI Self-Driving Cars — Lane Detection + PID Control Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to input video or image")
    parser.add_argument("--output", "-o", type=str, default="output/autonomous_demo.mp4",
                        help="Path for annotated output")
    parser.add_argument("--mode", "-m", choices=["video", "image", "simulation"], default="video",
                        help="Processing mode")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--show-preview", "-p", action="store_true",
                        help="Show live preview window (press 'q' to quit)")
    parser.add_argument("--duration", type=float, default=45.0,
                        help="Duration in seconds for simulation mode")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    logger.info("=== Task 5: AI Self-Driving Cars — Starting ===")
    stack = AutonomousDrivingStack(args.config)

    if args.mode == "video":
        if not args.input:
            args.input = "data/test_videos/solidWhiteRight.mp4"
            logger.info(f"No input specified, using default: {args.input}")
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return 1

        metrics = stack.process_video(args.input, args.output, show_preview=args.show_preview)
        stack.print_report(metrics)
        logger.info(f"✅ Annotated video saved to: {args.output}")

    elif args.mode == "image":
        if not args.input:
            args.input = "data/test_images/solidYellowLeft.jpg"
        import cv2
        frame = cv2.imread(args.input)
        if frame is None:
            logger.error(f"Cannot read image: {args.input}")
            return 1

        perception = stack.perception.process_frame(frame)
        vis = stack.perception.draw_perception(frame, perception)
        cte = perception["lanes"]["offset_px"]
        steer = stack.pid.update(cte)
        vis = draw_hud(vis, cte, steer, frame_num=0)  # from utils

        cv2.imwrite(args.output, vis)
        logger.info(f"✅ Annotated image saved to: {args.output}")
        stack.print_report({"single_frame_cte": cte, "steer_cmd": steer})

    elif args.mode == "simulation":
        metrics = stack.run_simulation(args.duration)
        stack.print_report(metrics)
        logger.info("✅ Pure simulation complete (great for PID tuning)")

    logger.info("=== Task 5 Complete — Ready for GitHub & Internship Submission ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
