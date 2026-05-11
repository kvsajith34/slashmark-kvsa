#!/usr/bin/env python3
"""
AetherNav - Standalone Simulation Runner
Launches the full AI-powered indoor navigation demo with Pygame.
This is the primary way to experience the system locally.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import time
from app.simulation.indoor_sim import IndoorSimulator
from app.perception.detector import ObstacleDetector
from app.navigation.planner import NavigationController

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ai_controller(state: dict) -> dict:
    """Main AI decision loop - Perception → Navigation → Action"""
    # Perception (already simulated in state, but we can enhance)
    detector = ai_controller.detector  # attached below
    
    # Navigation
    action = ai_controller.navigator.compute_action(state)
    
    # Add some telemetry flavor
    action['ai_confidence'] = action.get('confidence', 0.85)
    action['timestamp'] = time.time()
    
    return action

def main():
    print("=" * 60)
    print("🚀 AETHERNAV v1.0 — AI Indoor Obstacle Avoidance")
    print("   Advanced Robotics Portfolio Project")
    print("=" * 60)
    
    config = load_config()
    
    # Initialize modules
    detector = ObstacleDetector(config['perception'])
    navigator = NavigationController(config)
    
    # Attach to closure for controller
    ai_controller.detector = detector
    ai_controller.navigator = navigator
    
    # Launch simulator
    sim = IndoorSimulator(config['simulation'])
    
    print("\n✅ All systems initialized. Starting simulation...")
    print("   Controls: SPACE=pause | R=reset env | ESC=quit\n")
    
    try:
        sim.run(controller_callback=ai_controller)
    except KeyboardInterrupt:
        print("\n👋 Shutting down AetherNav gracefully...")

if __name__ == "__main__":
    main()