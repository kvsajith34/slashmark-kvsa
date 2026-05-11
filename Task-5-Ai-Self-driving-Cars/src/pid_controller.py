"""
Production-grade PID Controller for lateral vehicle control.
Includes anti-windup, derivative filtering, and output clamping.
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PIDController:
    def __init__(self, config: dict):
        pid_cfg = config.get("pid_controller", {})
        self.kp = pid_cfg.get("kp", 0.9)
        self.ki = pid_cfg.get("ki", 0.002)
        self.kd = pid_cfg.get("kd", 0.3)
        self.max_steer = pid_cfg.get("max_steer_deg", 25.0)
        self.max_integral = pid_cfg.get("max_integral", 100.0)
        self.alpha = pid_cfg.get("derivative_filter_alpha", 0.2)  # for derivative filter

        # State
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.dt = 0.05  # default timestep (20 Hz)

    def reset(self):
        """Reset internal state (useful between runs)."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def update(self, error: float, dt: Optional[float] = None) -> float:
        """
        Compute PID output (steering angle in degrees).
        error: cross-track error (positive = car is to the right of center)
        """
        if dt is None:
            dt = self.dt

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        i_term = self.ki * self.integral

        # Derivative (filtered)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        filtered_deriv = self.alpha * derivative + (1 - self.alpha) * self.prev_derivative
        d_term = self.kd * filtered_deriv

        # Total control
        steer = -(p_term + i_term + d_term)  # negative because positive error needs left steer

        # Clamp output
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        # Update state
        self.prev_error = error
        self.prev_derivative = filtered_deriv

        return float(steer)

    def get_gains(self) -> dict:
        return {"kp": self.kp, "ki": self.ki, "kd": self.kd}
