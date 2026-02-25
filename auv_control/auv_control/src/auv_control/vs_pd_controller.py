from dataclasses import dataclass
from typing import Tuple

from auv_control.vs_types import ControllerConfig, ControllerState, ErrorState


@dataclass
class PDController:
    config: ControllerConfig
    error_state: ErrorState

    def compute_angular(self, state: ControllerState) -> float:
        """PD control for angular velocity. Returns clamped command."""
        error = self.error_state.error
        d_term = self.config.kd_gain * self.error_state.error_derivative

        p_term = self.config.kp_gain * error
        cmd = p_term - d_term

        return max(
            min(cmd, self.config.max_angular_velocity),
            -self.config.max_angular_velocity,
        )

    def compute_linear(
        self, state: ControllerState, time_since_detection: float
    ) -> float:
        """Returns linear velocity. Zero if not navigating or detection timeout."""
        if state != ControllerState.NAVIGATING:
            return 0.0

        if time_since_detection > self.config.navigation_timeout_s:
            return 0.0

        return self.config.v_x_desired

    def update_error(self, angle: float, now_sec: float):
        """Update error state with new measurement."""
        if self.error_state.last_stamp is not None:
            dt = now_sec - self.error_state.last_stamp
            if dt > 0.001:
                self.error_state.error_derivative = (
                    angle - self.error_state.error
                ) / dt

        self.error_state.error = angle
        self.error_state.last_stamp = now_sec
