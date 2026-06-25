#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Target:
    valid: bool = False
    center_error: float = 0.0
    gate_width_ratio: float = 0.0
    gate_height_ratio: float = 0.0
    red_center_x: float = 0.0
    white_center_x: float = 0.0
    confidence: float = 0.0


@dataclass
class ControllerConfig:
    yaw_kp: float = 12.0
    yaw_kd: float = 2.0
    max_yaw_torque: float = 8.0
    forward_force: float = 12.0
    exit_force: float = 10.0
    align_error_threshold: float = 0.22
    near_height_ratio: float = 0.58
    near_width_ratio: float = 0.55
    pass_loss_duration: float = 0.45
    search_yaw_torque: float = 2.5
    exit_duration: float = 1.5
    yaw_sign: float = -1.0


@dataclass(frozen=True)
class ControlCommand:
    force_x: float
    torque_z: float
    state: str
    gates_passed: int
    finished: bool = False


class SlalomController:
    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self.reset()

    def reset(self, direction: str = "left", target_gate_count: int = 3) -> None:
        self.direction = direction if direction in ("left", "right") else "left"
        self.target_gate_count = max(1, int(target_gate_count))
        self.gates_passed = 0
        self.near_gate = False
        self.lost_since = None
        self.exit_started = None
        self.last_error = 0.0
        self.last_error_time = None

    def update(self, now: float, target: Target) -> ControlCommand:
        if self.gates_passed >= self.target_gate_count:
            if self.exit_started is None:
                self.exit_started = now
            finished = now - self.exit_started >= self.config.exit_duration
            return ControlCommand(
                0.0 if finished else self.config.exit_force,
                0.0,
                "FINISHED" if finished else "EXIT",
                self.gates_passed,
                finished,
            )

        if not target.valid:
            if self.near_gate:
                if self.lost_since is None:
                    self.lost_since = now
                elif now - self.lost_since >= self.config.pass_loss_duration:
                    self.gates_passed += 1
                    self.near_gate = False
                    self.lost_since = None
                    self.last_error_time = None
                    return self.update(now, Target())
            search_sign = -1.0 if self.direction == "left" else 1.0
            if self.gates_passed % 2:
                search_sign *= -1.0
            return ControlCommand(
                0.0,
                search_sign * self.config.search_yaw_torque,
                "PASS_CONFIRM" if self.near_gate else "SEARCH",
                self.gates_passed,
            )

        self.lost_since = None
        self.near_gate = self.near_gate or (
            target.gate_height_ratio >= self.config.near_height_ratio
            or target.gate_width_ratio >= self.config.near_width_ratio
        )

        derivative = 0.0
        if self.last_error_time is not None:
            dt = now - self.last_error_time
            if dt > 1e-3:
                derivative = (target.center_error - self.last_error) / dt
        self.last_error = target.center_error
        self.last_error_time = now

        yaw_torque = self.config.yaw_sign * (
            self.config.yaw_kp * target.center_error + self.config.yaw_kd * derivative
        )
        yaw_torque = max(
            -self.config.max_yaw_torque,
            min(self.config.max_yaw_torque, yaw_torque),
        )
        aligned = abs(target.center_error) <= self.config.align_error_threshold
        return ControlCommand(
            self.config.forward_force if aligned else 0.0,
            yaw_torque,
            "ADVANCE" if aligned else "ALIGN",
            self.gates_passed,
        )


def overlay_wrench(
    nominal: Sequence[float],
    visual: Sequence[float],
    active: bool,
    visual_fresh: bool,
) -> Tuple[float, float, float, float, float, float]:
    """Apply visual ownership of Fx/Fy/Tz while preserving depth and attitude."""
    if len(nominal) != 6 or len(visual) != 6:
        raise ValueError("wrench vectors must contain six elements")
    if not active:
        return tuple(nominal)
    output = list(nominal)
    output[0] = visual[0] if visual_fresh else 0.0
    output[1] = 0.0
    output[5] = visual[5] if visual_fresh else 0.0
    return tuple(output)
