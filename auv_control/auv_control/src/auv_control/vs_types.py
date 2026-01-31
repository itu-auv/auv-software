from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any


@dataclass
class ControllerConfig:
    kp_gain: float = 0.8
    kd_gain: float = 0.4
    v_x_desired: float = 0.3
    rate_hz: float = 10.0
    overall_timeout_s: float = 1500.0
    navigation_timeout_s: float = 12.0
    max_angular_velocity: float = 1.0
    max_lateral_velocity: float = 0.3
    slalom_mode: bool = False
    slalom_side: str = "left"  # "left" or "right"
    kp_lateral: float = 0.5
    kd_lateral: float = 0.2
    lateral_deadzone: float = 2.0


@dataclass
class ErrorState:
    error: float = 0.0
    error_derivative: float = 0.0
    last_stamp: Optional[Any] = None  # rospy.Time

    def reset(self):
        self.error = 0.0
        self.error_derivative = 0.0
        self.last_stamp = None


@dataclass
class SlalomState:
    detections: List[Any] = field(default_factory=list)  # List[PipeDetection]
    selected_pair: Optional[tuple] = None  # Tuple[PipeDetection, PipeDetection]
    lateral_error: float = 0.0
    last_stamp: Optional[Any] = None  # rospy.Time

    def reset(self):
        self.detections = []
        self.selected_pair = None
        self.lateral_error = 0.0
        self.last_stamp = None


class ControllerState(Enum):
    IDLE = "idle"
    CENTERING = "centering"
    NAVIGATING = "navigating"
