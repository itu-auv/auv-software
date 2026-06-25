#!/usr/bin/env python3

"""Pure image-space target selection shared by the ROS node and unit tests."""

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class Target:
    valid: bool = False
    center_error: float = 0.0
    gate_width_ratio: float = 0.0
    gate_height_ratio: float = 0.0
    red_center_x: float = 0.0
    white_center_x: float = 0.0
    confidence: float = 0.0


def select_target(
    detections: Iterable[Detection],
    image_width: float,
    image_height: float,
    direction: str,
    red_class_id: int = 2,
    white_class_id: int = 3,
    min_confidence: float = 0.25,
    min_separation_ratio: float = 0.04,
    max_separation_ratio: float = 0.95,
) -> Target:
    if image_width <= 0.0 or image_height <= 0.0:
        return Target()

    valid = [
        detection
        for detection in detections
        if detection.confidence >= min_confidence
        and detection.width > 0.0
        and detection.height > 0.0
    ]
    reds = [d for d in valid if d.class_id == red_class_id]
    whites = [d for d in valid if d.class_id == white_class_id]
    candidates = []
    for red in reds:
        for white in whites:
            separation = abs(white.center_x - red.center_x) / image_width
            if not min_separation_ratio <= separation <= max_separation_ratio:
                continue
            ordered = (
                white.center_x < red.center_x
                if direction.lower() == "left"
                else white.center_x > red.center_x
            )
            if not ordered:
                continue
            size_similarity = min(red.height, white.height) / max(
                red.height, white.height
            )
            vertical_offset = abs(red.center_y - white.center_y) / max(
                red.height, white.height
            )
            vertical_similarity = max(0.0, 1.0 - vertical_offset)
            score = (
                math.sqrt(red.area * white.area)
                * min(red.confidence, white.confidence)
                * size_similarity**2
                * vertical_similarity
                / (1.0 + separation)
            )
            candidates.append((score, red, white))

    if not candidates:
        return Target()

    _, red, white = max(candidates, key=lambda item: item[0])
    gate_center_x = 0.5 * (red.center_x + white.center_x)
    opening_width = abs(red.center_x - white.center_x) + 0.5 * (red.width + white.width)
    return Target(
        valid=True,
        center_error=max(
            -1.0,
            min(1.0, (gate_center_x - image_width * 0.5) / (image_width * 0.5)),
        ),
        gate_width_ratio=opening_width / image_width,
        gate_height_ratio=max(red.height, white.height) / image_height,
        red_center_x=red.center_x / image_width,
        white_center_x=white.center_x / image_width,
        confidence=min(red.confidence, white.confidence),
    )
