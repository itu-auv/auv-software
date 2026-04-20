#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge


def _invalid_result(debug_image=None):
    return {
        "valid": False,
        "center": None,
        "yaw": None,
        "width_px": None,
        "height_px": None,
        "radius_px": None,
        "diameter_px": None,
        "debug_image": debug_image,
    }


def _normalize_line_angle(angle_rad: float) -> float:
    while angle_rad > math.pi / 2:
        angle_rad -= math.pi
    while angle_rad <= -math.pi / 2:
        angle_rad += math.pi
    return angle_rad


def _largest_contour(mask: np.ndarray):
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary, None
    return binary, max(contours, key=cv2.contourArea)


def _base_debug_canvas(mask: np.ndarray, color=(255, 255, 0)):
    binary = (mask > 127).astype(np.uint8) * 255
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(vis)
    overlay[binary > 0] = color
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0.0)
    h, w = binary.shape[:2]
    cv2.arrowedLine(
        vis, (w // 2, h // 2), (w * 3 // 4, h // 2), (255, 0, 0), 2, tipLength=0.3
    )
    cv2.putText(
        vis,
        "Vehicle front",
        (max(8, w // 2 - 40), max(20, h // 2 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return vis


def _geometry_metric_lines(geometry: dict):
    lines = []
    yaw = geometry.get("yaw")
    width_px = geometry.get("width_px")
    height_px = geometry.get("height_px")
    radius_px = geometry.get("radius_px")
    diameter_px = geometry.get("diameter_px")

    if yaw is not None:
        lines.append(f"yaw={math.degrees(yaw):+.1f} deg")
    if width_px is not None and height_px is not None:
        lines.append(f"w={width_px:.1f} h={height_px:.1f}")
    if radius_px is not None and diameter_px is not None:
        lines.append(f"r={radius_px:.1f} d={diameter_px:.1f}")

    return lines


def findposes_rect(mask: np.ndarray, debug: bool = False):
    binary, contour = _largest_contour(mask)
    debug_image = _base_debug_canvas(binary) if debug else None
    if contour is None or len(contour) < 4:
        return _invalid_result(debug_image=debug_image)

    rect = cv2.minAreaRect(contour)
    (cx, cy), (width_px, height_px), _ = rect
    box = cv2.boxPoints(rect).astype(np.float32)

    edges = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        edges.append((length, p1, p2, vec))

    longest_length, edge_start, edge_end, longest_vec = max(edges, key=lambda e: e[0])
    if longest_length <= 0.0:
        return _invalid_result(debug_image=debug_image)

    dx, dy = float(longest_vec[0]), float(longest_vec[1])
    yaw = _normalize_line_angle(math.atan2(-dy, dx))

    result = {
        "valid": True,
        "center": (float(cx), float(cy)),
        "yaw": yaw,
        "width_px": float(width_px),
        "height_px": float(height_px),
        "radius_px": None,
        "diameter_px": None,
        "debug_image": debug_image,
    }

    if debug:
        vis = debug_image
        cv2.drawContours(vis, [box.astype(np.int32)], 0, (0, 255, 0), 2)
        cv2.line(
            vis,
            tuple(edge_start.astype(np.int32)),
            tuple(edge_end.astype(np.int32)),
            (0, 0, 255),
            2,
        )
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (255, 255, 255), -1)

    return result


def findposes_circle(mask: np.ndarray, debug: bool = False):
    binary, contour = _largest_contour(mask)
    debug_image = _base_debug_canvas(binary, color=(0, 255, 255)) if debug else None
    if contour is None or len(contour) < 5:
        return _invalid_result(debug_image=debug_image)

    (cx, cy), radius_px = cv2.minEnclosingCircle(contour)
    if radius_px <= 0.0:
        return _invalid_result(debug_image=debug_image)

    diameter_px = 2.0 * float(radius_px)
    result = {
        "valid": True,
        "center": (float(cx), float(cy)),
        "yaw": None,
        "width_px": None,
        "height_px": None,
        "radius_px": float(radius_px),
        "diameter_px": diameter_px,
        "debug_image": debug_image,
    }

    if debug:
        vis = debug_image
        center = (int(round(cx)), int(round(cy)))
        cv2.circle(vis, center, int(round(radius_px)), (0, 255, 0), 2)
        cv2.circle(vis, center, 4, (255, 255, 255), -1)

    return result


def publish_debug_image(
    publisher,
    header,
    prop_name,
    geometry,
    bbox_center=None,
    bridge=None,
):
    if publisher is None or geometry is None:
        return

    debug_image = geometry.get("debug_image")
    if debug_image is None:
        return

    vis = debug_image.copy()

    center = geometry.get("center")
    if center is not None:
        cv2.circle(
            vis,
            (int(round(center[0])), int(round(center[1]))),
            6,
            (0, 0, 255),
            -1,
        )

    if bbox_center is not None:
        cv2.circle(
            vis,
            (int(round(bbox_center.x)), int(round(bbox_center.y))),
            5,
            (255, 0, 255),
            -1,
        )
        cv2.putText(
            vis,
            "bbox center",
            (8, max(70, vis.shape[0] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        prop_name,
        (8, max(24, vis.shape[0] - 36)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    metrics = _geometry_metric_lines(geometry)
    for i, line in enumerate(metrics):
        cv2.putText(
            vis,
            line,
            (8, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255) if i == 0 else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if bridge is None:
        bridge = CvBridge()

    try:
        out_msg = bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out_msg.header = header
        publisher.publish(out_msg)
    except Exception as e:
        rospy.logwarn_throttle(5.0, f"Failed to publish segment pose debug image: {e}")


def publish_merged_debug_image(publisher, header, debug_items, bridge=None):
    """Publish a single debug image composed from all valid per-object debug canvases.

    Each item in debug_items is expected to have:
    - prop_name: str
    - geometry: dict (with debug_image and center)
    - bbox_center: geometry_msgs/Point (optional)
    """
    if publisher is None or not debug_items:
        return

    valid_items = []
    for item in debug_items:
        geometry = item.get("geometry")
        if geometry is None:
            continue
        debug_image = geometry.get("debug_image")
        if debug_image is None:
            continue
        valid_items.append(
            (
                item.get("prop_name", "unknown"),
                geometry,
                item.get("bbox_center"),
                debug_image,
            )
        )

    if not valid_items:
        return

    vis = np.zeros_like(valid_items[0][3])
    for _, _, _, debug_image in valid_items:
        if debug_image.shape != vis.shape:
            continue
        vis = np.maximum(vis, debug_image)

    colors = [
        (0, 255, 255),
        (255, 128, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 255, 255),
    ]

    for idx, (prop_name, geometry, bbox_center, _) in enumerate(valid_items):
        color = colors[idx % len(colors)]

        center = geometry.get("center")
        if center is not None:
            cv2.circle(
                vis,
                (int(round(center[0])), int(round(center[1]))),
                6,
                color,
                -1,
            )

        if bbox_center is not None:
            cv2.circle(
                vis,
                (int(round(bbox_center.x)), int(round(bbox_center.y))),
                4,
                color,
                1,
            )

        label_y = min(22 + idx * 44, vis.shape[0] - 8)
        cv2.putText(
            vis,
            prop_name,
            (8, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        metrics = _geometry_metric_lines(geometry)
        for line_idx, line in enumerate(metrics[:2]):
            metric_y = min(label_y + 16 + line_idx * 16, vis.shape[0] - 8)
            metric_color = (0, 255, 255) if line_idx == 0 else (0, 255, 0)
            cv2.putText(
                vis,
                line,
                (18, metric_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                metric_color,
                1,
                cv2.LINE_AA,
            )

    if bridge is None:
        bridge = CvBridge()

    try:
        out_msg = bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out_msg.header = header
        publisher.publish(out_msg)
    except Exception as e:
        rospy.logwarn_throttle(
            5.0, f"Failed to publish merged segment pose debug image: {e}"
        )
