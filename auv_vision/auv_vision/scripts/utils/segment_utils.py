#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage


SEGMENT_DEBUG_COLORS = {
    "bandaid_link": (128, 0, 0),
    "electric_link": (180, 105, 255),
    "nutbolt_link": (200, 200, 200),
    "pill_link": (0, 165, 255),
    "basket_redcross_segment_link": (0, 0, 255),
    "octagon_table_segment_link": (255, 200, 80),
    "basket_warning_segment_link": (0, 255, 255),
}

SEGMENT_DEBUG_PALETTE = (
    (0, 128, 255),
    (255, 0, 255),
    (0, 255, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 128, 0),
    (255, 255, 0),
    (128, 0, 255),
)


def get_segment_debug_color(prop_name: str):
    if prop_name in SEGMENT_DEBUG_COLORS:
        return SEGMENT_DEBUG_COLORS[prop_name]

    stable_index = sum((i + 1) * ord(ch) for i, ch in enumerate(prop_name))
    return SEGMENT_DEBUG_PALETTE[stable_index % len(SEGMENT_DEBUG_PALETTE)]


def _invalid_result(debug_image=None):
    return {
        "valid": False,
        "center": None,
        "yaw": None,
        "edges_px": None,
        "radius_px": None,
        "diameter_px": None,
        "debug_image": debug_image,
        "debug_mask": None,
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
    vis = np.zeros((*binary.shape[:2], 3), dtype=np.uint8)
    vis[binary > 0] = color
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
    # longest -> height, shortest -> width
    height_px, width_px = geometry.get("edges_px")
    radius_px = geometry.get("radius_px")
    diameter_px = geometry.get("diameter_px")

    if yaw is not None:
        lines.append(f"yaw={math.degrees(yaw):+.1f} deg")
    if width_px is not None and height_px is not None:
        lines.append(f"w={width_px:.1f} h={height_px:.1f}")
    if radius_px is not None and diameter_px is not None:
        lines.append(f"r={radius_px:.1f} d={diameter_px:.1f}")

    return lines


def _cv2_to_compressed_msg(image: np.ndarray, header, fmt: str = "jpeg"):
    msg = CompressedImage()
    msg.header = header
    msg.format = fmt

    ext = ".jpg" if fmt in ("jpeg", "jpg") else f".{fmt}"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {fmt}")

    msg.data = encoded.tobytes()
    return msg


def findposes_rect(
    mask: np.ndarray,
    last_yaw: float = None,
    debug: bool = False,
    debug_color=(255, 255, 0),
):
    binary, contour = _largest_contour(mask)
    debug_image = _base_debug_canvas(binary, color=debug_color) if debug else None
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
    shortest_length, _, _, _ = min(edges, key=lambda e: e[0])

    dx, dy = float(longest_vec[0]), float(longest_vec[1])
    yaw_base = math.atan2(-dy, dx)

    if last_yaw is not None:
        candidates = [yaw_base + i * (math.pi / 2) for i in range(4)]
        yaw = min(
            candidates,
            key=lambda a: abs(
                math.atan2(math.sin(a - last_yaw), math.cos(a - last_yaw))
            ),
        )
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    else:
        yaw = _normalize_line_angle(yaw_base)

    result = {
        "valid": True,
        "center": (float(cx), float(cy)),
        "yaw": yaw,
        "edges_px": (float(longest_length), float(shortest_length)),
        "radius_px": None,
        "diameter_px": None,
        "debug_image": debug_image,
        "debug_mask": binary if debug else None,
    }

    if debug:
        vis = debug_image
        cv2.drawContours(vis, [box.astype(np.int32)], 0, debug_color, 2)
        cv2.line(
            vis,
            tuple(edge_start.astype(np.int32)),
            tuple(edge_end.astype(np.int32)),
            (255, 255, 255),
            2,
        )
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, debug_color, -1)

    return result


def findposes_circle(
    mask: np.ndarray,
    last_yaw: float = None,
    debug: bool = False,
    debug_color=(0, 255, 255),
):
    binary, contour = _largest_contour(mask)
    debug_image = _base_debug_canvas(binary, color=debug_color) if debug else None
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
        "edges_px": (None, None),
        "radius_px": float(radius_px),
        "diameter_px": diameter_px,
        "debug_image": debug_image,
        "debug_mask": binary if debug else None,
    }

    if debug:
        vis = debug_image
        center = (int(round(cx)), int(round(cy)))
        cv2.circle(vis, center, int(round(radius_px)), debug_color, 2)
        cv2.circle(vis, center, 4, debug_color, -1)

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
    debug_color = get_segment_debug_color(prop_name)

    center = geometry.get("center")
    if center is not None:
        cv2.circle(
            vis,
            (int(round(center[0])), int(round(center[1]))),
            6,
            debug_color,
            -1,
        )

    if bbox_center is not None:
        cv2.circle(
            vis,
            (int(round(bbox_center.x)), int(round(bbox_center.y))),
            5,
            debug_color,
            -1,
        )
        cv2.putText(
            vis,
            "bbox center",
            (8, max(70, vis.shape[0] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            debug_color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        prop_name,
        (8, max(24, vis.shape[0] - 36)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        debug_color,
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
            debug_color,
            2,
            cv2.LINE_AA,
        )

    try:
        out_msg = _cv2_to_compressed_msg(vis, header)
        publisher.publish(out_msg)
    except Exception as e:
        rospy.logwarn_throttle(
            5.0, f"Failed to publish compressed segment pose debug image: {e}"
        )


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
        debug_mask = geometry.get("debug_mask")
        valid_items.append(
            (
                item.get("prop_name", "unknown"),
                geometry,
                item.get("bbox_center"),
                debug_image,
                item.get("debug_color"),
                debug_mask,
            )
        )

    if not valid_items:
        return

    vis = np.zeros_like(valid_items[0][3])
    layer_items = sorted(
        valid_items,
        key=lambda item: cv2.countNonZero(item[5]) if item[5] is not None else 0,
        reverse=True,
    )
    for prop_name, _, _, debug_image, debug_color, debug_mask in layer_items:
        if debug_image.shape != vis.shape:
            continue
        color = debug_color or get_segment_debug_color(prop_name)
        if debug_mask is None:
            vis = np.maximum(vis, debug_image)
        else:
            vis[debug_mask > 0] = color

    for idx, (prop_name, geometry, bbox_center, _, debug_color, _) in enumerate(
        valid_items
    ):
        color = debug_color or get_segment_debug_color(prop_name)

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
            cv2.putText(
                vis,
                line,
                (18, metric_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    try:
        out_msg = _cv2_to_compressed_msg(vis, header)
        publisher.publish(out_msg)
    except Exception as e:
        rospy.logwarn_throttle(
            5.0, f"Failed to publish compressed merged segment pose debug image: {e}"
        )
