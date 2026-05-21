#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, deque
import math
import os
import re
import sys

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from tf import transformations as tf_transformations
import tf2_ros
from ultralytics_ros.msg import YoloResult

# Add scripts directory to path to import detection_utils and segment_utils
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from utils.detection_utils import load_config
from utils.segment_utils import findposes_circle, findposes_rect


class SegmentDebugImageNode:
    def __init__(self):
        rospy.init_node("segment_debug_image_node", anonymous=True)
        rospy.loginfo("Segment debug image node starting...")

        self.bridge = CvBridge()
        self.last_yaws = {}

        # Distinct BGR color palette for detected segment objects
        self.COLOR_PALETTE = {
            "bandaid_link": (0, 255, 255),  # Cyan
            "electric_link": (0, 165, 255),  # Orange
            "nutbolt_link": (0, 255, 0),  # Green
            "pill_link": (255, 0, 255),  # Magenta
            "basket_redcross_segment_link": (0, 0, 255),  # Red
            "octagon_table_segment_link": (255, 255, 0),  # Yellow
            "basket_warning_segment_link": (0, 128, 255),  # Light Blue
        }

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Load ID to TF frame map from config if available, else use fallback
        config_file = rospy.get_param("~config_file", "")
        if config_file and os.path.exists(config_file):
            try:
                config = load_config(config_file)
                bottom_seg_cfg = config.get("cameras", {}).get("bottom_seg", {})
                self.id_tf_map = {
                    int(k): v for k, v in bottom_seg_cfg.get("id_tf_map", {}).items()
                }
                rospy.loginfo(
                    f"Successfully loaded id_tf_map from config: {self.id_tf_map}"
                )
            except Exception as e:
                rospy.logerr(f"Failed to parse config file: {e}. Using fallback map.")
                self._load_fallback_map()
        else:
            self._load_fallback_map()

        # Publisher for the annotated compressed debug image
        self.pub_debug = rospy.Publisher(
            "debug_image/compressed", CompressedImage, queue_size=1
        )

        # Synchronized subscribers
        self.sub_img = message_filters.Subscriber(
            "image_compressed", CompressedImage, buff_size=2**24
        )
        self.sub_yolo = message_filters.Subscriber(
            "yolo_result", YoloResult, buff_size=2**24
        )

        # ApproximateTimeSynchronizer to align camera frames with YOLO segmentation results
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_yolo], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.callback)

        rospy.loginfo("Segment debug image node initialized and ready.")

    def _load_fallback_map(self):
        self.id_tf_map = {
            0: "bandaid_link",
            1: "electric_link",
            2: "nutbolt_link",
            3: "pill_link",
            4: "basket_redcross_segment_link",
            5: "octagon_table_segment_link",
            6: "basket_warning_segment_link",
        }
        rospy.loginfo(f"Loaded fallback id_tf_map: {self.id_tf_map}")

    def _extract_mask_id(self, mask_msg):
        frame_id = (mask_msg.header.frame_id or "").strip()
        if not frame_id:
            return None
        if frame_id.isdigit() or (frame_id.startswith("-") and frame_id[1:].isdigit()):
            return int(frame_id)
        match = re.search(r"-?\d+", frame_id)
        if match is None:
            return None
        return int(match.group(0))

    def _mask_to_cv2(self, mask_msg):
        try:
            return self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Failed to decode segmentation mask: {e}")
            return None

    def callback(self, img_msg: CompressedImage, yolo_msg: YoloResult):
        try:
            np_arr = np.frombuffer(img_msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Failed to decode compressed image: {e}")
            return

        if cv_img is None:
            return

        masks_msgs = list(yolo_msg.masks) if yolo_msg.masks else []
        masks_by_id = defaultdict(deque)

        for mask_msg in masks_msgs:
            mask_id = self._extract_mask_id(mask_msg)
            if mask_id is None:
                continue
            masks_by_id[mask_id].append(mask_msg)

        for detection, mask_msg in zip(yolo_msg.detections.detections, masks_msgs):
            if len(detection.results) == 0:
                continue
            if self._extract_mask_id(mask_msg) is None:
                masks_by_id[detection.results[0].id].append(mask_msg)

        overlay = cv_img.copy()
        active_detections = []

        for detection in yolo_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            if detection_id not in self.id_tf_map:
                continue

            prop_name = self.id_tf_map[detection_id]

            # Process only nutbolt, electric, bandaid, and pill links
            if prop_name not in (
                "nutbolt_link",
                "electric_link",
                "bandaid_link",
                "pill_link",
            ):
                continue

            mask_msg = (
                masks_by_id[detection_id].popleft()
                if masks_by_id.get(detection_id)
                else None
            )
            if mask_msg is None:
                continue

            mask = self._mask_to_cv2(mask_msg)
            if mask is None:
                continue

            last_yaw = self.last_yaws.get(detection_id)
            geometry = None

            if prop_name in ("electric_link", "bandaid_link"):
                geometry = findposes_rect(mask, last_yaw=last_yaw, debug=True)
                if geometry is not None:
                    geometry["type"] = "object_rect"
            elif prop_name in ("nutbolt_link", "pill_link"):
                geometry = findposes_circle(mask, last_yaw=last_yaw, debug=True)
                if geometry is not None:
                    geometry["type"] = "object_circle"

            if geometry is not None and not geometry.get("valid", False):
                geometry = None

            if geometry is not None:
                if geometry.get("yaw") is not None:
                    self.last_yaws[detection_id] = geometry["yaw"]

                color = self.COLOR_PALETTE.get(prop_name, (255, 255, 255))

                binary = (mask > 127).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                size_str = ""
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    if geometry["type"] in ("object_rect", "basket"):
                        if len(contour) >= 4:
                            rect = cv2.minAreaRect(contour)
                            box = cv2.boxPoints(rect).astype(np.int32)
                            cv2.fillPoly(overlay, [box], color)
                            geometry["box_points"] = box

                            edges = geometry.get("edges_px")
                            if edges and len(edges) >= 2:
                                size_str = f"size: {edges[0]:.1f}x{edges[1]:.1f} px"
                    elif geometry["type"] == "object_circle":
                        (cx, cy), radius_px = cv2.minEnclosingCircle(contour)
                        cv2.circle(
                            overlay,
                            (int(round(cx)), int(round(cy))),
                            int(round(radius_px)),
                            color,
                            -1,
                        )
                        geometry["circle_data"] = (
                            int(round(cx)),
                            int(round(cy)),
                            int(round(radius_px)),
                        )
                        diam = geometry.get("diameter_px")
                        if diam is not None:
                            size_str = f"size: d={diam:.1f} px"

                if not size_str and hasattr(detection, "bbox"):
                    size_str = f"bbox: {detection.bbox.size_x:.1f}x{detection.bbox.size_y:.1f} px"

                # Lookup TF of the frame relative to odom
                pos = None
                rot = None
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "odom", prop_name, img_msg.header.stamp, rospy.Duration(0.05)
                    )
                    pos = transform.transform.translation
                    rot = transform.transform.rotation
                except Exception:
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            "odom", prop_name, rospy.Time(0)
                        )
                        pos = transform.transform.translation
                        rot = transform.transform.rotation
                    except Exception:
                        pass

                active_detections.append(
                    {
                        "prop_name": prop_name,
                        "geometry": geometry,
                        "color": color,
                        "pos": pos,
                        "rot": rot,
                        "size_str": size_str,
                    }
                )

        # Blend original image with overlay containing filled geometries with 0.5 alpha
        cv_img = cv2.addWeighted(cv_img, 0.5, overlay, 0.5, 0.0)

        # Draw solid, crisp outlines and centers on top
        for item in active_detections:
            geom = item["geometry"]
            color = item["color"]
            if geom["type"] in ("object_rect", "basket") and "box_points" in geom:
                cv2.polylines(cv_img, [geom["box_points"]], True, color, 2)
                center = geom["center"]
                cv2.circle(
                    cv_img,
                    (int(round(center[0])), int(round(center[1]))),
                    5,
                    (255, 255, 255),
                    -1,
                )
            elif geom["type"] == "object_circle" and "circle_data" in geom:
                cx, cy, radius = geom["circle_data"]
                cv2.circle(cv_img, (cx, cy), radius, color, 2)
                cv2.circle(cv_img, (cx, cy), 5, (255, 255, 255), -1)

        # Render glassmorphic side panel on the left (semi-transparent dark background)
        sidebar_width = 380
        sidebar_overlay = cv_img.copy()
        cv2.rectangle(
            sidebar_overlay, (0, 0), (sidebar_width, cv_img.shape[0]), (15, 15, 15), -1
        )
        cv_img = cv2.addWeighted(cv_img, 0.4, sidebar_overlay, 0.6, 0.0)
        cv2.line(
            cv_img,
            (sidebar_width, 0),
            (sidebar_width, cv_img.shape[0]),
            (60, 60, 60),
            1,
        )

        # Title (Size multiplied by 2: 0.5 -> 1.0)
        cv2.putText(
            cv_img,
            "OCTAGON DEBUGGER",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(cv_img, (20, 48), (sidebar_width - 20, 48), (80, 80, 80), 2)

        # Display details for currently detected objects
        y_offset = 90
        for item in active_detections:
            prop_name = item["prop_name"]
            color = item["color"]
            pos = item["pos"]
            rot = item["rot"]
            geom = item["geometry"]
            size_str = item.get("size_str", "")

            # Color block indicator
            cv2.rectangle(cv_img, (20, y_offset - 20), (40, y_offset), color, -1)
            cv2.rectangle(
                cv_img, (20, y_offset - 20), (40, y_offset), (255, 255, 255), 1
            )

            # Nice readable name (Size multiplied by 2: 0.45 -> 0.9)
            clean_name = prop_name.replace("_link", "").replace("_", " ").title()
            cv2.putText(
                cv_img,
                clean_name,
                (50, y_offset - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y_offset += 28

            # Position coords (Size multiplied by 2: 0.38 -> 0.76)
            if pos is not None:
                pos_str = f"x:{pos.x:.2f} y:{pos.y:.2f} z:{pos.z:.2f}"
            else:
                pos_str = "x:N/A y:N/A z:N/A"
            cv2.putText(
                cv_img,
                pos_str,
                (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.76,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            y_offset += 28

            # Pixel width & height size info
            if size_str:
                cv2.putText(
                    cv_img,
                    size_str,
                    (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.76,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += 28

            # Yaw (only for non-circular objects)
            if prop_name not in ("nutbolt_link", "pill_link"):
                if rot is not None:
                    _, _, yaw_val = tf_transformations.euler_from_quaternion(
                        [rot.x, rot.y, rot.z, rot.w]
                    )
                    yaw_deg = math.degrees(yaw_val)
                    yaw_str = f"yaw: {yaw_deg:+.1f} deg"
                elif geom.get("yaw") is not None:
                    yaw_str = f"yaw: {math.degrees(geom['yaw']):+.1f} deg"
                else:
                    yaw_str = "yaw: N/A"

                cv2.putText(
                    cv_img,
                    yaw_str,
                    (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.76,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += 28

            y_offset += 40

        # Publish the finalized debug image
        try:
            out_msg = CompressedImage()
            out_msg.header = img_msg.header
            out_msg.format = "jpeg"
            out_msg.data = np.array(
                cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
            ).tobytes()
            self.pub_debug.publish(out_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish debug compressed image: {e}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = SegmentDebugImageNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
