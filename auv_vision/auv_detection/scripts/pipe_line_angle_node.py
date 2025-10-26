#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipe Line Angle Detector (ROS1)

- Subscribes: sensor_msgs/Image (mono8 pipe mask, white=pipe, black=background)
- Publishes: std_msgs/Float32 (deviation angle in radians from forward direction)
            std_msgs/Float32MultiArray (detailed info: angle, confidence, etc.)

Camera Coordinate System (as per pipe_followe_node.py):
- Image X (columns): left→right
- Image Y (rows): top→bottom
- After rotation/flip transforms, assumed frame:
  - Left (-X in image) = Forward direction (+X in vehicle frame)
  - Up (-Y in image) = Right direction (+Y in vehicle frame)

Deviation Angle:
- 0 rad: pipe aligned with forward direction
- +angle: pipe tilted toward right
- -angle: pipe tilted toward left

Author: Cascade AI
"""

import math
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge


def rot90n(img, n_cw):
    """Rotate image by n*90 degrees clockwise."""
    n_cw = int(n_cw) % 4
    if n_cw == 0:
        return img
    elif n_cw == 1:
        return np.ascontiguousarray(np.rot90(img, k=3))
    elif n_cw == 2:
        return np.ascontiguousarray(np.rot90(img, k=2))
    else:  # 3
        return np.ascontiguousarray(np.rot90(img, k=1))


class PipeLineAngleNode(object):
    def __init__(self):
        # ---- Topics ----
        self.mask_topic = rospy.get_param("~mask_topic", "pipe_mask")
        self.angle_topic = rospy.get_param("~angle_topic", "pipe_line_angle")
        self.debug_topic = rospy.get_param("~debug_topic", "pipe_line_angle_debug")

        # ---- Image orientation controls ----
        # Same as pipe_followe_node.py
        self.rotate_cw = rospy.get_param("~rotate_cw", 0)  # 0,1,2,3 (×90° clockwise)
        self.flip_x = rospy.get_param("~flip_x", False)  # horizontal flip
        self.flip_y = rospy.get_param("~flip_y", False)  # vertical flip

        # ---- Validity ----
        self.min_pipe_area_px = int(rospy.get_param("~min_pipe_area_px", 1500))
        self.min_contour_points = int(rospy.get_param("~min_contour_points", 10))

        # ---- Debug ----
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        self.bridge = CvBridge()
        self.pub_angle = rospy.Publisher(self.angle_topic, Float32, queue_size=1)
        self.pub_debug_info = rospy.Publisher(
            self.debug_topic, Float32MultiArray, queue_size=1
        )
        self.pub_debug_img = rospy.Publisher(
            self.debug_topic + "_img", Image, queue_size=1
        )

        self.sub_mask = rospy.Subscriber(
            self.mask_topic, Image, self.cb_mask, queue_size=1, buff_size=2**24
        )

        rospy.loginfo(
            "[pipe_line_angle] started. Subscribing to %s, publishing %s",
            self.mask_topic,
            self.angle_topic,
        )

    def cb_mask(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn("cv_bridge err: %s", e)
            return

        # Bring into assumed orientation (same as pipe_followe_node.py)
        if self.rotate_cw:
            mask = rot90n(mask, self.rotate_cw)
        if self.flip_x:
            mask = np.ascontiguousarray(np.fliplr(mask))
        if self.flip_y:
            mask = np.ascontiguousarray(np.flipud(mask))

        h, w = mask.shape[:2]

        # ---------- Validity check ----------
        pipe_area = int(np.count_nonzero(mask))
        if pipe_area < self.min_pipe_area_px:
            rospy.logdebug_throttle(
                2.0,
                "[pipe_line_angle] pipe area too small: %d < %d",
                pipe_area,
                self.min_pipe_area_px,
            )
            # Publish invalid angle (NaN or large value)
            self.pub_angle.publish(Float32(float("nan")))
            if self.publish_debug:
                self._publish_debug_info(
                    angle=float("nan"), confidence=0.0, pipe_area=pipe_area
                )
            return

        # ---------- Extract contours and fit line ----------
        _ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(_ret) == 3:
            _img_out, contours, _hier = _ret
        else:
            contours, _hier = _ret

        contours = list(contours) if contours is not None else []
        if len(contours) == 0:
            rospy.logdebug_throttle(2.0, "[pipe_line_angle] no contours found")
            self.pub_angle.publish(Float32(float("nan")))
            if self.publish_debug:
                self._publish_debug_info(
                    angle=float("nan"), confidence=0.0, pipe_area=pipe_area
                )
            return

        # Use largest contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        pts = contours[0].reshape(-1, 2)

        if len(pts) < self.min_contour_points:
            rospy.logdebug_throttle(
                2.0,
                "[pipe_line_angle] contour too small: %d points",
                len(pts),
            )
            self.pub_angle.publish(Float32(float("nan")))
            if self.publish_debug:
                self._publish_debug_info(
                    angle=float("nan"), confidence=0.0, pipe_area=pipe_area
                )
            return

        # Fit line to contour
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx), float(vy)

        # ---------- Coordinate transformation ----------
        # Camera frame after rotation/flip:
        # - X axis points left (toward -X in original image) = forward direction
        # - Y axis points up (toward -Y in original image) = right direction
        #
        # The fitted line direction (vx, vy) is in image coordinates:
        # - vx: positive = right in image
        # - vy: positive = down in image
        #
        # We need to transform to vehicle frame:
        # - vehicle_x (forward) corresponds to -image_x (left)
        # - vehicle_y (right) corresponds to -image_y (up)
        #
        # So: vehicle_vx = -vx, vehicle_vy = -vy

        vehicle_vx = -vx  # forward direction in vehicle frame
        vehicle_vy = -vy  # right direction in vehicle frame

        # Compute deviation angle from forward direction
        # Forward direction is (1, 0) in vehicle frame
        # Angle = atan2(vehicle_vy, vehicle_vx)
        # Positive angle = tilted toward right
        # Negative angle = tilted toward left

        angle = math.atan2(vehicle_vy, vehicle_vx)

        # Normalize to [-pi/2, +pi/2] to avoid flipping
        if angle > math.pi / 2:
            angle -= math.pi
        elif angle < -math.pi / 2:
            angle += math.pi

        # Confidence metric: how well-aligned is the line
        # Use the magnitude of the direction vector (should be ~1 after normalization)
        line_magnitude = math.sqrt(vx * vx + vy * vy)
        confidence = min(1.0, line_magnitude)

        # Publish angle
        self.pub_angle.publish(Float32(angle))

        if self.publish_debug:
            self._publish_debug_info(
                angle=angle,
                confidence=confidence,
                pipe_area=pipe_area,
                vx=vx,
                vy=vy,
                vehicle_vx=vehicle_vx,
                vehicle_vy=vehicle_vy,
            )
            self._publish_debug_image(mask, (vx, vy, x0, y0), angle)

    def _publish_debug_info(
        self,
        angle,
        confidence,
        pipe_area,
        vx=None,
        vy=None,
        vehicle_vx=None,
        vehicle_vy=None,
    ):
        """Publish detailed debug information."""
        msg = Float32MultiArray()
        msg.data = [
            angle if not math.isnan(angle) else -999.0,
            confidence,
            float(pipe_area),
            vx if vx is not None else -999.0,
            vy if vy is not None else -999.0,
            vehicle_vx if vehicle_vx is not None else -999.0,
            vehicle_vy if vehicle_vy is not None else -999.0,
        ]
        self.pub_debug_info.publish(msg)

    def _publish_debug_image(self, mask, fit, angle):
        """Publish annotated debug image."""
        h, w = mask.shape[:2]
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if fit is not None:
            vx, vy, x0, y0 = fit
            x0, y0, vx, vy = float(x0), float(y0), float(vx), float(vy)

            # Draw fitted line
            def line_point(t):
                return int(x0 + t * vx), int(y0 + t * vy)

            p1 = line_point(-2000)
            p2 = line_point(+2000)
            cv2.line(vis, p1, p2, (0, 0, 255), 2)

            # Draw center point
            cv2.circle(vis, (int(x0), int(y0)), 5, (0, 255, 0), -1)

        # Draw forward direction indicator (left side of image = forward)
        cv2.arrowedLine(
            vis,
            (w // 2, h // 2),
            (w // 4, h // 2),
            (255, 0, 0),
            2,
            tipLength=0.3,
        )

        # Draw text
        angle_deg = math.degrees(angle) if not math.isnan(angle) else 0.0
        txt = f"Angle: {angle_deg:+.1f} deg ({angle:+.3f} rad)"
        cv2.putText(
            vis,
            txt,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        try:
            msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            self.pub_debug_img.publish(msg)
        except Exception as e:
            rospy.logwarn("Failed to publish debug image: %s", e)


def main():
    rospy.init_node("pipe_line_angle")
    PipeLineAngleNode()
    rospy.spin()


if __name__ == "__main__":
    main()
