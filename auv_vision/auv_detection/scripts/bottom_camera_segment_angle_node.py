#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bottle Angle and Thickness Detector (ROS1)

- Subscribes: sensor_msgs/Image (mono8 bottle mask)
- Publishes:
    - std_msgs/Float32 (deviation angle in radians)
    - std_msgs/Float32 (median thickness in pixels)
    - std_msgs/Float32MultiArray (detailed debug info)
    - sensor_msgs/Image (annotated debug image)

"""

import math
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge

# --- YENİ İMPORTLAR ---
# Kalınlık hesaplaması için gerekli kütüphaneler
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# --- BİTTİ ---


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


class BottomCameraSegmentAngleNode(object):
    def __init__(self):
        # ---- Topics (relative - will be prefixed by namespace) ----
        self.mask_topic = rospy.get_param("~mask_topic", "bottle_mask")
        self.angle_topic = rospy.get_param("~angle_topic", "bottle_angle")
        # --- YENİ TOPIC ---
        self.thickness_topic = rospy.get_param("~thickness_topic", "bottle_thickness")
        # --- BİTTİ ---
        self.debug_topic = rospy.get_param("~debug_topic", "bottle_angle_debug")

        # ---- Image orientation controls ----
        self.rotate_cw = rospy.get_param("~rotate_cw", 0)
        self.flip_x = rospy.get_param("~flip_x", False)
        self.flip_y = rospy.get_param("~flip_y", False)

        # ---- Validity ----
        self.min_bottle_area_px = int(rospy.get_param("~min_bottle_area_px", 1500))
        self.min_contour_points = int(rospy.get_param("~min_contour_points", 10))

        # ---- Debug ----
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        self.bridge = CvBridge()
        self.pub_angle = rospy.Publisher(self.angle_topic, Float32, queue_size=1)
        # --- YENİ PUBLISHER ---
        self.pub_thickness = rospy.Publisher(
            self.thickness_topic, Float32, queue_size=1
        )
        # --- BİTTİ ---
        self.pub_debug_info = rospy.Publisher(
            self.debug_topic, Float32MultiArray, queue_size=1
        )
        self.pub_debug_img = rospy.Publisher(
            self.debug_topic + "_img/compressed", CompressedImage, queue_size=1
        )

        self.sub_mask = rospy.Subscriber(
            self.mask_topic, Image, self.cb_mask, queue_size=1, buff_size=2**24
        )

        rospy.loginfo(
            "[bottom_camera_segment_angle] started. Subscribing to %s. Publishing angle to %s and thickness to %s",
            self.mask_topic,
            self.angle_topic,
            self.thickness_topic,
        )

    def cb_mask(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn("cv_bridge err: %s", e)
            return

        # Bring into assumed orientation
        if self.rotate_cw:
            mask = rot90n(mask, self.rotate_cw)
        if self.flip_x:
            mask = np.ascontiguousarray(np.fliplr(mask))
        if self.flip_y:
            mask = np.ascontiguousarray(np.flipud(mask))

        h, w = mask.shape[:2]

        # ---------- Validity check ----------
        bottle_area = int(np.count_nonzero(mask))
        
        if bottle_area < self.min_bottle_area_px:
            rospy.logdebug_throttle(
                2.0,
                "[bottom_camera_segment_angle] area too small: %d < %d",
                bottle_area,
                self.min_bottle_area_px,
            )
            self.pub_angle.publish(Float32(float("nan")))
            self.pub_thickness.publish(Float32(float("nan")))  # Yeni
            if self.publish_debug:
                self._publish_debug_info(
                    angle=float("nan"),
                    confidence=0.0,
                    bottle_area=bottle_area,
                    thickness=float("nan"),
                )
            return

        # ---------- Extract contours FIRST (for bbox ROI) ----------
        _ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(_ret) == 3:
            _img_out, contours, _hier = _ret
        else:
            contours, _hier = _ret

        contours = list(contours) if contours is not None else []
        if len(contours) == 0:
            rospy.logdebug_throttle(
                2.0, "[bottom_camera_segment_angle] no contours found"
            )
            self.pub_angle.publish(Float32(float("nan")))
            self.pub_thickness.publish(Float32(float("nan")))
            if self.publish_debug:
                self._publish_debug_info(
                    angle=float("nan"),
                    confidence=0.0,
                    bottle_area=bottle_area,
                    thickness=float("nan"),
                )
            return

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        pts = largest_contour.reshape(-1, 2)

        if len(pts) < self.min_contour_points:
            rospy.logdebug_throttle(
                2.0,
                "[bottom_camera_segment_angle] contour too small: %d points",
                len(pts),
            )
            self.pub_angle.publish(Float32(float("nan")))
            self.pub_thickness.publish(Float32(float("nan")))
            if self.publish_debug:
                self._publish_debug_info(
                    angle=float("nan"),
                    confidence=0.0,
                    bottle_area=bottle_area,
                    thickness=float("nan"),
                )
            return

        # Binarize mask for calculations
        binary_mask = mask > 127

        # ===================================================================
        # ----------  (DT + Skeleton) ----------
        # ===================================================================
        # Get bounding box from contour for ROI extraction
        x, y, w, roi_h = cv2.boundingRect(largest_contour)
        
        # Add padding for safety
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(binary_mask.shape[1], x + w + pad), min(binary_mask.shape[0], y + roi_h + pad)
        
        # Make contiguous for skeletonize compatibility
        roi_mask = np.ascontiguousarray(binary_mask[y1:y2, x1:x2])
        
        median_thickness_px = 0.0
        try:
            # Distance Transform + Skeleton on ROI only
            dist = distance_transform_edt(roi_mask)
            skel = skeletonize(roi_mask)

            # Calculate thickness from skeleton
            if np.any(skel):
                thickness_values = 2.0 * dist[skel]
                if thickness_values.size > 0:
                    median_thickness_px = float(np.median(thickness_values))
        except Exception as e:
            rospy.logwarn_throttle(
                5.0, "[bottom_camera_segment_angle] Thickness calculation failed: %s", e
            )
            median_thickness_px = float("nan")
        # ===================================================================

        # ---------- Fit line to contour ----------
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx), float(vy)

        # ... (Koordinat dönüşümü ve açı hesaplama kodu aynı kalıyor)
        # Camera is rotated 180 degrees, so "Front" is now Right (positive X axis in image)
        # We calculate the angle of the line relative to this Front vector.
        # atan2(-vy, vx) gives the angle relative to the positive X axis with Y-up convention (CCW positive).
        # We negate vy because image Y-axis is down, but we want standard math Y-up behavior.
        angle = math.atan2(-vy, vx)

        # Normalize to [-pi/2, pi/2] to handle line ambiguity (line has no direction)
        # This effectively forces the vector to point roughly towards the "Front" (Right)
        if angle > math.pi / 2:
            angle -= math.pi
        elif angle <= -math.pi / 2:
            angle += math.pi

        vehicle_vx = math.cos(angle)
        vehicle_vy = math.sin(angle)

        line_magnitude = math.sqrt(vx * vx + vy * vy)
        confidence = min(1.0, line_magnitude)

        # ---------- Publish results ----------
        self.pub_angle.publish(Float32(angle))
        self.pub_thickness.publish(Float32(median_thickness_px))

        if self.publish_debug:
            self._publish_debug_info(
                angle=angle,
                confidence=confidence,
                bottle_area=bottle_area,
                thickness=median_thickness_px,  # Yeni
                vx=vx,
                vy=vy,
                vehicle_vx=vehicle_vx,
                vehicle_vy=vehicle_vy,
            )
            self._publish_debug_image(
                mask, (vx, vy, x0, y0), angle, median_thickness_px
            )  # Yeni

    def _publish_debug_info(
        self,
        angle,
        confidence,
        bottle_area,
        thickness,  # Yeni parametre
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
            float(bottle_area),
            thickness if not math.isnan(thickness) else -1.0,  # Yeni veri
            vx if vx is not None else -999.0,
            vy if vy is not None else -999.0,
            vehicle_vx if vehicle_vx is not None else -999.0,
            vehicle_vy if vehicle_vy is not None else -999.0,
        ]
        self.pub_debug_info.publish(msg)

    def _publish_debug_image(self, mask, fit, angle, thickness):  # Yeni parametre
        """Publish annotated debug image."""
        h, w = mask.shape[:2]
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if fit is not None:
            # ... (mevcut çizgi ve merkez çizimi aynı)
            vx, vy, x0, y0 = fit
            x0, y0, vx, vy = float(x0), float(y0), float(vx), float(vy)
            p1 = (int(x0 - 2000 * vx), int(y0 - 2000 * vy))
            p2 = (int(x0 + 2000 * vx), int(y0 + 2000 * vy))
            cv2.line(vis, p1, p2, (0, 0, 255), 2)
            cv2.circle(vis, (int(x0), int(y0)), 5, (0, 255, 0), -1)

        # ... (mevcut yön oku çizimi aynı)
        # Draw Reference Arrow (Vehicle Front)
        # Camera is rotated 180 degrees, so Front is Right.
        cv2.arrowedLine(
            vis, (w // 2, h // 2), (w * 3 // 4, h // 2), (255, 0, 0), 2, tipLength=0.3
        )

        # Draw text (Açı ve Kalınlık)
        angle_deg = math.degrees(angle) if not math.isnan(angle) else 0.0
        txt_angle = f"Angle: {angle_deg:+.1f} deg"
        txt_thickness = f"Thickness: {thickness:.1f} px"  # Yeni

        cv2.putText(
            vis,
            txt_angle,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            txt_thickness,
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )  # Yeni

        try:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).tobytes()
            self.pub_debug_img.publish(msg)
        except Exception as e:
            rospy.logwarn("Failed to publish debug image: %s", e)


def main():
    rospy.init_node("bottom_camera_segment_angle")
    BottomCameraSegmentAngleNode()
    rospy.spin()


if __name__ == "__main__":
    main()
