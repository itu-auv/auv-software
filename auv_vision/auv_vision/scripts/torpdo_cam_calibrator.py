#!/usr/bin/env python3
"""
torpdo_cam_calibrator.py

Continuously estimates the downward pitch angle of the torpedo camera
by detecting a bright white plus-shaped ("+") opening and computing
the angle using camera intrinsics and distortion coefficients.

Subscribes to:
  - /taluy/cameras/cam_torpedo/camera_info
  - /taluy/cameras/cam_torpedo/image_raw

Publishes:
  - ~debug_image   (sensor_msgs/Image, BGR8)
  - ~binary_mask   (sensor_msgs/Image, mono8)
"""

import cv2
import numpy as np
import math
import json
import os
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

IMAGE_TOPIC = "/taluy/cameras/cam_torpedo/image_raw"
CAMERA_INFO_TOPIC = "/taluy/cameras/cam_torpedo/camera_info"
OUTPUT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/pitch_output"
THRESHOLD_VALUE = 200  # Binary threshold for bright-pixel detection
MIN_AREA = 100  # Minimum pixel area for a valid connected component


class PitchEstimator:
    def __init__(self):
        # Calibration (filled once from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calib_received = False

        self.bridge = CvBridge()

        # Publishers (latched so last message is always available)
        self.debug_pub = rospy.Publisher(
            "~debug_image", Image, queue_size=1, latch=True
        )
        self.mask_pub = rospy.Publisher("~binary_mask", Image, queue_size=1, latch=True)

        # Output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Subscribers
        rospy.Subscriber(
            CAMERA_INFO_TOPIC, CameraInfo, self.camera_info_cb, queue_size=1
        )
        rospy.Subscriber(
            IMAGE_TOPIC, Image, self.image_cb, queue_size=1, buff_size=2**24
        )

        rospy.loginfo("PitchEstimator started. Waiting for camera_info and images...")

    # ----------------------------------------------------------
    def camera_info_cb(self, msg):
        """Receive calibration once (or update if it changes)."""
        self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D, dtype=np.float64)
        if not self.calib_received:
            rospy.loginfo("Camera matrix:\n%s", self.camera_matrix)
            rospy.loginfo("Distortion coefficients: %s", self.dist_coeffs)
            self.calib_received = True

    # ----------------------------------------------------------
    def image_cb(self, msg):
        """Process each incoming image."""
        if not self.calib_received:
            rospy.logwarn_throttle(5.0, "No camera_info yet — skipping frame.")
            return

        # Convert ROS Image → BGR
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        # --- Detect white opening ---
        try:
            center_uv, mask, debug_image, bbox = self.detect_white_opening_center(image)
        except RuntimeError as e:
            rospy.logwarn_throttle(5.0, str(e))
            return

        # --- Compute pitch ---
        result = self.compute_downward_pitch(center_uv)

        cx_principal = self.camera_matrix[0, 2]
        cy_principal = self.camera_matrix[1, 2]

        # Draw principal point (blue)
        cv2.circle(
            debug_image, (int(cx_principal), int(cy_principal)), 8, (255, 0, 0), -1
        )
        cv2.putText(
            debug_image,
            "Principal ({:.1f},{:.1f})".format(cx_principal, cy_principal),
            (int(cx_principal) + 12, int(cy_principal) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

        # Draw pitch text on debug image
        cv2.putText(
            debug_image,
            "Pitch: {:.2f} deg ({:.4f} rad)".format(
                result["pitch_deg"], result["pitch_rad"]
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # --- Publish ---
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8"))
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8"))

        # --- Log ---
        rospy.loginfo(
            "center=(%.1f, %.1f)  pitch=%.3f deg (%.4f rad)  (norm_y=%.5f)",
            center_uv[0],
            center_uv[1],
            result["pitch_deg"],
            result["pitch_rad"],
            result["y_normalized"],
        )

    # ----------------------------------------------------------
    def detect_white_opening_center(self, image):
        """
        Detect the center of the bright white plus-shaped opening.

        Pipeline:
            1. BGR → grayscale
            2. Gaussian blur (5×5)
            3. Binary threshold
            4. Morphological open + close
            5. connectedComponentsWithStats
            6. Largest bright component above MIN_AREA
            7. Build debug image with contours, bbox, center dot

        Returns: (center_uv, mask, debug_image, bbox)
        Raises RuntimeError if nothing detected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )

        best_label = -1
        best_area = 0
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= MIN_AREA and area > best_area:
                best_area = area
                best_label = label_id

        if best_label == -1:
            raise RuntimeError(
                "No bright component found (min_area={}).".format(MIN_AREA)
            )

        cx = centroids[best_label][0]
        cy = centroids[best_label][1]
        center_uv = (cx, cy)

        x = stats[best_label, cv2.CC_STAT_LEFT]
        y = stats[best_label, cv2.CC_STAT_TOP]
        w = stats[best_label, cv2.CC_STAT_WIDTH]
        h = stats[best_label, cv2.CC_STAT_HEIGHT]
        bbox = (x, y, w, h)

        # --- Debug image ---
        debug_image = image.copy()

        # + shape outline (cyan-yellow)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bx < x + w and bx + bw > x and by < y + h and by + bh > y:
                cv2.drawContours(debug_image, [cnt], -1, (255, 255, 0), 2)

        # Bounding box (green)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detected center (red)
        cv2.circle(debug_image, (int(cx), int(cy)), 8, (0, 0, 255), -1)
        cv2.putText(
            debug_image,
            "Center ({:.1f},{:.1f})".format(cx, cy),
            (int(cx) + 12, int(cy) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        return center_uv, cleaned, debug_image, bbox

    # ----------------------------------------------------------
    def compute_downward_pitch(self, center_uv):
        """
        Compute downward pitch from a pixel using camera intrinsics.

        Convention:
            - Positive pitch = camera tilted DOWNWARD
            - OpenCV image y increases downward
            - pitch = atan(y_normalized)
        """
        pts = np.array([[[center_uv[0], center_uv[1]]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs)
        x_norm = float(undistorted[0, 0, 0])
        y_norm = float(undistorted[0, 0, 1])

        pitch_rad = math.atan(y_norm)
        pitch_deg = math.degrees(pitch_rad)

        return {
            "x_normalized": x_norm,
            "y_normalized": y_norm,
            "pitch_rad": pitch_rad,
            "pitch_deg": pitch_deg,
        }


def main():
    rospy.init_node("torpedo_pitch_estimator", anonymous=False)
    estimator = PitchEstimator()
    rospy.spin()


if __name__ == "__main__":
    main()
