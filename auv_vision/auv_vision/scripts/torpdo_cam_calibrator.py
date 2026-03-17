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
THRESHOLD_VALUE = 160  # Binary threshold for bright-pixel detection
MIN_AREA = 50  # Minimum pixel area for a valid connected component
MANUAL_MODE = True  # If True, user selects center via mouse click.
ACTUAL_HORIZONTAL_DISTANCE = 0.00285  # (meters) Known physical horizontal distance to the target

# --- Refraction settings (Flat Port) ---
APPLY_REFRACTION_CORRECTION = True  # Set to True if calibrating in air for underwater use
REFRACTIVE_INDEX_AIR = 1.0003
REFRACTIVE_INDEX_WATER = 1.333  # Approximation for fresh/pool water


class PitchEstimator:
    def __init__(self):
        # Calibration (filled once from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calib_received = False
        self.manual_center_uv = None

        self.bridge = CvBridge()

        if MANUAL_MODE:
            cv2.namedWindow("Manual Calibration", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Manual Calibration", self.mouse_callback)

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

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.manual_center_uv = (x, y)
            rospy.loginfo("Manual center set to: (%.1f, %.1f)", x, y)

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
        if MANUAL_MODE:
            debug_image = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            if self.manual_center_uv is None:
                cv2.putText(debug_image, "Click to select center", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Manual Calibration", debug_image)
                cv2.waitKey(1)
                return
                
            center_uv = self.manual_center_uv
            cx, cy = center_uv
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
        else:
            try:
                center_uv, mask, debug_image, bbox = self.detect_white_opening_center(image)
            except RuntimeError as e:
                rospy.logwarn_throttle(5.0, str(e))
                return

        # --- Compute pitch ---
        result = self.compute_downward_pitch(center_uv)

        # In manual mode, show the distance early as well
        if MANUAL_MODE:
            if result["y_normalized"] != 0:
                vertical_offset = ACTUAL_HORIZONTAL_DISTANCE * math.tan(abs(result["pitch_rad_water"]))
                cv2.putText(
                    debug_image,
                    "Z-Offset at {:.5f}m: {:.5f}m".format(ACTUAL_HORIZONTAL_DISTANCE, vertical_offset),
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

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

        # Draw pitch and distance text on debug image
        cv2.putText(
            debug_image,
            "Air Pitch: {:.2f} deg ({:.4f} rad)".format(result["pitch_deg_air"], result["pitch_rad_air"]),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        
        cv2.putText(
            debug_image,
            "Water Pitch: {:.2f} deg ({:.4f} rad)".format(result["pitch_deg_water"], result["pitch_rad_water"]),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255) if APPLY_REFRACTION_CORRECTION else (0, 255, 255),
            2,
        )

        
        if result["y_normalized"] != 0:
            vertical_offset = ACTUAL_HORIZONTAL_DISTANCE * math.tan(abs(result["pitch_rad_water"]))
            
            cv2.putText(
                debug_image,
                "Z-Offset at {:.5f}m: {:.5f}m".format(ACTUAL_HORIZONTAL_DISTANCE, vertical_offset),
                (10, 90 if not MANUAL_MODE else 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            
            
            rospy.loginfo(
                "center=(%.1f, %.1f) air_pitch=%.3f deg (%.4f rad) water_pitch=%.3f deg (%.4f rad) Z-offset=%.5fm (dist=%.1fpx)",
                center_uv[0],
                center_uv[1],
                result["pitch_deg_air"],
                result["pitch_rad_air"],
                result["pitch_deg_water"],
                result["pitch_rad_water"],
                vertical_offset,
                result["euclidean_pixel_dist"]
            )
        else:
            rospy.loginfo(
                "center=(%.1f, %.1f) air_pitch=%.3f deg (%.4f rad) water_pitch=%.3f deg (%.4f rad) (dist=%.1fpx)",
                center_uv[0],
                center_uv[1],
                result["pitch_deg_air"],
                result["pitch_rad_air"],
                result["pitch_deg_water"],
                result["pitch_rad_water"],
                result["euclidean_pixel_dist"]
            )


        # --- Publish ---
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8"))
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8"))

        if MANUAL_MODE:
            cv2.imshow("Manual Calibration", debug_image)
            cv2.waitKey(1)

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
            
        Using pixel distances (geometric interpretation):
            Pitch angle is derived from the pinhole camera model.
            y_pixel_dist = (y_pixel - cy)
            pitch = atan(y_pixel_dist / fy)
        """
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]

        # 1) Standard undistortion approach for normalized coords
        pts = np.array([[[center_uv[0], center_uv[1]]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs)
        x_norm = float(undistorted[0, 0, 0])
        y_norm = float(undistorted[0, 0, 1])

        # 2) Pixel distance method (geometric interpretation)
        y_pixel_dist = center_uv[1] - cy
        x_pixel_dist = center_uv[0] - cx
        
        # Calculate pitch in air (what the camera sees inside its housing)
        # pitch_air = arctan(pixel_distance / focal_length)
        pitch_rad_air = math.atan(y_pixel_dist / fy)
        pitch_deg_air = math.degrees(pitch_rad_air)

        # 3) Flat Port Refraction Correction
        # The light ray bends as it travels from Water (n1) through the flat glass/acrylic port into Air (n2).
        # We assume the camera looks perfectly perpendicular out of a flat port.
        # Snell's Law: n1 * sin(theta1) = n2 * sin(theta2)
        # Therefore, theta1 (water pitch) = arcsin( (n2 / n1) * sin(theta2 (air pitch)) )
        
        if APPLY_REFRACTION_CORRECTION:
            # We must handle the sign independently, as arcsin might have negative values depending on Python math functions.
            sign = 1 if pitch_rad_air >= 0 else -1
            sin_theta_air = math.sin(abs(pitch_rad_air))
            
            # Prevent domain error if calculation goes out of bounds (shouldn't realistically happen unless extreme FOV)
            ratio = (REFRACTIVE_INDEX_AIR / REFRACTIVE_INDEX_WATER) * sin_theta_air
            if ratio > 1.0: ratio = 1.0
            
            pitch_rad_water = sign * math.asin(ratio)
        else:
            pitch_rad_water = pitch_rad_air

        pitch_deg_water = math.degrees(pitch_rad_water)

        return {
            "x_normalized": x_norm,
            "y_normalized": y_norm,
            "pitch_rad_air": pitch_rad_air,
            "pitch_deg_air": pitch_deg_air,
            "pitch_rad_water": pitch_rad_water,
            "pitch_deg_water": pitch_deg_water,
            "y_pixel_dist": y_pixel_dist,
            "x_pixel_dist": x_pixel_dist,
            "euclidean_pixel_dist": math.hypot(x_pixel_dist, y_pixel_dist),
            "focal_length_y": fy,
            "default_pitch_deg": math.degrees(math.atan(y_norm)) # Just keeping the old one for reference
        }


def main():
    rospy.init_node("torpedo_pitch_estimator", anonymous=False)
    estimator = PitchEstimator()
    rospy.spin()


if __name__ == "__main__":
    main()
