#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class PipeDetector:
    def __init__(self):
        rospy.init_node("pipe_detector", anonymous=True)

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Subscriber
        self.image_sub = rospy.Subscriber(
            "/taluy/cameras/cam_front/image_raw",
            Image,
            self.image_callback,
            queue_size=1,
        )

        # Publishers
        self.image_pub = rospy.Publisher(
            "/pipe_detector/result_image", Image, queue_size=1
        )
        self.detection_pub = rospy.Publisher(
            "/pipe_detector/detections", Image, queue_size=1
        )

        # Color ranges for white and red detection in HSV
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])

        # Minimum contour area to filter noise
        self.min_contour_area = 500

        rospy.loginfo("Pipe Detector initialized. Subscribing to camera/image_raw")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        # Convert to HSV for color masking
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Create masks for white and red
        white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        r1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        r2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(r1, r2)
        combined_mask = cv2.bitwise_or(white_mask, red_mask)

        # Morphological clean-up
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result_image = cv_image.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            # 1) Aspect ratio filter: height > width
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / (w + 1e-6)
            if aspect_ratio < 2.0:
                continue

            # # 2) Orientation filter: ensure near-vertical
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            if angle < -45:
                angle += 90
            if abs(angle) > 10:
                continue

            # 3) Solidity filter: remove irregular shapes
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            if solidity < 0.8:
                continue

            # Color label based on red mask ratio
            roi = red_mask[y : y + h, x : x + w]
            red_ratio = cv2.countNonZero(roi) / float(w * h)
            label = "Red" if red_ratio > 0.5 else "White"
            color = (0, 0, 255) if label == "Red" else (255, 255, 255)

            # Draw contour and bounding box
            cv2.drawContours(result_image, [cnt], -1, color, 2)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                result_image,
                f"{label} Pipe [{int(area)}]",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Publish result and mask for debugging
        try:
            res_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            res_msg.header = msg.header
            self.image_pub.publish(res_msg)

            mask_msg = self.bridge.cv2_to_imgmsg(
                cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR), "bgr8"
            )
            mask_msg.header = msg.header
            self.detection_pub.publish(mask_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing images: {e}")


def main():
    try:
        detector = PipeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Pipe Detector node terminated.")


if __name__ == "__main__":
    main()
