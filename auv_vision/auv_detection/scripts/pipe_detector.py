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
        # White color range
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])

        # Red color range (covering both low and high hue values)
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])

        # Minimum contour area to filter noise
        self.min_contour_area = 500

        rospy.loginfo("Pipe Detector initialized. Subscribing to camera/image_raw")

    def create_color_mask(self, hsv_image):
        """Create masks for white and red colors"""
        # White mask
        white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)

        # Red mask (combining two ranges)
        red_mask1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Combine white and red masks
        combined_mask = cv2.bitwise_or(white_mask, red_mask)

        return combined_mask, white_mask, red_mask

    def find_pipe_contours(self, mask):
        """Find contours in the mask and filter by area"""
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area
        filtered_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area
        ]

        return filtered_contours

    def draw_detection_results(self, image, contours, color, label):
        """Draw bounding rectangles around detected contours"""
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Draw contour
            cv2.drawContours(image, [contour], -1, color, 2)

            # Add label
            cv2.putText(
                image,
                f"{label} Pipe",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Add area information
            area = cv2.contourArea(contour)
            cv2.putText(
                image,
                f"Area: {int(area)}",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    def image_callback(self, msg):
        """Main callback function for processing camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert to HSV for better color detection
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Create color masks
            combined_mask, white_mask, red_mask = self.create_color_mask(hsv_image)

            # Find contours for white and red pipes separately
            white_contours = self.find_pipe_contours(white_mask)
            red_contours = self.find_pipe_contours(red_mask)

            # Create result image (copy of original)
            result_image = cv_image.copy()

            # Draw detection results
            self.draw_detection_results(
                result_image, white_contours, (255, 255, 255), "White"
            )
            self.draw_detection_results(result_image, red_contours, (0, 0, 255), "Red")

            # Add detection count information
            total_detections = len(white_contours) + len(red_contours)
            cv2.putText(
                result_image,
                f"White Pipes: {len(white_contours)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                result_image,
                f"Red Pipes: {len(red_contours)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                result_image,
                f"Total: {total_detections}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Publish result image
            try:
                result_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                result_msg.header = msg.header
                self.image_pub.publish(result_msg)

                # Also publish the combined mask for debugging
                mask_colored = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_colored, "bgr8")
                mask_msg.header = msg.header
                self.detection_pub.publish(mask_msg)

            except Exception as e:
                rospy.logerr(f"Error publishing images: {e}")

        except Exception as e:
            rospy.logerr(f"Error in image processing: {e}")


def main():
    try:
        detector = PipeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Pipe Detector node terminated.")
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")


if __name__ == "__main__":
    main()
