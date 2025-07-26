#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import WrenchStamped


class RobustPipeFollower:
    def __init__(self):
        rospy.init_node("robust_pipe_follower")

        # ROS setup
        self.bridge = CvBridge()
        self.wrench_pub = rospy.Publisher("/taluy/wrench", WrenchStamped, queue_size=10)
        self.debug_pub = rospy.Publisher("/pipe_follower/debug", Image, queue_size=10)

        # Parameters
        self.num_segments = 5
        self.min_contour_area = 150
        self.base_speed = 8.0
        self.kp = 0.35
        self.ki = 0.01
        self.kd = 0.04
        self.weights = [0.1, 0.2, 0.3, 0.4]  # Bottom segments have more weight

        # Image processing
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 40])

        # Control variables
        self.integral = 0
        self.last_error = 0
        self.filtered_angle = 0
        self.last_time = rospy.Time.now()

        # Subscriber
        self.image_sub = rospy.Subscriber(
            "/taluy/cameras/cam_front/image_rect_color/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("Robust Pipe Follower initialized")

    def segment_pipeline(self, mask):
        """Divide pipeline into horizontal segments"""
        height, width = mask.shape
        segment_height = height // self.num_segments
        segments = []

        for i in range(self.num_segments):
            y_start = i * segment_height
            y_end = (i + 1) * segment_height
            segment = mask[y_start:y_end, :]
            segments.append(segment)

        return segments

    def find_segment_midpoint(self, segment):
        """Find center point of a pipe segment with contour filtering"""
        contours, _ = cv2.findContours(
            segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Filter small contours
        valid_contours = [
            c for c in contours if cv2.contourArea(c) > self.min_contour_area
        ]
        if not valid_contours:
            return None

        # Find most central contour
        width = segment.shape[1]
        best_contour = min(
            valid_contours,
            key=lambda c: abs(
                cv2.moments(c)["m10"] / cv2.moments(c)["m00"] - width / 2
            ),
        )

        M = cv2.moments(best_contour)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def calculate_pipe_direction(self, midpoints, img_width):
        """Improved direction calculation using weighted vector sum"""
        if len(midpoints) < 2:
            return 0.0, 0.0  # angle_error, position_error

        # Use weighted segments (bottom segments have more influence)
        used_points = midpoints[-min(4, len(midpoints)) :]  # Use max 4 segments
        weights = self.weights[-len(used_points) :]  # Apply corresponding weights

        total_vector = np.zeros(2)
        total_weight = 0.0

        for i in range(len(used_points) - 1):
            dx = used_points[i + 1][0] - used_points[i][0]
            dy = used_points[i + 1][1] - used_points[i][1]
            weight = weights[i]

            total_vector += weight * np.array([dx, dy])
            total_weight += weight

        if total_weight == 0:
            return 0.0, 0.0

        avg_vector = total_vector / total_weight
        angle_error = np.degrees(np.arctan2(avg_vector[1], avg_vector[0]))
        position_error = midpoints[-1][0] - (img_width // 2)

        return angle_error, position_error

    def pid_control(self, angle_error, position_error, dt):
        """Enhanced PID controller with smoothing"""
        self.integral += angle_error * dt
        derivative = (angle_error - self.last_error) / dt if dt > 0 else 0

        # Smooth angle error
        self.filtered_angle = 0.7 * self.filtered_angle + 0.3 * angle_error

        # Combined error with position compensation
        total_error = self.filtered_angle + (position_error / 100.0)

        output = self.kp * total_error + self.ki * self.integral + self.kd * derivative

        self.last_error = angle_error
        return np.clip(output, -1.5, 1.5)

    def image_callback(self, msg):
        try:
            # Timing
            now = rospy.Time.now()
            dt = (now - self.last_time).to_sec()
            self.last_time = now

            # Convert image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            height, width = cv_image.shape[:2]
            roi = cv_image[int(height * 0.4) :, :]  # Lower 60% of image

            # Preprocessing
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            # Segment analysis
            segments = self.segment_pipeline(mask)
            midpoints = []

            for i, seg in enumerate(segments):
                if mp := self.find_segment_midpoint(seg):
                    midpoints.append(mp)

            # Create debug image
            debug_img = roi.copy()
            cv2.line(debug_img, (width // 2, 0), (width // 2, height), (0, 0, 255), 2)

            if len(midpoints) >= 2:
                # Calculate direction
                angle_error, pos_error = self.calculate_pipe_direction(midpoints, width)
                torque = self.pid_control(angle_error, pos_error, dt)

                # Draw debug info
                for i in range(len(midpoints) - 1):
                    cv2.line(debug_img, midpoints[i], midpoints[i + 1], (0, 255, 0), 2)

                cv2.circle(debug_img, midpoints[-1], 7, (255, 0, 0), -1)

                # Apply control
                wrench = WrenchStamped()
                wrench.header.stamp = now
                wrench.header.frame_id = "taluy/base_link"
                wrench.wrench.force.x = self.base_speed * (1 - 0.3 * abs(torque))
                wrench.wrench.torque.z = torque

                # Debug text
                texts = [
                    f"Angle: {angle_error:.1f}°",
                    f"Position: {pos_error:.1f}px",
                    f"Torque: {torque:.2f}",
                    f"Segments: {len(midpoints)}/{self.num_segments}",
                ]

                for i, text in enumerate(texts):
                    cv2.putText(
                        debug_img,
                        text,
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
            else:
                wrench = self.create_search_wrench()
                cv2.putText(
                    debug_img,
                    "SEARCHING...",
                    (width // 2 - 100, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            # Publish results
            self.wrench_pub.publish(wrench)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")
            wrench = WrenchStamped()
            wrench.header.stamp = rospy.Time.now()
            self.wrench_pub.publish(wrench)

    def create_search_wrench(self):
        """Create gentle search pattern"""
        wrench = WrenchStamped()
        wrench.header.stamp = rospy.Time.now()
        wrench.header.frame_id = "taluy/base_link"
        wrench.wrench.force.x = self.base_speed * 0.4
        wrench.wrench.torque.z = 0.8
        return wrench


if __name__ == "__main__":
    try:
        follower = RobustPipeFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
