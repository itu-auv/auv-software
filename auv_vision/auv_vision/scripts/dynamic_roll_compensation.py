#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import math
import angles  # ROS angles library
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion
import threading
import queue


class DynamicRollCompensationNode:
    def __init__(self):
        rospy.init_node("dynamic_roll_compensation", anonymous=True)
        rospy.loginfo("Dynamic roll compensation node started.")

        # Initialize variables
        self.bridge = CvBridge()
        self.roll_angle_deg = 0.0
        self.lock = threading.Lock()
        self.image_queue = queue.Queue(maxsize=1)  # Only keep the most recent image
        self.processing = False

        # Pre-allocate rotation matrix
        self.M = None
        self.last_roll = None
        self.last_dimensions = None

        # Set up subscribers with appropriate queue sizes
        self.odom_subscriber = rospy.Subscriber(
            "odometry", Odometry, self.odom_callback, queue_size=1, tcp_nodelay=True
        )

        self.image_sub = rospy.Subscriber(
            "camera/image_rect_color",
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,  # Increase buffer size for large images
        )

        self.image_pub = rospy.Publisher("camera/image_corrected", Image, queue_size=2)

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def odom_callback(self, odom_msg):
        q = [
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        ]
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = euler_from_quaternion(q)

        # Compute correction angle (in radians) as the shortest angular distance from roll to 0.
        correction_rad = angles.shortest_angular_distance(roll, 0.0)

        with self.lock:
            self.roll_angle_deg = math.degrees(correction_rad)

    def image_callback(self, img_msg):
        # Just put the image in the queue and return immediately
        # This prevents blocking in the callback
        try:
            if not self.image_queue.full():
                self.image_queue.put_nowait(img_msg)
            else:
                # If queue is full, replace the old image with the new one
                try:
                    self.image_queue.get_nowait()
                    self.image_queue.put_nowait(img_msg)
                except queue.Empty:
                    pass  # Queue was emptied by processing thread
        except queue.Full:
            pass  # Queue is full, skip this frame

    def process_images(self):
        while not rospy.is_shutdown():
            try:
                # Get the next image from the queue
                img_msg = self.image_queue.get(timeout=0.1)

                # Mark that we're processing an image
                self.processing = True

                try:
                    # Convert the ROS Image message to an OpenCV image
                    cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                except CvBridgeError as e:
                    rospy.logerr("CvBridge error: %s", e)
                    self.processing = False
                    continue

                # Get image dimensions
                (h, w) = cv_image.shape[:2]
                center = (w // 2, h // 2)

                # Get current roll angle (thread-safe)
                with self.lock:
                    roll_angle = self.roll_angle_deg

                # Only recompute rotation matrix if roll changed or dimensions changed
                if (
                    self.last_roll is None
                    or abs(self.last_roll - roll_angle) > 0.1
                    or self.last_dimensions != (w, h)
                ):

                    self.M = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
                    self.last_roll = roll_angle
                    self.last_dimensions = (w, h)

                # Apply rotation - use INTER_LINEAR for better speed
                rotated_image = cv2.warpAffine(
                    cv_image,
                    self.M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

                try:
                    # Convert back to ROS Image message
                    corrected_img_msg = self.bridge.cv2_to_imgmsg(rotated_image, "bgr8")
                    corrected_img_msg.header = img_msg.header  # Keep original header

                    # Publish the corrected image
                    self.image_pub.publish(corrected_img_msg)
                except CvBridgeError as e:
                    rospy.logerr("CvBridge error while converting rotated image: %s", e)

                # Done processing
                self.processing = False

            except queue.Empty:
                # No images in queue, just continue
                pass
            except Exception as e:
                rospy.logerr("Error in image processing thread: %s", e)
                self.processing = False


def main():
    try:
        node = DynamicRollCompensationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Dynamic roll compensation node terminated.")


if __name__ == "__main__":
    main()
