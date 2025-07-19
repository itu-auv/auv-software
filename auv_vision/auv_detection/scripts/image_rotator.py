#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageRotator:
    def __init__(self):
        rospy.init_node("image_rotator", anonymous=True)

        # Parameters
        # Defaults are relative to the node's namespace
        self.input_topic = rospy.get_param("~input_topic", "camera/color/image_raw")
        self.output_topic = rospy.get_param(
            "~output_topic", "camera/color/image_rotated"
        )

        # CV Bridge
        self.bridge = CvBridge()

        # Publisher and Subscriber
        self.image_pub = rospy.Publisher(self.output_topic, Image, queue_size=10)
        self.image_sub = rospy.Subscriber(self.input_topic, Image, self.image_callback)

        rospy.loginfo(
            f"Image Rotator node started. Subscribing to '{self.input_topic}' and publishing to '{self.output_topic}'."
        )

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Rotate the image 180 degrees
        rotated_image = cv2.rotate(cv_image, cv2.ROTATE_180)

        try:
            # Convert rotated OpenCV image back to ROS Image message
            rotated_msg = self.bridge.cv2_to_imgmsg(rotated_image, "bgr8")
            # Preserve the header from the original message
            rotated_msg.header = msg.header
            self.image_pub.publish(rotated_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        rotator = ImageRotator()
        rotator.run()
    except rospy.ROSInterruptException:
        pass
