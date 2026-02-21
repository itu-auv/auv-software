#!/usr/bin/env python3
"""
Camera info publisher node.

Reads camera calibration parameters from ROS parameter server
and publishes sensor_msgs/CameraInfo synchronized with incoming images.
This is used for cameras (e.g. OAK) that don't publish camera_info natively
through our calibration YAML.
"""

import rospy
from sensor_msgs.msg import Image, CameraInfo


class CameraInfoPublisher:
    def __init__(self):
        rospy.init_node("camera_info_publisher", anonymous=False)

        # Load calibration from params
        self.cam_info = CameraInfo()

        self.cam_info.width = rospy.get_param("~image_width", 1280)
        self.cam_info.height = rospy.get_param("~image_height", 720)
        self.cam_info.distortion_model = rospy.get_param(
            "~distortion_model", "plumb_bob"
        )
        self.cam_info.D = rospy.get_param(
            "~distortion_coefficients/data", [0.0, 0.0, 0.0, 0.0, 0.0]
        )
        K = rospy.get_param(
            "~camera_matrix/data",
            [640.0, 0.0, 640.0, 0.0, 640.0, 360.0, 0.0, 0.0, 1.0],
        )
        self.cam_info.K = K

        R = rospy.get_param(
            "~rectification_matrix/data",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        self.cam_info.R = R

        P = rospy.get_param(
            "~projection_matrix/data",
            [640.0, 0.0, 640.0, 0.0, 0.0, 640.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.cam_info.P = P

        rospy.loginfo(
            "Loaded camera info: %dx%d, model=%s",
            self.cam_info.width,
            self.cam_info.height,
            self.cam_info.distortion_model,
        )

        # Publisher
        self.info_pub = rospy.Publisher("camera_info", CameraInfo, queue_size=1)

        # Subscribe to image to sync timestamps
        self.image_sub = rospy.Subscriber("image_raw", Image, self.image_cb)

    def image_cb(self, msg):
        self.cam_info.header = msg.header
        self.info_pub.publish(self.cam_info)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CameraInfoPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
