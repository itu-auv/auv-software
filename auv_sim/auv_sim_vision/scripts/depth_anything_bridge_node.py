#!/usr/bin/env python3

"""Relay Gazebo depth data through the Depth Anything ROS interface."""

import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool, SetBoolResponse


class DepthAnythingSimBridge:
    def __init__(self):
        self.enabled = rospy.get_param("~enabled", True)

        self.depth_pub = rospy.Publisher("raw_depth", Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher(
            "scaled_camera_info", CameraInfo, queue_size=1, latch=True
        )

        self.depth_sub = rospy.Subscriber(
            "input_depth", Image, self._depth_callback, queue_size=1
        )
        self.camera_info_sub = rospy.Subscriber(
            "input_camera_info", CameraInfo, self._camera_info_callback, queue_size=1
        )
        self.enable_service = rospy.Service("enable", SetBool, self._enable_callback)

        rospy.loginfo(
            "[DA3-SIM] Bridge ready (%s)",
            "enabled" if self.enabled else "disabled",
        )

    def _depth_callback(self, msg):
        if self.enabled:
            self.depth_pub.publish(msg)

    def _camera_info_callback(self, msg):
        self.camera_info_pub.publish(msg)

    def _enable_callback(self, req):
        self.enabled = req.data
        message = "Depth Anything TRT node " + (
            "enabled" if self.enabled else "disabled"
        )
        rospy.loginfo("[DA3-SIM] %s", message)
        return SetBoolResponse(success=True, message=message)


if __name__ == "__main__":
    try:
        rospy.init_node("depth_anything_trt_node")
        DepthAnythingSimBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
