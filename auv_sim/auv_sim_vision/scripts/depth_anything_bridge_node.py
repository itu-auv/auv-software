#!/usr/bin/env python3

"""Relay Gazebo depth data through the Depth Anything ROS interface."""

import rclpy
from auv_common_lib import ros
from auv_common_lib.ros import get_param, latched_qos, get_node, init_node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool


class DepthAnythingSimBridge:
    def __init__(self):
        self.enabled = get_param(get_node(), "~enabled", True)

        self.depth_pub = get_node().create_publisher(Image, "raw_depth", 1)
        self.camera_info_pub = get_node().create_publisher(
            CameraInfo, "scaled_camera_info", latched_qos()
        )

        self.depth_sub = get_node().create_subscription(
            Image, "input_depth", self._depth_callback, 1
        )
        self.camera_info_sub = get_node().create_subscription(
            CameraInfo, "input_camera_info", self._camera_info_callback, 1
        )
        self.enable_service = get_node().create_service(
            SetBool, "enable", self._enable_callback
        )

        get_node().get_logger().info(
            "[DA3-SIM] Bridge ready (%s)" % ("enabled" if self.enabled else "disabled",)
        )

    def _depth_callback(self, msg):
        if self.enabled:
            self.depth_pub.publish(msg)

    def _camera_info_callback(self, msg):
        self.camera_info_pub.publish(msg)

    def _enable_callback(self, req, response):
        self.enabled = req.data
        message = "Depth Anything TRT node " + (
            "enabled" if self.enabled else "disabled"
        )
        get_node().get_logger().info("[DA3-SIM] %s" % (message,))
        return SetBool.Response(success=True, message=message)


def main():
    init_node("depth_anything_trt_node")
    DepthAnythingSimBridge()
    ros.spin(get_node())


if __name__ == "__main__":
    main()
