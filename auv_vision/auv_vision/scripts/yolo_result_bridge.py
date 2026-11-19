#!/usr/bin/env python3

import rclpy
from auv_common_lib import ros
from rclpy.node import Node
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2DArray


class YoloBridge(Node):
    def __init__(self):
        super().__init__("yolo_result_bridge")

        self.detection_pub = self.create_publisher(
            Detection2DArray, "/yolo_detections", 10
        )
        self.detection_pub_torpedo = self.create_publisher(
            Detection2DArray, "/yolo_detections_torpedo", 10
        )

        # Subscriber to the custom YoloResult message
        self.create_subscription(
            YoloResult, "/yolo_result_realsense", self.yolo_callback, 10
        )
        self.create_subscription(
            YoloResult, "/yolo_result_torpedo_front", self.yolo_callback_torpedo, 10
        )

        self.get_logger().info(
            "YOLO Result Bridge started. Subscribing to /yolo_result_realsense and /yolo_result_torpedo_front and publishing to /yolo_detections and /yolo_detections_torpedo."
        )

    def yolo_callback(self, msg):
        detections_msg = msg.detections

        detections_msg.header = msg.header
        self.detection_pub.publish(detections_msg)

    def yolo_callback_torpedo(self, msg):
        detections_msg = msg.detections

        detections_msg.header = msg.header
        self.detection_pub_torpedo.publish(detections_msg)


def main():
    ros.init()
    node = YoloBridge()
    ros.spin(node)


if __name__ == "__main__":
    main()
