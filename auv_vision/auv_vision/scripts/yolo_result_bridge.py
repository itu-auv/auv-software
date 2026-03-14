#!/usr/bin/env python3

import rospy
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2DArray


class YoloBridge:
    def __init__(self):
        rospy.init_node("yolo_result_bridge", anonymous=True)

        # Publisher for the standard Detection2DArray message
        self.detection_pub = rospy.Publisher(
            "/yolo_detections", Detection2DArray, queue_size=10
        )
        self.detection_pub_torpedo = rospy.Publisher(
            "/yolo_detections_torpedo", Detection2DArray, queue_size=10
        )

        # Subscriber to the custom YoloResult message
        rospy.Subscriber("/yolo_result_realsense", YoloResult, self.yolo_callback)
        rospy.Subscriber("/yolo_result_torpedo", YoloResult, self.yolo_callback_torpedo)

        rospy.loginfo(
            "YOLO Result Bridge started. Subscribing to /yolo_result_realsense and /yolo_result_torpedo and publishing to /yolo_detections and /yolo_detections_torpedo."
        )

    def yolo_callback(self, msg):
        # The YoloResult message contains a Detection2DArray in its 'detections' field.
        # We simply extract it and republish it.

        # Create a new Detection2DArray message to publish
        detections_msg = msg.detections

        # It's good practice to update the header timestamp
        detections_msg.header.stamp = rospy.Time.now()

        self.detection_pub.publish(detections_msg)

    def yolo_callback_torpedo(self, msg):
        # The YoloResult message contains a Detection2DArray in its 'detections' field.
        # We simply extract it and republish it.

        # Create a new Detection2DArray message to publish
        detections_msg = msg.detections

        # It's good practice to update the header timestamp
        detections_msg.header.stamp = rospy.Time.now()

        self.detection_pub_torpedo.publish(detections_msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        bridge = YoloBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        pass
