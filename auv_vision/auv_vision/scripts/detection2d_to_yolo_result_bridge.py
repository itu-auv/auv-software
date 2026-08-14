#!/usr/bin/env python3

"""Wrap a Detection2DArray in the legacy ultralytics_ros/YoloResult type."""

import rospy
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2DArray


class Detection2DToYoloResultBridge:
    def __init__(self):
        input_topic = rospy.get_param("~input_topic")
        output_topic = rospy.get_param("~output_topic")

        self.publisher = rospy.Publisher(output_topic, YoloResult, queue_size=1)
        self.subscriber = rospy.Subscriber(
            input_topic, Detection2DArray, self.callback, queue_size=1
        )
        rospy.loginfo(
            "Detection2DArray -> YoloResult bridge: %s -> %s",
            input_topic,
            output_topic,
        )

    def callback(self, message):
        output = YoloResult()
        output.header = message.header
        output.detections = message
        self.publisher.publish(output)


if __name__ == "__main__":
    rospy.init_node("detection2d_to_yolo_result_bridge")
    Detection2DToYoloResultBridge()
    rospy.spin()
