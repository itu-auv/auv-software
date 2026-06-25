#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import SlalomTarget

from mini_slalom_core import Detection, select_target


class MiniSlalomBboxNode:
    def __init__(self):
        rospy.init_node("mini_slalom_bbox_node")
        self.image_width = float(rospy.get_param("~image_width", 1280))
        self.image_height = float(rospy.get_param("~image_height", 720))
        self.red_class_id = int(rospy.get_param("~red_class_id", 2))
        self.white_class_id = int(rospy.get_param("~white_class_id", 3))
        self.min_confidence = float(rospy.get_param("~min_confidence", 0.25))
        self.min_separation_ratio = float(
            rospy.get_param("~min_separation_ratio", 0.04)
        )
        self.max_separation_ratio = float(
            rospy.get_param("~max_separation_ratio", 0.95)
        )
        self.direction = rospy.get_param("~default_direction", "left")
        rospy.loginfo(
            "Mini slalom YOLO classes: red=%d white=%d",
            self.red_class_id,
            self.white_class_id,
        )

        self.target_pub = rospy.Publisher("slalom/target", SlalomTarget, queue_size=1)
        rospy.Subscriber(
            "slalom/direction", String, self.direction_callback, queue_size=1
        )
        rospy.Subscriber("yolo_result", YoloResult, self.yolo_callback, queue_size=1)

    def direction_callback(self, msg):
        if msg.data in ("left", "right"):
            self.direction = msg.data
        else:
            rospy.logwarn_throttle(
                2.0, f"Ignoring invalid mini slalom direction: {msg.data}"
            )

    def yolo_callback(self, msg):
        detections = []
        for detection in msg.detections.detections:
            if not detection.results:
                continue
            hypothesis = detection.results[0]
            detections.append(
                Detection(
                    class_id=int(hypothesis.id),
                    center_x=float(detection.bbox.center.x),
                    center_y=float(detection.bbox.center.y),
                    width=float(detection.bbox.size_x),
                    height=float(detection.bbox.size_y),
                    confidence=float(hypothesis.score),
                )
            )

        target = select_target(
            detections,
            self.image_width,
            self.image_height,
            self.direction,
            self.red_class_id,
            self.white_class_id,
            self.min_confidence,
            self.min_separation_ratio,
            self.max_separation_ratio,
        )
        output = SlalomTarget()
        output.header = msg.header
        if output.header.stamp.is_zero():
            output.header.stamp = rospy.Time.now()
        output.valid = target.valid
        output.center_error = target.center_error
        output.gate_width_ratio = target.gate_width_ratio
        output.gate_height_ratio = target.gate_height_ratio
        output.red_center_x = target.red_center_x
        output.white_center_x = target.white_center_x
        output.confidence = target.confidence
        self.target_pub.publish(output)


if __name__ == "__main__":
    try:
        MiniSlalomBboxNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
