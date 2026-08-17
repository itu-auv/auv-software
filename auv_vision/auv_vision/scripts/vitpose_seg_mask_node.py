#!/usr/bin/env python3
"""VitposeResult -> binary mono8 mask Image (the pipe follower's input).

Thresholds one channel of the result's probability masks (or the union of
all of them) with the checkpoint's calibrated threshold and republishes it
as a plain sensor_msgs/Image at source resolution, header preserved. Drop-in
for yolo_seg_to_mask.py on the /seg_mask hook (pipe_frame_publisher).

Params: ~input_topic (VitposeResult), ~output_topic (Image), ~mask_class
(channel name; empty = union of all channels), ~threshold (0..1; negative =
use the message's mask_threshold).
"""

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from auv_msgs.msg import VitposeResult


class VitposeSegMaskNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.mask_class = rospy.get_param("~mask_class", "")
        self.threshold = float(rospy.get_param("~threshold", -1.0))
        self.morph_kernel = int(rospy.get_param("~morph_kernel", 0))
        self.pub = rospy.Publisher(
            rospy.get_param("~output_topic", "/seg_mask"), Image, queue_size=1
        )
        self.sub = rospy.Subscriber(
            rospy.get_param("~input_topic", "/vitpose_result_bottom"),
            VitposeResult,
            self.cb,
            queue_size=1,
            buff_size=2**24,
        )

    def cb(self, msg):
        if not msg.masks:
            return
        if self.mask_class:
            try:
                planes = [msg.masks[list(msg.mask_classes).index(self.mask_class)]]
            except ValueError:
                rospy.logwarn_throttle(
                    5.0,
                    f"mask_class '{self.mask_class}' not in {list(msg.mask_classes)}",
                )
                return
        else:
            planes = list(msg.masks)
        threshold = self.threshold if self.threshold >= 0 else float(msg.mask_threshold)
        cut = int(round(threshold * 255.0))
        out = None
        for plane in planes:
            prob = self.bridge.imgmsg_to_cv2(plane, desired_encoding="mono8")
            binary = (prob > cut).astype(np.uint8) * 255
            out = binary if out is None else cv2.bitwise_or(out, binary)
        if self.morph_kernel > 1:
            kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        image = self.bridge.cv2_to_imgmsg(out, encoding="mono8")
        image.header = msg.header
        self.pub.publish(image)


if __name__ == "__main__":
    rospy.init_node("vitpose_seg_mask_node")
    VitposeSegMaskNode()
    rospy.spin()
