#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from ultralytics_ros.msg import YoloResult


class YoloSegToMaskNode:
    def __init__(self):
        # Parameters
        self.target_class_ids = set(rospy.get_param("~target_class_ids", [0]))
        self.conf_min = float(rospy.get_param("~conf_min", 0.25))
        self.morph_kernel = int(rospy.get_param("~morph_kernel", 0))
        self.publish_debug = bool(rospy.get_param("~publish_debug", False))
        self.debug_topic = rospy.get_param("~debug_topic", "seg_mask_debug")

        self.bridge = CvBridge()
        self.pub_mask = rospy.Publisher("/seg_mask", Image, queue_size=1)
        self.pub_debug = (
            rospy.Publisher(self.debug_topic, Image, queue_size=1)
            if self.publish_debug
            else None
        )

        self.sub = rospy.Subscriber(
            "/yolo_result_seg",
            YoloResult,
            self.cb_result,
            queue_size=1,
            buff_size=2**24,
        )

    def cb_result(self, msg: YoloResult):
        detections: Detection2DArray = msg.detections
        masks_msgs = list(msg.masks) if msg.masks else []

        # If there are no masks, publish an empty image (skip if we don't know the size)
        if len(masks_msgs) == 0 or len(detections.detections) == 0:
            rospy.logdebug_throttle(
                2.0,
                "[yolo_seg_to_mask] no masks/detections -> publish empty if possible",
            )
            return  # if we don't have size information, let's wait silently

        # Get the size from the first mask
        mask0 = self.bridge.imgmsg_to_cv2(masks_msgs[0], desired_encoding="mono8")
        h, w = mask0.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)

        # Mask/detection lengths can deviate; take the common minimum for synchronization
        N = min(len(masks_msgs), len(detections.detections))

        for i in range(N):
            det = detections.detections[i]
            # default: single hypothesis, skip if none
            if len(det.results) == 0:
                continue
            cls_id = int(det.results[0].id)
            conf = float(det.results[0].score)

            if cls_id not in self.target_class_ids:
                continue
            if conf < self.conf_min:
                continue

            m = self.bridge.imgmsg_to_cv2(masks_msgs[i], desired_encoding="mono8")
            # m is already 0/255; threshold for safety
            m_bin = (m > 127).astype(np.uint8) * 255
            out = cv2.bitwise_or(out, m_bin)

        # Morphological correction (optional)
        if self.morph_kernel > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
            )
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)

        # Publish
        out_msg = self.bridge.cv2_to_imgmsg(out, encoding="mono8")
        out_msg.header = msg.header  # YoloResult header for time synchronization
        self.pub_mask.publish(out_msg)

        # Optional debug overlay
        if self.pub_debug is not None:
            overlay = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                overlay,
                f"cls={sorted(self.target_class_ids)} conf>={self.conf_min:.2f}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            dbg_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            dbg_msg.header = msg.header
            self.pub_debug.publish(dbg_msg)


def main():
    rospy.init_node("yolo_seg_to_mask")
    YoloSegToMaskNode()
    rospy.spin()


if __name__ == "__main__":
    main()
