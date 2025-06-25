#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber


class DualImageMerger:
    def __init__(self):
        rospy.init_node("dual_image_merger")
        self.bridge = CvBridge()

        # Replace these with your actual topic names:
        sub_front = Subscriber("cam_front/image_raw", Image)
        sub_bottom = Subscriber("cam_bottom/image_raw", Image)

        # approximately sync by timestamp
        ats = ApproximateTimeSynchronizer(
            [sub_front, sub_bottom], queue_size=10, slop=0.1
        )
        ats.registerCallback(self.callback)

        self.pub = rospy.Publisher("merged_image", Image, queue_size=1)

    def callback(self, front_msg, bottom_msg):
        try:
            img_front = self.bridge.imgmsg_to_cv2(front_msg, "bgr8")
            img_bottom = self.bridge.imgmsg_to_cv2(bottom_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        # create a black canvas
        canvas_h, canvas_w = 480, 1280
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        half_w = canvas_w // 2

        # Place front image (left) centered in the left half
        img_h, img_w = img_front.shape[:2]
        y_offset = (canvas_h - img_h) // 2
        x_offset = (half_w - img_w) // 2
        canvas[y_offset : y_offset + img_h, x_offset : x_offset + img_w] = img_front

        # Place bottom image (right) centered in the right half
        img_h2, img_w2 = img_bottom.shape[:2]
        y_offset2 = (canvas_h - img_h2) // 2
        x_offset2 = half_w + (half_w - img_w2) // 2
        canvas[y_offset2 : y_offset2 + img_h2, x_offset2 : x_offset2 + img_w2] = (
            img_bottom
        )

        try:
            out = self.bridge.cv2_to_imgmsg(canvas, "bgr8")
            out.header = front_msg.header
            self.pub.publish(out)
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    DualImageMerger().spin()
