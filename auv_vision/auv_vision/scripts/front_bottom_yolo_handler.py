#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np

from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2D


class YoloHandler:
    def __init__(self):
        rospy.init_node("front_bottom_yolo_handler_node")
        self.bridge = CvBridge()

        # Params
        self.canvas_w = 1280
        self.canvas_h = 480
        self.split_x = self.canvas_w // 2

        # Image sync & merger
        sub_front = Subscriber("cam_front/image_raw", Image)
        sub_bottom = Subscriber("cam_bottom/image_raw", Image)

        ats = ApproximateTimeSynchronizer(
            [sub_front, sub_bottom], queue_size=10, slop=0.1
        )
        ats.registerCallback(self.image_callback)

        self.merged_pub = rospy.Publisher("/merged_image", Image, queue_size=1)

        # Subscribe to YOLO combined detections
        rospy.Subscriber("/yolo_result", YoloResult, self.yolo_callback)

        # Output separated detections
        self.pub_front = rospy.Publisher(
            "/yolo/front_view_detections", YoloResult, queue_size=1
        )
        self.pub_bottom = rospy.Publisher(
            "/yolo/bottom_view_detections", YoloResult, queue_size=1
        )

    def image_callback(self, front_msg, bottom_msg):
        try:
            img_front = self.bridge.imgmsg_to_cv2(front_msg, "bgr8")
            img_bottom = self.bridge.imgmsg_to_cv2(bottom_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        half_w = self.canvas_w // 2

        # Front image (left)
        y_off = (self.canvas_h - img_front.shape[0]) // 2
        x_off = (half_w - img_front.shape[1]) // 2
        canvas[
            y_off : y_off + img_front.shape[0], x_off : x_off + img_front.shape[1]
        ] = img_front

        # Bottom image (right)
        y_off2 = (self.canvas_h - img_bottom.shape[0]) // 2
        x_off2 = half_w + (half_w - img_bottom.shape[1]) // 2
        canvas[
            y_off2 : y_off2 + img_bottom.shape[0], x_off2 : x_off2 + img_bottom.shape[1]
        ] = img_bottom

        try:
            out = self.bridge.cv2_to_imgmsg(canvas, "bgr8")
            out.header = front_msg.header
            self.merged_pub.publish(out)
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)

    def yolo_callback(self, msg: YoloResult):
        front_result = YoloResult()
        bottom_result = YoloResult()

        front_result.header = msg.header
        bottom_result.header = msg.header

        front_result.masks = msg.masks
        bottom_result.masks = msg.masks

        front_result.detections.header = msg.detections.header
        bottom_result.detections.header = msg.detections.header

        for i, det in enumerate(msg.detections.detections):
            x = det.bbox.center.x

            if x < self.split_x:
                front_result.detections.detections.append(
                    self.shift_detection(det, shift_x=0)
                )
            else:
                bottom_result.detections.detections.append(
                    self.shift_detection(det, shift_x=-self.split_x)
                )

        self.pub_front.publish(front_result)
        self.pub_bottom.publish(bottom_result)

    def shift_detection(self, det: Detection2D, shift_x=0):
        new_det = Detection2D()
        new_det.header = det.header
        new_det.results = det.results
        new_det.source_img = det.source_img

        new_det.bbox.center.x = det.bbox.center.x + shift_x
        new_det.bbox.center.y = det.bbox.center.y
        new_det.bbox.size_x = det.bbox.size_x
        new_det.bbox.size_y = det.bbox.size_y

        return new_det

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    YoloHandler().spin()
