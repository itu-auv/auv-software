#!/usr/bin/env python3

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image


class PrincipalPointOverlayNode:
    def __init__(self):
        rospy.init_node("principal_point_overlay")

        self.bridge = CvBridge()
        self.camera_info = None
        self.image_topic = rospy.get_param("~image_topic", "image_rect_color")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "camera_info")
        self.image_out_topic = rospy.get_param(
            "~image_out_topic", "principal_point/image"
        )
        self.require_camera_info = rospy.get_param("~require_camera_info", True)
        self.principal_point_matrix = rospy.get_param("~principal_point_matrix", "P")
        self.point_radius = rospy.get_param("~point_radius", 8)
        self.line_length = rospy.get_param("~line_length", 10)
        self.thickness = rospy.get_param("~thickness", 1)
        self.color = tuple(rospy.get_param("~color_bgr", [0, 0, 255]))

        self.image_pub = rospy.Publisher(self.image_out_topic, Image, queue_size=1)
        rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self._camera_info_callback,
            queue_size=1,
        )
        rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1)

        rospy.loginfo(
            "Principal point overlay started. Subscribed to '%s' and '%s', "
            "publishing '%s'.",
            self.image_topic,
            self.camera_info_topic,
            self.image_out_topic,
        )

    def _camera_info_callback(self, msg):
        self.camera_info = msg

    def _image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, f"Could not convert image: {exc}")
            return

        point = self._principal_point_for_image(image)
        if point is None:
            return

        self._draw_principal_point(image, point)

        try:
            output_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, f"Could not convert overlay image: {exc}")
            return

        output_msg.header = msg.header
        self.image_pub.publish(output_msg)

    def _principal_point_for_image(self, image):
        image_height, image_width = image.shape[:2]

        if self.camera_info is None:
            if self.require_camera_info:
                rospy.logwarn_throttle(
                    5.0,
                    "Waiting for camera_info before publishing principal point overlay.",
                )
                return None

            return image_width / 2.0, image_height / 2.0

        matrix = self.principal_point_matrix.upper()
        if matrix == "K":
            cx = self.camera_info.K[2]
            cy = self.camera_info.K[5]
        elif matrix == "P":
            cx = self.camera_info.P[2]
            cy = self.camera_info.P[6]
        else:
            rospy.logwarn_throttle(
                5.0,
                "Unknown principal_point_matrix '%s'. Use 'K' or 'P'.",
                self.principal_point_matrix,
            )
            return None

        if self.camera_info.width > 0 and self.camera_info.height > 0:
            cx *= float(image_width) / float(self.camera_info.width)
            cy *= float(image_height) / float(self.camera_info.height)

        return cx, cy

    def _draw_principal_point(self, image, point):
        x = int(round(point[0]))
        y = int(round(point[1]))

        cv2.circle(image, (x, y), self.point_radius, self.color, self.thickness)
        cv2.line(
            image,
            (x - self.line_length, y),
            (x + self.line_length, y),
            self.color,
            self.thickness,
        )
        cv2.line(
            image,
            (x, y - self.line_length),
            (x, y + self.line_length),
            self.color,
            self.thickness,
        )


if __name__ == "__main__":
    node = PrincipalPointOverlayNode()
    rospy.spin()
