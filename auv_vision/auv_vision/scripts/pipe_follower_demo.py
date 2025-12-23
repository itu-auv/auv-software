#!/usr/bin/env python3
import math
import numpy as np
import cv2

from cv_bridge import CvBridge
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from auv_msgs.srv import (
    AlignFrameController,
    AlignFrameControllerResponse,
    SetObjectTransform,
    SetObjectTransformRequest,
)

from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import convolve


class PipeFollowerDemo:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher("taluy/cmd_vel", Twist, queue_size=1)
        self.sub_mask = rospy.Subscriber(
            "yolo_fake_image", Image, self.cb_mask, queue_size=1, buff_size=2**24
        )
        self.pub_debug = rospy.Publisher("pipe_result_debug", Image, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        # TODO: figure out what's the difference between tf_buffer and tf_listener
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "/taluy/map/set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.pipe_carrot_frame = "pipe_carrot"

        pose = Pose()
        pose.position.x = 2
        pose.position.y = 2
        pose.position.z = 0
        pose.orientation.x = 1
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 0
        msg = self._build_transform_message(self.pipe_carrot_frame, pose)
        self._send_transform(msg)

        self.align_frame_service = rospy.ServiceProxy(
            "/taluy/control/align_frame/start", AlignFrameController
        )
        try:
            res = self.align_frame_service(
                "taluy/base_link", self.pipe_carrot_frame, 0, False, 0, 0
            )
            print(res)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        self.count = 0

    def cb_mask(self, msg):
        self.count += 1
        if self.count % 10 != 0:
            pass
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge err: %s", e)
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        skel_bool = skeletonize(binary > 0)
        skel = (skel_bool.astype(np.uint8)) * 255
        skel_rgb = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        ordered_lines = self._get_ordered_points_from_skel(skel)

        final_points_list = []

        for line in ordered_lines:
            cnt_format = line.reshape(-1, 1, 2).astype(np.int32)
            line_len = cv2.arcLength(cnt_format, closed=False)
            approx = cv2.approxPolyDP(cnt_format, 0.02 * line_len, closed=False)

            pts = approx.reshape(-1, 2)
            aaa = self._filter_close_points(pts, min_dist=30.0)
            final_points_list.append(aaa)

        segments = self._merge_into_segments(final_points_list, 50)
        segments = self._filter_segments(segments, 50)

        for seg in segments:
            for i in range(1, len(seg)):
                cv2.line(skel_rgb, seg[i - 1], seg[i], (0, 255, 0), 2)

        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for seg in segments:
            for i, pt in enumerate(seg):
                color = COLORS[i % len(COLORS)]
                cv2.circle(skel_rgb, pt, 4, color, -1)

        img_msg = self.bridge.cv2_to_imgmsg(skel_rgb, encoding="bgr8")
        img_msg.header = msg.header
        self.pub_debug.publish(img_msg)

        vel_msg = Twist()
        # vel_msg.linear.x = 0.1
        self.pub_cmd.publish(vel_msg)

    def _filter_close_points(self, points, min_dist=5.0):
        if len(points) <= 2:
            return points

        filtered = [points[0]]
        for i in range(1, len(points)):
            dist = np.linalg.norm(np.array(points[i]) - np.array(filtered[-1]))

            if dist >= min_dist or i == len(points) - 1:
                filtered.append(points[i])

        return np.array(filtered)

    def _get_ordered_points_from_skel(self, skeleton_img):
        white_pixels = np.column_stack(np.where(skeleton_img > 0))
        pixel_set = set(tuple(p) for p in white_pixels)

        neighbors = [
            (1, 0),
            (1, 1),
            (1, -1),
            (0, 1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ]

        paths = []
        while pixel_set:
            start_pixel = next(iter(pixel_set))

            path = [start_pixel]
            pixel_set.remove(start_pixel)

            changed = True
            while changed:
                changed = False
                curr = path[-1]
                for dy, dx in neighbors:
                    neighbor = (curr[0] + dy, curr[1] + dx)
                    if neighbor in pixel_set:
                        path.append(neighbor)
                        pixel_set.remove(neighbor)
                        changed = True
                        break

            changed = True
            while changed:
                changed = False
                curr = path[0]
                for dy, dx in neighbors:
                    neighbor = (curr[0] + dy, curr[1] + dx)
                    if neighbor in pixel_set:
                        path.insert(0, neighbor)
                        pixel_set.remove(neighbor)
                        changed = True
                        break

            paths.append(np.array([[p[1], p[0]] for p in path], dtype=np.float32))
        return paths

    def _merge_into_segments(self, final_points_list, max_seg_dist):
        segments = []
        max_seg_dist = 100

        remaining = [line.tolist() for line in final_points_list]

        while remaining:
            current = remaining.pop(0)
            match_found = True
            while match_found:
                match_found = False
                for i, line in enumerate(remaining):
                    dists = {
                        "hh": self._get_dist(current[0], line[0]),
                        "th": self._get_dist(current[-1], line[0]),
                        "ht": self._get_dist(current[0], line[-1]),
                        "tt": self._get_dist(current[-1], line[-1]),
                    }
                    best = min(dists, key=dists.get)
                    if dists[best] < max_seg_dist:
                        match_line = remaining.pop(i)
                        if best == "hh":
                            current = match_line[::-1] + current
                        elif best == "th":
                            current = current + match_line
                        elif best == "ht":
                            current = match_line + current
                        elif best == "tt":
                            current = current + match_line[::-1]

                        match_found = True
                        break
            segments.append(current)
        return segments

    def _get_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _filter_segments(self, segments, min_len=50):
        return list(
            filter(
                lambda x: cv2.arcLength(np.array(x), closed=False) >= min_len, segments
            )
        )

    def _build_transform_message(
        self, child_frame_id: str, pose: Pose
    ) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "odom"
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def _send_transform(self, transform):
        req = SetObjectTransformRequest()
        req.transform = transform
        try:
            resp = self.set_object_transform_service.call(req)
            if not resp.success:
                print(
                    f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
                )
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")


def main():
    rospy.init_node("pipe_follower_demo")
    PipeFollowerDemo()
    rospy.spin()


if __name__ == "__main__":
    main()
