#!/usr/bin/env python3

# TODO: ortaya en yakın possible_target'ı target seçmek doğru değil önümüzdeki(x > 0) en yakın pointi seçmemiz lazım.
# TODO: target segmenti seçme doğru değil img_mid'e olan uzakların ortalaması değil farklı bir yöntem gerekli.

import math
import numpy as np
import cv2

from cv_bridge import CvBridge
import rospy
import tf2_ros
import tf
import angles
from tf.transformations import *
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
from camera_detection_pose_estimator import CameraCalibration


class PipeFollowerDemo:
    def __init__(self):
        self.started = False

        self.bridge = CvBridge()
        self.cam = CameraCalibration("taluy/cameras/cam_bottom")
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
        self.pipe_temp_frame = "pipe_temp"
        self.bottom_cam_frame = "taluy/base_link/bottom_camera_optical_link"
        self.taluy_base_frame = "taluy/base_link"

        pose = Pose()
        pose.position.x = 0
        pose.position.y = 0
        pose.position.z = 0
        pose.orientation.x = 1
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 0
        msg = self._build_transform_message(
            self.pipe_carrot_frame,
            pose,
            frame=self.bottom_cam_frame,
        )
        self._send_transform(msg)
        msg = self._build_transform_message(
            self.pipe_temp_frame,
            pose,
            frame=self.bottom_cam_frame,
        )
        self._send_transform(msg)

        self.align_frame_service = rospy.ServiceProxy(
            "/taluy/control/align_frame/start", AlignFrameController
        )
        try:
            if True:
                res = self.align_frame_service(
                    self.taluy_base_frame, self.pipe_carrot_frame, 0, False, 0.3, 0.3
                )
                print(res)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))

    def cb_mask(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge err: %s", e)
            return

        starting_pixel = [img.shape[1] // 2, 1]  # Mid right/left
        mid_img = [320, 240]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        widths = dist * 2

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
            # TODO: this is not working as expected
            aaa = self._filter_close_points(pts, 50.0)
            final_points_list.append(aaa)

        segments = self._merge_into_segments(final_points_list, 50)
        segments = self._filter_segments(segments, 100)
        # TODO: this is not a great way!!!
        # segments.sort(
        #     key=lambda x: cv2.arcLength(np.array(x), closed=False), reverse=True
        # )
        segments.sort(
            key=lambda x: sum(math.dist(c, mid_img) for c in x) / len(x), reverse=False
        )

        target = None

        if len(segments) > 0:
            seg = segments[0]
            # make first segment from right to left (according to image)
            if seg[0][0] < seg[-1][0]:
                seg.reverse()
                segments[0] = seg

            seg.reverse()
            changed = True
            possible_targets = []
            for i in range(len(seg) - 1):
                k = seg[i]
                l = seg[i + 1]
                ang = (self._get_angle(l, k) / math.pi) * 180

                if -30 <= ang <= 30:  # or ang <= (-180+30) or ang >= (180-30):
                    if changed:
                        print(k)
                        possible_targets.append(k)
                    changed = False
                else:
                    changed = True

            possible_targets = list(
                filter(lambda x: x[0] <= mid_img[0] - 50, possible_targets)
            )
            possible_targets.sort(key=lambda x: math.dist(x, mid_img), reverse=False)
            target = possible_targets[0]

            """
            _, yaw = self._get_tf_error(self.pipe_carrot_frame, self.pipe_temp_frame)
            if yaw >= (20/180)*math.pi:
            """

        print(img.shape)

        line_widths = []
        for line in segments:
            w = []
            for x, y in line:
                w.append(widths[int(y), int(x)])
            line_widths.append(np.array(w))

        # TODO: this is not a great way to find width, add filter and take avg.
        width = max(line_widths[0])
        distance = self.cam.distance_from_width(0.12, width)

        fx = self.cam.calibration.K[0]
        fy = self.cam.calibration.K[4]
        cx = self.cam.calibration.K[2]
        cy = self.cam.calibration.K[5]
        u, v = target[0], target[1]

        rx = (u - cx) * distance / fx
        ry = (v - cy) * distance / fy

        _, yaw = self._get_tf_error(self.taluy_base_frame, "odom")

        pose = Pose()
        # TODO: figure out rotations based on `taluy/base_link/bottom_camera_optical_link`
        pose.position.x = rx
        pose.position.y = ry
        pose.orientation.x = 1
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 0
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = quaternion_from_euler(0, 0, yaw)
        msg = self._build_transform_message(
            self.pipe_carrot_frame,
            pose,
            frame=self.bottom_cam_frame,
        )
        self._send_transform(msg)

        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for j, seg in enumerate(segments):
            for i in range(1, len(seg)):
                cv2.line(skel_rgb, seg[i - 1], seg[i], COLORS[j % len(COLORS)], 2)
        for seg in segments:
            for i, pt in enumerate(seg):
                color = (0, 0, 255)
                if pt == target:
                    # for debugging purposes
                    color = (255, 255, 255)
                cv2.circle(skel_rgb, pt, 4, color, -1)

        img_msg = self.bridge.cv2_to_imgmsg(skel_rgb, encoding="bgr8")
        img_msg.header = msg.header
        self.pub_debug.publish(img_msg)

    def _filter_close_points(self, points, min_dist=5.0):
        if len(points) <= 2:
            return points

        filtered = [points[0]]
        for i in range(1, len(points)):
            dist = np.linalg.norm(np.array(points[i]) - np.array(filtered[-1]))

            if dist >= min_dist or i == len(points) - 1:
                filtered.append(points[i])

        return np.array(filtered)

    def _filter_points(self, points, min_dist):
        filtered = [points[0]]

        for p in points[1:-1]:
            keep = True
            for q in filtered:
                if math.dist(p, q) < min_dist:
                    keep = False
                    break
            if keep:
                filtered.append(p)

        filtered.append(points[-1])
        return filtered

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

    def _get_angle(self, p1, p2):
        dy = p1[1] - p2[1]
        dx = p1[0] - p2[0]
        return math.atan2(dy, dx)

    def _get_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _filter_segments(self, segments, min_len=50):
        segments = list(
            filter(
                lambda x: cv2.arcLength(np.array(x), closed=False) >= min_len, segments
            )
        )
        for i in range(len(segments)):
            segments[i] = self._filter_points(segments[i], 20)
        return segments

    def _build_transform_message(
        self, child_frame_id: str, pose: Pose, frame: str = "odom"
    ) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = frame
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

    def _get_tf_error(self, source_frame, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                source_frame, target_frame, rospy.Time(0), rospy.Duration(2.0)
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation
            trans_error = (trans.x, trans.y, trans.z)
            _, _, yaw_error = euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
            yaw_error = angles.normalize_angle(yaw_error)
            return trans_error, yaw_error
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            return None, None


def main():
    rospy.init_node("pipe_follower_demo")
    PipeFollowerDemo()
    rospy.spin()


if __name__ == "__main__":
    main()
