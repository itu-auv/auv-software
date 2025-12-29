#!/usr/bin/env python3

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
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Twist
from auv_msgs.srv import (
    AlignFrameController,
    AlignFrameControllerResponse,
    SetObjectTransform,
    SetObjectTransformRequest,
)

from skimage.morphology import skeletonize
from camera_detection_pose_estimator import CameraCalibration


class PipeFollowerDemo:
    def __init__(self):
        self.count = 0
        self.rotating = False
        self.rotate_timer = 0
        self.mid_img = [320, 240]

        self.bridge = CvBridge()
        self.cam = CameraCalibration("taluy/cameras/cam_bottom")
        self.sub_mask = rospy.Subscriber(
            "yolo_fake_image", Image, self.cb_mask, queue_size=1, buff_size=2**24
        )
        self.pub_debug = rospy.Publisher("pipe_result_debug", Image, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "/taluy/map/set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.pipe_carrot_frame = "pipe_carrot"
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

        self.align_frame_cancel_service = rospy.ServiceProxy(
            "/taluy/control/align_frame/cancel", Trigger
        )
        self.align_frame_start_service = rospy.ServiceProxy(
            "/taluy/control/align_frame/start", AlignFrameController
        )
        try:
            res = self.align_frame_start_service(
                self.taluy_base_frame, self.pipe_carrot_frame, 0, False, 0.3, 0.3
            )
            print(res)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))

    def cb_mask(self, msg):
        if self.count % 20 == 0:
            pass
        else:
            self.count += 1
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge err: %s", e)
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        widths = dist * 2

        skel_bool = skeletonize(binary > 0)
        skel = (skel_bool.astype(np.uint8)) * 255
        debug_img = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        ordered_lines = self._get_ordered_points_from_skel(skel)

        final_points_list = []

        for line in ordered_lines:
            cnt_format = line.reshape(-1, 1, 2).astype(np.int32)
            line_len = cv2.arcLength(cnt_format, closed=False)
            approx = cv2.approxPolyDP(cnt_format, 0.15 * line_len, closed=False)

            pts = approx.reshape(-1, 2)
            # TODO: this is not working as expected
            aaa = self._filter_close_points(pts, 50.0)
            final_points_list.append(aaa)

        segments = self._merge_into_segments(final_points_list, 50)
        segments = self._filter_segments(segments, 100)

        min_line_dist = 1e6
        target_segment_index = 0
        for i, seg in enumerate(segments):
            for j in range(len(seg) - 1):
                p1 = np.array(seg[j])
                p2 = np.array(seg[j + 1])
                p3 = np.array(self.mid_img)
                x = abs(np.cross(p2 - p1, p1 - self.mid_img) / np.linalg.norm(p2 - p1))
                if x <= min_line_dist:
                    min_line_dist = x
                    target_segment_index = i

        target_point = None
        target_point_index = None

        if len(segments) > 0:
            seg = segments[target_segment_index]
            # make first segment from right to left (according to image)
            if seg[0][0] < seg[-1][0]:
                seg.reverse()
                segments[target_segment_index] = seg

            seg.reverse()
            changed = True
            possible_targets = []
            for i in range(len(seg) - 1):
                k = seg[i]
                l = seg[i + 1]
                # TODO: do we need to normalize_angle??
                ang = (self._get_angle(l, k) / math.pi) * 180

                if -30 <= ang <= 30:  # or ang <= (-180+30) or ang >= (180-30):
                    if changed:
                        possible_targets.append(
                            (len(seg) - 1 - i, k)
                        )  # index and value itself
                    changed = False
                else:
                    changed = True

            possible_targets = list(
                filter(lambda x: x[1][0] <= self.mid_img[0] + 50, possible_targets)
            )
            # TODO: this shouldn't be the final approach, but it works 90%
            possible_targets.sort(
                key=lambda x: abs(x[1][1] - self.mid_img[1]) * 0.8
                + abs(x[1][0] - self.mid_img[0]) * 0.2,
                reverse=False,
            )
            target_point = possible_targets[0][1] if len(possible_targets) > 0 else None
            target_point_index = (
                possible_targets[0][0] if len(possible_targets) > 0 else None
            )

        if (
            self.rotating
            and target_point != None
            and len(segments[target_segment_index]) >= target_point_index + 2
        ):
            ang_err = (
                self._get_angle(
                    self.mid_img, segments[target_segment_index][target_point_index + 1]
                )
                - math.pi / 2
            )
            ang_err_deg = (ang_err / math.pi) * 180
            if rospy.get_time() - self.rotate_timer >= 10:
                print("rotate finished")
                self.rotating = False

        if (
            not self.rotating
            and target_point is not None
            and math.dist(target_point, self.mid_img) <= 25
        ):
            rospy.loginfo("waiting for rotation")
            ang_err = self._get_angle(
                    self.mid_img, segments[target_segment_index][target_point_index + 1]
                ) - math.pi / 2
            ang_err_deg = (ang_err / math.pi) * 180
            rospy.loginfo(f"angle error: {ang_err_deg}")
            if ang_err_deg >= 180 - 20 or ang_err_deg <= -180 + 20:
                return
            try:
                tfm = self.tf_buffer.lookup_transform(
                    self.bottom_cam_frame,
                    self.pipe_carrot_frame,
                    rospy.Time(0),
                    rospy.Duration(0.5),
                )
                pos = tfm.transform.translation
                q_curr = tfm.transform.rotation
                q_curr_np = np.array([q_curr.x, q_curr.y, q_curr.z, q_curr.w])
                q_yaw = quaternion_from_euler(0, 0, ang_err)
                q_new = quaternion_multiply(q_curr_np, q_yaw)
                self.current_orientation = q_new

                pose = Pose()
                pose.position = pos
                pose.orientation.x = q_new[0]
                pose.orientation.y = q_new[1]
                pose.orientation.z = q_new[2]
                pose.orientation.w = q_new[3]

                msg = self._build_transform_message(
                    self.pipe_carrot_frame,
                    pose,
                    frame=self.bottom_cam_frame,
                )
                self._send_transform(msg)
            except Exception as e:
                rospy.logwarn(f"rotating lookup failed: {e}")
            self.rotating = True
            self.rotate_timer = rospy.get_time()
        elif not self.rotating and target_point is not None:
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
            u, v = target_point[0], target_point[1]

            rx = (u - cx) * distance / fx
            ry = (v - cy) * distance / fy

            pose = Pose()
            pose.position.x = rx
            pose.position.y = ry
            base_rot = self._get_frame_rotation(
                self.taluy_base_frame, self.bottom_cam_frame
            )
            pose.orientation = base_rot
            msg = self._build_transform_message(
                self.pipe_carrot_frame,
                pose,
                frame=self.bottom_cam_frame,
            )
            self._send_transform(msg)

        debug_img = self._build_debug_img(
            debug_img, segments, target_segment_index, target_point
        )
        img_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        img_msg.header = msg.header
        self.pub_debug.publish(img_msg)

    def _build_debug_img(self, debug_img, segments, target_segment_index, target_point):
        for j, seg in enumerate(segments):
            for i in range(1, len(seg)):
                color = (0, 0, 255)
                if j == target_segment_index:
                    color = (255, 0, 0)
                cv2.line(debug_img, seg[i - 1], seg[i], color, 2)
        for seg in segments:
            for i, pt in enumerate(seg):
                color = (0, 0, 255)
                if pt == target_point:
                    color = (255, 255, 255)
                cv2.circle(debug_img, pt, 4, color, -1)
        return debug_img

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

    def _get_frame_rotation(self, source_frame, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            q = transform.transform.rotation
            return q
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            return None

    def _normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    rospy.init_node("pipe_follower_demo")
    PipeFollowerDemo()
    rospy.spin()


if __name__ == "__main__":
    main()
