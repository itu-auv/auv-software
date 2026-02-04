#!/usr/bin/env python3

import math
import numpy as np
from collections import defaultdict
import cv2

from cv_bridge import CvBridge
import rospy
import tf2_ros
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.srv import (
    SetObjectTransform,
    SetObjectTransformRequest,
)

from skimage.morphology import skeletonize
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from auv_mapping.cfg import PipeFollowerConfig
from dynamic_reconfigure.server import Server


class PipeFramePublisher:
    def __init__(self):
        rospy.loginfo("[PipeFramePublisher] Initializing...")
        self.callback_time = rospy.Time.now()

        self.image_center = None
        self.morph_kernel_size = 20
        self.approx_poly_epsilon_factor = 0.05

        self.center_offset_threshold = 50
        self.max_width_threshold = 150

        self.close_point_filter_eps = rospy.get_param("~close_point_filter_eps", 20)
        self.short_segment_filter_eps = rospy.get_param(
            "~short_segment_filter_eps", 100
        )
        self.merge_segment_eps = rospy.get_param("~merge_segment_eps", 50)
        self.ang_error_eps = rospy.get_param("~ang_error_eps", 40)
        self.ang_error_close_point_eps = rospy.get_param(
            "~ang_error_close_point_eps", 40
        )
        # radius*2
        self.pipe_width = rospy.get_param("~pipe_width", 0.24)

        self.is_enabled = False

        self.srv = Server(PipeFollowerConfig, self.callback_reconfigure)

        self.start_service = rospy.Service("~enable", Trigger, self.cb_start)
        self.stop_service = rospy.Service("~disable", Trigger, self.cb_stop)
        rospy.loginfo("[PipeFramePublisher] Services registered.")

        self.bridge = CvBridge()
        self.cam = CameraCalibrationFetcher("cameras/cam_bottom").get_camera_info()
        self.sub_mask = rospy.Subscriber(
            "/seg_mask", Image, self.cb_mask, queue_size=1, buff_size=2**24
        )
        self.pub_debug = rospy.Publisher("/pipe_result_debug", Image, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.object_transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped
        )

        self.pipe_carrot_frame = "pipe_carrot"
        self.bottom_cam_frame = "taluy/base_link/bottom_camera_optical_link"
        self.taluy_base_frame = "taluy/base_link"

        rospy.loginfo("[PipeFramePublisher] Initialization complete.")

    def callback_reconfigure(self, config, level):
        self.close_point_filter_eps = config.close_point_filter_eps
        self.short_segment_filter_eps = config.short_segment_filter_eps
        self.merge_segment_eps = config.merge_segment_eps
        self.ang_error_eps = config.ang_error_eps
        self.ang_error_close_point_eps = config.ang_error_close_point_eps
        self.pipe_width = config.pipe_width
        return config

    def cb_start(self, req):
        self.is_enabled = True
        return TriggerResponse(success=True, message="Pipe frame publisher enabled")

    def cb_stop(self, req):
        self.is_enabled = False
        return TriggerResponse(success=True, message="Pipe frame publisher disabled")

    def cb_mask(self, msg):
        if not self.is_enabled:
            return

        self.callback_time = rospy.Time.now()
        self.image_center = [msg.width // 2, msg.height // 2]

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge err: %s", e)
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        widths = dist * 2

        # get rid of temp pixels caused by arucos
        kernel_size = self.morph_kernel_size
        opening = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8)
        )

        debug_img = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)

        skel_bool = skeletonize(opening > 0)
        skel = (skel_bool.astype(np.uint8)) * 255

        ordered_lines = self._get_ordered_points_from_skel(
            skel, self.close_point_filter_eps
        )

        # https://stackoverflow.com/questions/52782359/is-there-a-way-to-use-cv2-approxpolydp-to-approximate-open-curve
        final_points_list = []
        for line in ordered_lines:
            cnt_format = line.reshape(-1, 1, 2).astype(np.int32)
            line_len = cv2.arcLength(cnt_format, closed=False)
            approx = cv2.approxPolyDP(
                cnt_format, self.approx_poly_epsilon_factor * line_len, closed=False
            )

            pts = approx.reshape(-1, 2)
            final_points_list.append(pts)

        segments = self._merge_into_segments(final_points_list, self.merge_segment_eps)
        segments = self._filter_segments(segments, self.short_segment_filter_eps)

        # find closest segment (perpendicular distance between line and point)
        target_segment_index = 0
        min_line_dist = 1e6
        for i, seg in enumerate(segments):
            for j in range(len(seg) - 1):
                p1 = np.array(seg[j])
                p2 = np.array(seg[j + 1])
                p3 = np.array(self.image_center)
                v = p2 - p1
                w = p3 - p1

                t = np.dot(w, v) / np.dot(v, v)
                t = np.clip(t, 0, 1)

                closest = p1 + t * v
                x = np.linalg.norm(p3 - closest)
                if x <= min_line_dist:
                    min_line_dist = x
                    target_segment_index = i

        # find target point
        target_point = None
        target_point_index = None
        if len(segments) > 0:
            seg = segments[target_segment_index]
            # make first segment from left to right (according to image)
            # kinda risky but it was my only idea to decide for what direction we should move
            if seg[0][0] > seg[-1][0]:
                seg.reverse()
                segments[target_segment_index] = seg

            possible_targets = self._find_turns(seg)

            # filter targets close to image center x
            # TODO: direct distance check could be better?
            possible_targets = list(
                filter(
                    lambda x: x[1][0]
                    >= self.image_center[0] + self.center_offset_threshold,
                    possible_targets,
                )
            )

            # find point closest to center
            min_point_dist = 1e6
            closest_point_index = 0
            for i, pt in enumerate(segments[target_segment_index]):
                dist = self._get_dist(pt, self.image_center)
                if dist <= min_point_dist:
                    closest_point_index = i
                    min_point_dist = dist

            # select target point ahead of the closest point (kinda cursed)
            for idx, pt in possible_targets:
                if idx > closest_point_index:
                    target_point_index = idx
                    target_point = pt
                    break

        width = None
        if target_point is not None:
            line_widths = []
            for line in segments:
                w = []
                for x, y in line:
                    w.append(widths[int(y), int(x)])
                line_widths.append(np.array(w))

            line_widths[0] = list(
                filter(lambda x: x < self.max_width_threshold, line_widths[0])
            )
            width = max(line_widths[0])

            distance = self._distance_from_width(self.pipe_width, width)

            rx, ry = self._world_pos_from_cam(
                distance, target_point[0], target_point[1]
            )

            ang = self._get_angle(
                segments[target_segment_index][target_point_index - 1],
                segments[target_segment_index][target_point_index],
            )
            ang_err = self._normalize_angle(ang - math.pi)

            self._relocate_carrot(rx, ry, rot_offset=ang_err)

        self._publish_debug_img(
            msg, debug_img, segments, target_segment_index, target_point, width
        )

    def _find_turns(self, seg):
        # if angle changed more than self.ang_error_eps between two continous points(that are not so close), it is a turning point
        targets = []
        if len(seg) < 2:
            return targets

        p1, p2 = seg[0], seg[1]
        last_ang = self._get_angle_deg(p1, p2)

        for i in range(2, len(seg) - 1):
            k = seg[i]
            l = seg[i + 1]
            if self._get_dist(k, l) <= self.ang_error_close_point_eps:
                continue

            ang = self._get_angle_deg(k, l)
            if abs(last_ang - ang) > self.ang_error_eps:
                targets.append((i, k))
                last_ang = ang

        targets.append((len(seg) - 1, seg[-1]))
        return targets

    def _get_angle_deg(self, p1, p2):
        ang_rad = self._normalize_angle(self._get_angle(p1, p2))
        return (ang_rad / math.pi) * 180

    def _distance_from_width(self, real_width, measured_width):
        focal_length = self.cam.K[0]
        distance = (real_width * focal_length) / measured_width
        return distance

    def _world_pos_from_cam(self, distance, u, v):
        fx = self.cam.K[0]
        fy = self.cam.K[4]
        cx = self.cam.K[2]
        cy = self.cam.K[5]

        rx = (u - cx) * distance / fx
        ry = (v - cy) * distance / fy

        return rx, ry

    def _relocate_carrot(self, rx, ry, rot_offset=None):
        pose = Pose()
        pose.position.x = rx
        pose.position.y = ry
        base_rot = self._get_frame_rotation(
            self.taluy_base_frame, self.bottom_cam_frame
        )
        if rot_offset:
            q_yaw = quaternion_from_euler(0, 0, rot_offset)
            pose.orientation.x = q_yaw[0]
            pose.orientation.y = q_yaw[1]
            pose.orientation.z = q_yaw[2]
            pose.orientation.w = q_yaw[3]
        else:
            pose.orientation = base_rot

        msg = self._build_transform_message(
            self.pipe_carrot_frame,
            pose,
            frame=self.bottom_cam_frame,
        )
        self.object_transform_pub.publish(msg)

    def _publish_debug_img(
        self, msg, debug_img, segments, target_segment_index, target_point, width=None
    ):
        if self.image_center:
            cv2.circle(debug_img, tuple(self.image_center), 4, (0, 255, 0), -1)

        if width is not None:
            cv2.putText(
                debug_img,
                f"pipe width: {width}px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                1,
            )
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
                    color = (0, 255, 0)
                cv2.circle(debug_img, pt, 4, color, -1)
        img_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        img_msg.header = msg.header
        self.pub_debug.publish(img_msg)

    def _remove_close_points_global(self, lines, eps=10):
        # copy paste, idk how it works but it works as intended, kinda expensive
        cell = eps
        grid = defaultdict(list)

        def cell_key(p):
            return (int(p[0] // cell), int(p[1] // cell))

        new_lines = []

        for line in lines:
            new_line = []
            for p in line:
                key = cell_key(p)
                keep = True

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neigh = (key[0] + dx, key[1] + dy)
                        for q in grid.get(neigh, []):
                            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < eps * eps:
                                keep = False
                                break
                        if not keep:
                            break
                    if not keep:
                        break

                if keep:
                    new_line.append(p)
                    grid[key].append(p)

            if len(new_line) >= 2:
                new_lines.append(np.array(new_line))

        return new_lines

    def _get_ordered_points_from_skel(self, skeleton_img, close_point_eps=20):
        # main purpose is, putting points in right order
        # checks every pixel's neighors and creates continuous ordered connected point pieces
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
            # random pixel
            start_pixel = next(iter(pixel_set))

            path = [start_pixel]
            pixel_set.remove(start_pixel)

            # move forward
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

            # move backwards
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
        paths = self._remove_close_points_global(paths, eps=close_point_eps)
        return paths

    def _merge_into_segments(self, final_points_list, max_seg_dist):
        # connect disconnected pipe, pieces together
        # how it works:
        # let x, y be a pipe pieces disconneted because of arucos
        # if x's first element ('h') is close enough (max_seg_dist) to y's first element ('h'), reverse x and add it to y, etc etc... ('t' = tail [last point], 'h' = head [first point])
        # if nothing is near then it is an another independent segment, no need for connecting it
        segments = []

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

    def _filter_segments(self, segments, min_len):
        # ultra max pro filtering
        segments = list(
            filter(
                lambda x: cv2.arcLength(np.array(x), closed=False) >= min_len, segments
            )
        )
        return segments

    def _build_transform_message(self, child_frame_id, pose, frame="odom"):
        t = TransformStamped()
        t.header.stamp = self.callback_time
        t.header.frame_id = frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def _get_frame_rotation(self, source_frame, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            q = transform.transform.rotation
            return q
        except Exception as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            return None

    def _normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    rospy.init_node("pipe_frame_publisher")
    PipeFramePublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
