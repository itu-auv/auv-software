#!/usr/bin/env python3
import math
import numpy as np
import cv2

from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import convolve


class PipeFollowerDemo():
    def __init__(self):
        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher("taluy/cmd_vel", Twist, queue_size=1)
        self.sub_mask = rospy.Subscriber(
            "yolo_fake_image", Image, self.cb_mask, queue_size=1, buff_size=2**24
        )
        self.pub_debug = rospy.Publisher("pipe_result_debug", Image, queue_size=1)
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

        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for line_pts in final_points_list:
            for i, pt in enumerate(line_pts):
                color = COLORS[i % len(COLORS)]
                cv2.circle(skel_rgb, pt, 4, color, -1)

        for line_pts in final_points_list:
            for i in range(1, len(line_pts)):
                cv2.line(skel_rgb, line_pts[i-1], line_pts[i], (0, 0, 255), 2)

        img_msg = self.bridge.cv2_to_imgmsg(skel_rgb, encoding="bgr8")
        img_msg.header = msg.header
        self.pub_debug.publish(img_msg)

        vel_msg = Twist()
        #vel_msg.linear.x = 0.1
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

        neighbors = [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        paths = []
        while pixel_set:
            start_pixel = next(iter(pixel_set))

            path = [start_pixel]
            pixel_set.remove(start_pixel)

            changed = True
            while changed:
                changed = False
                curr = path[-1]
                for (dy, dx) in neighbors:
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
                for (dy, dx) in neighbors:
                    neighbor = (curr[0] + dy, curr[1] + dx)
                    if neighbor in pixel_set:
                        path.insert(0, neighbor)
                        pixel_set.remove(neighbor)
                        changed = True
                        break

            paths.append(np.array([[p[1], p[0]] for p in path], dtype=np.float32))
        return paths


def main():
    rospy.init_node("pipe_follower_demo")
    PipeFollowerDemo()
    rospy.spin()

if __name__ == "__main__":
    main()
