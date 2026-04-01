#!/usr/bin/env python3

import rospy
import math
import itertools
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as transformations
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import (
    PointStamped,
    PoseArray,
    PoseStamped,
    Pose,
    Point,
    TransformStamped,
    Transform,
    Vector3,
    Quaternion,
)
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from dataclasses import dataclass, field
from typing import List


@dataclass
class Point:
    x: float
    y: float


@dataclass
class SlalomGroup:
    left: Point
    right: Point
    mid: Point


@dataclass
class Slalom:
    groups: List[SlalomGroup] = field(default_factory=list)


class SlalomExpFramePublisher:
    def __init__(self):
        rospy.init_node("slalom_exp_frame_publisher")

        self.base_link_frame = rospy.get_param("~base_link_frame", "taluy/base_link")
        self.odom_frame = "odom"
        self.red_pipe_prefix = "slalom_red_pipe_link"
        self.white_pipe_prefix = "slalom_white_pipe_link"

        self.slalom_width = rospy.get_param("~slalom_width", 0.0254)
        self.slalom_height = rospy.get_param("~slalom_height", 0.9)
        self.ratio_threshold = rospy.get_param("~ratio_threshold", 20)
        self.ratio = self.slalom_height / self.slalom_width

        self.cam = CameraCalibrationFetcher("cameras/cam_front").get_camera_info()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

        self.srv_publish_search_points = rospy.Service(
            "slalom/publish_search_points", Trigger, self.publish_search_points_callback
        )
        self.srv_publish_waypoints = rospy.Service(
            "slalom/publish_waypoints", SetBool, self.publish_waypoints_callback
        )

    def get_all_pipes(self, base_frame="odom"):
        if not self.tf_buffer.can_transform(
            base_frame, self.red_pipe_prefix, rospy.Time(0), rospy.Duration(0.05)
        ) and self.tf_buffer.can_transform(
            base_frame, self.white_pipe_prefix, rospy.Time(0), rospy.Duration(0.05)
        ):
            return [], []

        red_pipes, white_pipes = [], []

        red_pipes.append(
            self.get_pos(
                self.tf_buffer.lookup_transform(
                    base_frame,
                    self.red_pipe_prefix,
                    rospy.Time(0),
                    rospy.Duration(1.0),
                )
            )
        )
        white_pipes.append(
            self.get_pos(
                self.tf_buffer.lookup_transform(
                    base_frame,
                    self.white_pipe_prefix,
                    rospy.Time(0),
                    rospy.Duration(1.0),
                )
            )
        )

        i = 0
        while self.tf_buffer.can_transform(
            base_frame,
            f"{self.red_pipe_prefix}_{i}",
            rospy.Time(0),
            rospy.Duration(0.05),
        ):
            red_pipes.append(
                self.get_pos(
                    self.tf_buffer.lookup_transform(
                        base_frame,
                        f"{self.red_pipe_prefix}_{i}",
                        rospy.Time(0),
                        rospy.Duration(1.0),
                    )
                )
            )
            i += 1
        i = 0
        while self.tf_buffer.can_transform(
            base_frame,
            f"{self.white_pipe_prefix}_{i}",
            rospy.Time(0),
            rospy.Duration(0.05),
        ):
            white_pipes.append(
                self.get_pos(
                    self.tf_buffer.lookup_transform(
                        base_frame,
                        f"{self.white_pipe_prefix}_{i}",
                        rospy.Time(0),
                        rospy.Duration(1.0),
                    )
                )
            )
            i += 1

        return red_pipes, white_pipes

    def publish_search_points_callback(self, req):
        red_pipes, white_pipes = self.get_all_pipes(self.odom_frame)
        if len(red_pipes) == 0 or len(white_pipes) == 0:
            return TriggerResponse(
                success=False, message="Couldn't find frame to any pipes"
            )

        robot_trans = self.tf_buffer.lookup_transform(
            self.odom_frame, self.base_link_frame, rospy.Time.now(), rospy.Duration(1.0)
        )
        robot_pos = self.get_pos(robot_trans)

        red_pipes.sort(key=lambda x: np.linalg.norm(robot_pos - x))
        closest_red = red_pipes[0]
        # assume closest red and white pipes are in the same row
        white_pipes.sort(key=lambda x: np.linalg.norm(robot_pos - x))
        closest_white = white_pipes[0]

        diff = closest_red - closest_white
        diff_unit = diff / np.linalg.norm(diff)
        perp = np.array([-diff_unit[1], diff_unit[0]])

        q = robot_trans.transform.rotation
        robot_yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        robot_fwd = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])

        if np.dot(perp, robot_fwd) < 0:
            perp = -perp

        target_pos = closest_red[:2] - (perp * 2.0)

        target_yaw = math.atan2(perp[1], perp[0])
        target_rot = transformations.quaternion_from_euler(0, 0, target_yaw)

        try:
            for c in ["start", "left", "right"]:
                t_odom = TransformStamped()
                t_odom.header.stamp = rospy.Time.now()
                t_odom.header.frame_id = self.odom_frame
                t_odom.child_frame_id = f"slalom_search_{c}"
                t_odom.transform.translation.x = target_pos[0]
                t_odom.transform.translation.y = target_pos[1]
                t_odom.transform.translation.z = 0.0
                t_odom.transform.rotation = Quaternion(*target_rot)

                if c == "left":
                    t_odom.transform.translation.x -= math.sin(target_yaw) * 1.0
                    t_odom.transform.translation.y += math.cos(target_yaw) * 1.0
                    q_c = transformations.quaternion_from_euler(
                        0, 0, target_yaw + math.radians(30)
                    )
                    t_odom.transform.rotation = Quaternion(*q_c)
                elif c == "right":
                    t_odom.transform.translation.x -= math.sin(target_yaw) * -1.0
                    t_odom.transform.translation.y += math.cos(target_yaw) * -1.0
                    q_c = transformations.quaternion_from_euler(
                        0, 0, target_yaw + math.radians(-30)
                    )
                    t_odom.transform.rotation = Quaternion(*q_c)

                req_obj = SetObjectTransformRequest()
                req_obj.transform = t_odom
                self.set_object_transform_service.call(req_obj)

            return TriggerResponse(
                success=True, message="Published all search frames (start, left, right)"
            )
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def publish_waypoints_callback(self, req):
        base_frame = "slalom_search_start"
        red_pipes, white_pipes = self.get_all_pipes(base_frame=base_frame)

        if len(red_pipes) != 3 or len(white_pipes) != 6:
            return SetBoolResponse(
                success=False,
                message=f"Need exactly 3 red and 6 white pipes, got {len(red_pipes)} red and {len(white_pipes)} white",
            )

        red_pipes.sort(key=lambda p: np.linalg.norm(p))

        groups = {}
        available_whites = list(white_pipes)

        for i, red in enumerate(red_pipes):
            available_whites.sort(key=lambda w: np.linalg.norm(red - w))
            w1 = available_whites.pop(0)
            w2 = available_whites.pop(0)

            if w1[1] > w2[1]:
                left = w1
                right = w2
            else:
                left = w2
                right = w1

            groups[i] = {"left": left, "right": right, "mid": red}

        for idx, g in groups.items():
            for w in ["left", "right"]:
                pos_wp = (g[w] + g["mid"]) / 2.0
                v_pipe = g[w] - g["mid"]
                v_pipe = v_pipe / np.linalg.norm(v_pipe)
                v_forward = np.array([-v_pipe[1], v_pipe[0]])

                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.odom_frame, base_frame, rospy.Time(0), rospy.Duration(1.0)
                    )
                except Exception as e:
                    return SetBoolResponse(
                        success=False,
                        message=f"Failed to lookup {base_frame} to odom: {e}",
                    )

                q_base = [
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w,
                ]
                matrix_base = transformations.quaternion_matrix(q_base)
                fwd_base = matrix_base[:2, 0]

                if np.dot(v_forward, fwd_base) < 0:
                    v_forward = -v_forward

                yaw = math.atan2(v_forward[1], v_forward[0])
                q_wp = transformations.quaternion_from_euler(0, 0, yaw)

                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = base_frame
                pose_stamped.pose.position.x = pos_wp[0]
                pose_stamped.pose.position.y = pos_wp[1]
                pose_stamped.pose.position.z = 0.0
                pose_stamped.pose.orientation.w = 1.0

                try:
                    odom_pose = self.tf_buffer.transform(
                        pose_stamped, self.odom_frame, rospy.Duration(1.0)
                    )
                except Exception as e:
                    return SetBoolResponse(
                        success=False, message=f"Failed to transform wp to odom: {e}"
                    )

                t_wp = TransformStamped()
                t_wp.header.stamp = rospy.Time.now()
                t_wp.header.frame_id = self.odom_frame
                t_wp.child_frame_id = f"slalom_wp_{w}_{idx}"
                t_wp.transform.translation.x = odom_pose.pose.position.x
                t_wp.transform.translation.y = odom_pose.pose.position.y
                t_wp.transform.translation.z = 0.0
                t_wp.transform.rotation = Quaternion(*q_wp)

                req_obj = SetObjectTransformRequest()
                req_obj.transform = t_wp
                self.set_object_transform_service.call(req_obj)

        return SetBoolResponse(
            success=True, message="Published waypoints based on slalom_search_start"
        )

    def get_pos(self, transform: TransformStamped):
        return np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
            ]
        )

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = SlalomExpFramePublisher()
    node.spin()
