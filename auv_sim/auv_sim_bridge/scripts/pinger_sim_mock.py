#!/usr/bin/env python3

import threading
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import random
import rospy
import tf2_ros
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Transform, TransformStamped
from tf.transformations import quaternion_from_matrix, quaternion_matrix, translation_matrix

def pose_to_matrix(pose: Pose) -> np.ndarray:
    p = pose.position
    o = pose.orientation
    return translation_matrix((p.x, p.y, p.z)) @ quaternion_matrix((o.x, o.y, o.z, o.w))


def transform_to_matrix(transform: Transform) -> np.ndarray:
    t = transform.translation
    r = transform.rotation
    return translation_matrix((t.x, t.y, t.z)) @ quaternion_matrix((r.x, r.y, r.z, r.w))


def invert_rigid_transform(matrix: np.ndarray) -> np.ndarray:
    inv = np.eye(4)
    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


class GazeboModelBuffer:
    _BUFFER_SIZE = 100

    def __init__(self):
        self.model_poses: Dict[str, Pose] = {}
        self.model_matrices: Dict[str, np.ndarray] = {}
        self._pose_buffer: deque = deque(maxlen=self._BUFFER_SIZE)
        self._state_lock = threading.Lock()

        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self._model_states_cb, queue_size=1
        )

    def _model_states_cb(self, msg: ModelStates):
        names = list(msg.name) if msg.name is not None else []
        pose_list = list(msg.pose) if msg.pose is not None else []
        poses = {name: pose_list[idx] for idx, name in enumerate(names) if idx < len(pose_list)}
        matrices = {name: pose_to_matrix(pose) for name, pose in poses.items()}
        with self._state_lock:
            self._pose_buffer.append((rospy.Time.now(), poses, matrices))
            self.model_poses = poses
            self.model_matrices = matrices

    def snapshot_at_time(
        self, stamp: rospy.Time
    ) -> Tuple[Dict[str, Pose], Dict[str, np.ndarray]]:
        with self._state_lock:
            if not self._pose_buffer:
                return self.model_poses, self.model_matrices

            snapshot = min(
                tuple(self._pose_buffer),
                key=lambda entry: abs((entry[0] - stamp).to_sec()),
            )

        _, poses, matrices = snapshot
        return poses, matrices


class PingerSimMockNode:
    @staticmethod
    def _get_float_param(name: str, default: float) -> float:
        value = rospy.get_param(name, default)
        if isinstance(value, (int, float)):
            return float(value)

        rospy.logwarn("Invalid numeric param %s=%s, using %s", name, value, default)
        return default

    def __init__(self):
        rospy.init_node("pinger_sim_mock")

        self.odom_frame = str(rospy.get_param("~odom_frame", "odom"))
        self.base_frame = str(rospy.get_param("~base_frame", "taluy/base_link"))
        self.robot_name = str(rospy.get_param("~robot_name", "taluy"))
        self.pinger_frame = str(rospy.get_param("~pinger_frame", "pinger_link"))

        self.mode = str(rospy.get_param("~mode", "random")).strip().lower()
        self._random_selected_once = False
        if self.mode == "random":
            self.mode = random.choice(["octagon", "torpedo_map"])
            self._random_selected_once = True
            rospy.loginfo("Randomly selected pinger mode: %s", self.mode)

        self.publish_rate = self._get_float_param("~publish_rate", 10.0)

        self.model_octagon = str(rospy.get_param("~octagon_model", "robosub_octagon"))
        self.model_torpedo = str(rospy.get_param("~torpedo_model", "robosub_torpedo"))

        self.octagon_local_point = np.array(
            rospy.get_param("~octagon_local_point", [0.0, 0.0, 0.347]),
            dtype=float,
        )
        self.torpedo_map_local_point = np.array(
            rospy.get_param("~torpedo_map_local_point", [-0.1166, 0.6472, -0.9490]),
            dtype=float,
        )

        self.exact_model_match = bool(rospy.get_param("~exact_model_match", True))
        self.prefer_closest_to_robot = bool(
            rospy.get_param("~prefer_closest_to_robot", True)
        )
        self.debug_pose_log = bool(rospy.get_param("~debug_pose_log", False))
        self.model_instance_name = str(rospy.get_param("~model_instance_name", "")).strip()
        self.lock_target_model = bool(rospy.get_param("~lock_target_model", True))
        self._selected_model_name: Optional[str] = None

        self.gazebo = GazeboModelBuffer()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        rospy.loginfo(
            "pinger_sim_mock started mode=%s odom=%s base=%s robot=%s",
            self.mode,
            self.odom_frame,
            self.base_frame,
            self.robot_name,
        )

    def _resolve_target(self) -> Optional[Tuple[str, np.ndarray]]:
        mode_param = str(rospy.get_param("~mode", self.mode)).strip().lower()

        if mode_param == "random":
            if not self._random_selected_once:
                self.mode = random.choice(["octagon", "torpedo_map"])
                self._random_selected_once = True
                rospy.loginfo("Randomly selected pinger mode: %s", self.mode)
            mode = self.mode
        else:
            mode = mode_param
            self._random_selected_once = False

        self.mode = mode

        if mode == "octagon":
            return self.model_octagon, self.octagon_local_point
        if mode == "torpedo_map":
            return self.model_torpedo, self.torpedo_map_local_point

        rospy.logwarn_throttle(
            5.0,
            "Unsupported mode '%s' (expected: octagon | torpedo_map)",
            mode,
        )
        return None

    def _resolve_model_name(
        self,
        model_poses: Dict[str, Pose],
        model_root: str,
        robot_matrix: np.ndarray,
    ) -> Optional[str]:
        names = list(model_poses.keys())

        if self.model_instance_name:
            if self.model_instance_name in model_poses:
                return self.model_instance_name
            rospy.logwarn_throttle(
                5.0,
                "model_instance_name '%s' not found in model states",
                self.model_instance_name,
            )
            return None

        if self.lock_target_model and self._selected_model_name in model_poses:
            return self._selected_model_name

        selected: Optional[str] = None
        if self.exact_model_match:
            if model_root in model_poses:
                selected = model_root
            else:
                candidates = sorted([name for name in names if name.startswith(model_root + "_")])
                rospy.logwarn_throttle(
                    5.0,
                    "Exact model '%s' not found. Prefix candidates=%s",
                    model_root,
                    candidates,
                )
                return None
        else:
            candidates = sorted(
                [name for name in names if name == model_root or name.startswith(model_root + "_")]
            )
            if candidates:
                if len(candidates) == 1 or not self.prefer_closest_to_robot:
                    selected = candidates[0]
                else:
                    robot_pos = robot_matrix[:3, 3]

                    def _dist_sq(candidate_name: str) -> float:
                        p = model_poses[candidate_name].position
                        d = np.array([p.x, p.y, p.z], dtype=float) - robot_pos
                        return float(np.dot(d, d))

                    selected = min(candidates, key=_dist_sq)

        if selected and self.lock_target_model:
            if self._selected_model_name != selected:
                rospy.loginfo("Locked pinger target model: %s", selected)
            self._selected_model_name = selected

        return selected

    def _lookup_odom_from_world(
        self, robot_matrix: np.ndarray, stamp: rospy.Time
    ) -> Optional[np.ndarray]:
        try:
            # Same transform chain style as sim_bbox_node.py:
            # world_to_camera = base_to_camera @ inverse(robot_matrix)
            # Here we replace camera with odom.
            odom_to_base_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                stamp,
                rospy.Duration.from_sec(2.0),
            )
            odom_from_base = transform_to_matrix(odom_to_base_tf.transform)
            base_from_world = invert_rigid_transform(robot_matrix)
            return odom_from_base @ base_from_world
        except Exception as exc:
            try:
                odom_to_base_tf = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.base_frame,
                    rospy.Time(0),
                    rospy.Duration.from_sec(2.0),
                )
                odom_from_base = transform_to_matrix(odom_to_base_tf.transform)
                base_from_world = invert_rigid_transform(robot_matrix)
                return odom_from_base @ base_from_world
            except Exception:
                rospy.logwarn_throttle(
                    5.0,
                    "TF lookup failed %s <- %s at stamp/fallback: %s",
                    self.odom_frame,
                    self.base_frame,
                    str(exc),
                )
                return None

    def _build_pinger_transform(
        self,
        stamp: rospy.Time,
        odom_from_model: np.ndarray,
        local_point: np.ndarray,
    ) -> TransformStamped:
        point_h = np.array([local_point[0], local_point[1], local_point[2], 1.0])
        point_odom = odom_from_model @ point_h
        quat_odom = quaternion_from_matrix(odom_from_model)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.pinger_frame
        transform.transform.translation.x = float(point_odom[0])
        transform.transform.translation.y = float(point_odom[1])
        transform.transform.translation.z = float(point_odom[2])
        transform.transform.rotation.x = float(quat_odom[0])
        transform.transform.rotation.y = float(quat_odom[1])
        transform.transform.rotation.z = float(quat_odom[2])
        transform.transform.rotation.w = float(quat_odom[3])
        return transform

    def spin(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            target = self._resolve_target()
            if target is None:
                rate.sleep()
                continue

            model_root, local_point = target

            stamp = rospy.Time.now()
            model_poses, model_matrices = self.gazebo.snapshot_at_time(stamp)

            robot_matrix = model_matrices.get(self.robot_name)
            if robot_matrix is None:
                rospy.logwarn_throttle(
                    5.0,
                    "Robot model '%s' not found in /gazebo/model_states",
                    self.robot_name,
                )
                rate.sleep()
                continue

            model_name = self._resolve_model_name(model_poses, model_root, robot_matrix)
            if model_name is None:
                rate.sleep()
                continue

            model_matrix = model_matrices.get(model_name)
            if model_matrix is None:
                rospy.logwarn_throttle(
                    5.0,
                    "Selected model '%s' has no pose matrix",
                    model_name,
                )
                rate.sleep()
                continue

            odom_from_world = self._lookup_odom_from_world(robot_matrix, stamp)
            if odom_from_world is None:
                rate.sleep()
                continue

            odom_from_model = odom_from_world @ model_matrix
            pinger_tf = self._build_pinger_transform(stamp, odom_from_model, local_point)
            self.tf_broadcaster.sendTransform(pinger_tf)
            if self.debug_pose_log:
                rospy.loginfo_throttle(
                    1.0,
                    "pinger model=%s mode=%s odom_xyz=(%.3f, %.3f, %.3f)",
                    model_name,
                    self.mode,
                    pinger_tf.transform.translation.x,
                    pinger_tf.transform.translation.y,
                    pinger_tf.transform.translation.z,
                )
            rate.sleep()


if __name__ == "__main__":
    try:
        node = PingerSimMockNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
