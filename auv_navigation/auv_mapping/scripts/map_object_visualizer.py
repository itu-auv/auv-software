#!/usr/bin/env python3

import re
from collections import defaultdict

import numpy as np
import rospy
import tf.transformations as tft
import tf2_ros
import yaml
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def _c(r, g, b, a):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = r, g, b, a
    return c


# Default world quaternions from pool.world spawn poses.
# Bin:     RPY(90, 0, 90) deg
# Torpedo: RPY(90, 0, -2.5) deg
# Gate:    RPY(0, 0, 196) deg - unused, two-frame rotation is preferred.
_Q_BIN = tft.quaternion_from_euler(1.5708, 0, 1.5708)
_Q_TORP = tft.quaternion_from_euler(1.5708, 0, -0.0436)
_IDENT_Q = tft.quaternion_from_euler(0, 0, 0)
_TF_STALE_TIMEOUT_SEC = 1.0
_TORPEDO_TARGET_FRAME = "torpedo_target_realsense"
_TORP_ROLL, _TORP_PITCH, _ = tft.euler_from_quaternion(_Q_TORP)
_TORPEDO_TARGET_YAW_OFFSET = np.pi


# model_key -> (mesh_uri, default_world_q, [(frame_name, color, local_offset), ...])
MESH_MODELS = {
    "robosub_gate": (
        "package://auv_sim_description/models/robosub_gate/meshes/gate.dae",
        _IDENT_Q,
        [
            (
                "gate_sawfish_link",
                _c(0.1, 0.8, 0.1, 1.0),
                np.array([0.0, -0.762, 1.356]),
            ),
            (
                "gate_shark_link",
                _c(0.1, 0.8, 0.1, 1.0),
                np.array([0.0, 0.762, 1.356]),
            ),
        ],
    ),
    "robosub_torpedo": (
        "package://auv_sim_description/models/robosub_torpedo/meshes/torpedo.dae",
        _Q_TORP,
        [
            (
                "torpedo_map_link",
                _c(0.2, 0.2, 0.8, 1.0),
                np.array([-0.1166, 0.6472, -0.9490]),
            ),
            (
                "torpedo_hole_shark_link",
                _c(0.5, 0.0, 0.8, 1.0),
                np.array([-0.1166, 0.6089, -0.9687]),
            ),
            (
                "torpedo_hole_sawfish_link",
                _c(0.5, 0.0, 0.8, 1.0),
                np.array([-0.1166, 0.8619, -1.0787]),
            ),
        ],
    ),
    "robosub_bin": (
        "package://auv_sim_description/models/robosub_bin/meshes/bin.dae",
        _Q_BIN,
        [
            (
                "bin_whole_link",
                _c(0.2, 0.6, 0.9, 1.0),
                np.array([0.0, 0.0812, 0.9337]),
            ),
        ],
    ),
    "robosub_octagon": (
        "package://auv_sim_description/models/robosub_octagon/meshes/octagon.dae",
        _IDENT_Q,
        [
            (
                "octagon_link",
                _c(0.9, 0.7, 0.1, 1.0),
                np.array([0.0, 0.0, 0.347]),
            ),
        ],
    ),
    "robosub_buoy": (
        "package://auv_sim_description/models/robosub_buoy/meshes/buoy.dae",
        _IDENT_Q,
        [("red_buoy_link", _c(1.0, 0.2, 0.2, 1.0), np.array([0.0, 0.0, 0.0]))],
    ),
    "robosub_slalom": (
        None,
        _IDENT_Q,
        [
            ("red_pipe_link", _c(0.9, 0.1, 0.1, 1.0), np.array([0.0, 0.0, 0.45])),
            (
                "white_pipe_link",
                _c(0.95, 0.95, 0.95, 1.0),
                np.array([0.0, 0.0, 0.45]),
            ),
        ],
    ),
}


# Primitive shapes for objects without a mesh.
# Matches `auv_sim_description/models/robosub_slalom/model.sdf`:
# cylinder radius=0.0127 m, length=0.9 m, centered at z=0.45 m.
FRAME_PRIMITIVES = {
    "red_pipe_link": (
        Marker.CYLINDER,
        _c(0.9, 0.1, 0.1, 0.3),
        (0.0254, 0.0254, 0.9),
    ),
    "white_pipe_link": (
        Marker.CYLINDER,
        _c(0.95, 0.95, 0.95, 0.3),
        (0.0254, 0.0254, 0.9),
    ),
}


_FRAME_TO_MODEL = {}
for _model_key, (_mesh_uri, _default_world_q, _frame_defs) in MESH_MODELS.items():
    for _frame_name, _color, _offset in _frame_defs:
        _FRAME_TO_MODEL[_frame_name] = _model_key

KNOWN_OBJECT_FRAMES = set(_FRAME_TO_MODEL) | set(FRAME_PRIMITIVES)


class MapObjectVisualizerNode:
    def __init__(self):
        rospy.init_node("map_object_visualizer")

        self.world_frame = "odom"
        self.tf_lookup_timeout = rospy.Duration(1.0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.marker_pub = rospy.Publisher(
            "map/visual_markers", MarkerArray, queue_size=1
        )

        self.published_marker_stamps = {}

        rospy.loginfo("[map_object_visualizer] ready")

    def _publish_markers(self):
        now = rospy.Time.now()
        frame_infos = self._get_tf_frame_infos()
        active_frames = self._collect_tf_frames(frame_infos, now)
        markers = []

        for child_name in sorted(active_frames):
            frame_data = active_frames[child_name]
            primitive = FRAME_PRIMITIVES.get(frame_data["base_name"])
            if primitive is None:
                continue

            shape, color, scale = primitive
            markers.append(
                self._make_primitive_marker(
                    child_name,
                    frame_data["position"],
                    frame_data["rotation"],
                    shape,
                    color,
                    scale,
                    frame_data["stamp"],
                )
            )

        markers.extend(self._make_mesh_markers(active_frames, frame_infos, now))
        self._append_delete_markers(markers, now)

        if markers:
            self.marker_pub.publish(MarkerArray(markers=markers))

    def _collect_tf_frames(self, frame_infos, now):
        active_frames = {}

        for frame_name, frame_info in frame_infos.items():
            base_name, instance_index = self._split_frame_name(frame_name)
            if base_name not in KNOWN_OBJECT_FRAMES:
                continue
            if not self._is_frame_fresh(frame_info, now):
                continue

            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.world_frame,
                    frame_name,
                    rospy.Time(0),
                    self.tf_lookup_timeout,
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                continue

            active_frames[frame_name] = {
                "base_name": base_name,
                "instance_index": instance_index,
                "stamp": tf_msg.header.stamp,
                "position": np.array(
                    [
                        tf_msg.transform.translation.x,
                        tf_msg.transform.translation.y,
                        tf_msg.transform.translation.z,
                    ],
                    dtype=float,
                ),
                "rotation": tf_msg.transform.rotation,
            }

        return active_frames

    def _get_tf_frame_infos(self):
        try:
            frames_yaml = self.tf_buffer.all_frames_as_yaml()
        except AttributeError:
            return {}
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0,
                "[map_object_visualizer] failed to query TF frames: %s",
                exc,
            )
            return {}

        if not frames_yaml:
            return {}

        try:
            frame_map = yaml.safe_load(frames_yaml) or {}
        except yaml.YAMLError as exc:
            rospy.logwarn_throttle(
                5.0,
                "[map_object_visualizer] failed to parse TF frame list: %s",
                exc,
            )
            return {}

        if not isinstance(frame_map, dict):
            return {}

        return {str(name).lstrip("/"): info for name, info in frame_map.items()}

    @staticmethod
    def _is_frame_fresh(frame_info, now):
        if not isinstance(frame_info, dict):
            return True

        most_recent = frame_info.get("most_recent_transform")
        if most_recent is None:
            return True

        try:
            return (now.to_sec() - float(most_recent)) <= _TF_STALE_TIMEOUT_SEC
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _split_frame_name(frame_name):
        normalized = frame_name.lstrip("/")
        match = re.match(r"^(.*)_(\d+)$", normalized)
        if match and match.group(1) in KNOWN_OBJECT_FRAMES:
            return match.group(1), int(match.group(2))
        return normalized, None

    @staticmethod
    def _marker_id(instance_index):
        return 0 if instance_index is None else instance_index + 1

    def _make_mesh_markers(self, active_frames, frame_infos, now):
        model_instances = defaultdict(dict)

        for frame_data in active_frames.values():
            model_key = _FRAME_TO_MODEL.get(frame_data["base_name"])
            if model_key is None:
                continue
            instance_key = (model_key, frame_data["instance_index"])
            model_instances[instance_key][frame_data["base_name"]] = frame_data

        markers = []
        for (model_key, instance_index), positions_by_frame in sorted(
            model_instances.items(), key=self._model_instance_sort_key
        ):
            mesh_uri, default_world_q, frame_defs = MESH_MODELS[model_key]
            if mesh_uri is None:
                continue

            origin, rotation_matrix, stamp = self._compute_model_pose(
                model_key,
                frame_defs,
                default_world_q,
                positions_by_frame,
                frame_infos,
                now,
            )
            if origin is None:
                continue

            markers.append(
                self._make_mesh_marker(
                    model_key,
                    mesh_uri,
                    origin,
                    rotation_matrix,
                    instance_index,
                    stamp,
                )
            )

        return markers

    @staticmethod
    def _model_instance_sort_key(item):
        model_key, instance_index = item[0]
        return (model_key, -1 if instance_index is None else instance_index)

    def _compute_model_pose(
        self,
        model_key,
        frame_defs,
        default_world_q,
        positions_by_frame,
        frame_infos,
        now,
    ):
        available = [
            (frame_name, local_offset, positions_by_frame[frame_name])
            for frame_name, _color, local_offset in frame_defs
            if frame_name in positions_by_frame
        ]
        if not available:
            return None, None, None

        default_rotation = tft.quaternion_matrix(default_world_q)[:3, :3]

        if model_key == "robosub_gate":
            return self._compute_gate_pose(available, default_rotation)

        if model_key == "robosub_torpedo":
            return self._compute_torpedo_pose(
                available, default_rotation, frame_infos, now
            )

        origins = [
            frame_data["position"] - default_rotation.dot(local_offset)
            for _, local_offset, frame_data in available
        ]
        stamp = self._combine_stamps(
            frame_data["stamp"] for _, _, frame_data in available
        )
        return np.mean(origins, axis=0), default_rotation, stamp

    def _compute_gate_pose(self, available, default_rotation):
        if len(available) < 2:
            _frame_name, local_offset, frame_data = available[0]
            return (
                frame_data["position"] - default_rotation.dot(local_offset),
                default_rotation,
                frame_data["stamp"],
            )

        (_, offset_a, frame_data_a), (_, offset_b, frame_data_b) = available[:2]
        rotation = self._rotation_between(
            offset_b - offset_a,
            frame_data_b["position"] - frame_data_a["position"],
        )
        if rotation is None:
            rotation = default_rotation

        origins = [
            frame_data["position"] - rotation.dot(local_offset)
            for _, local_offset, frame_data in available
        ]
        stamp = self._combine_stamps(
            frame_data["stamp"] for _, _, frame_data in available
        )
        return np.mean(origins, axis=0), rotation, stamp

    def _compute_torpedo_pose(self, available, default_rotation, frame_infos, now):
        rotation, rotation_stamp = self._lookup_torpedo_rotation(frame_infos, now)
        if rotation is None:
            rotation = default_rotation
            rotation_stamp = None

        anchor = next(
            (item for item in available if item[0] == "torpedo_map_link"), available[0]
        )
        _frame_name, local_offset, frame_data = anchor
        stamp = self._combine_stamps([frame_data["stamp"], rotation_stamp])
        return frame_data["position"] - rotation.dot(local_offset), rotation, stamp

    def _lookup_torpedo_rotation(self, frame_infos, now):
        yaw, stamp = self._lookup_frame_yaw(_TORPEDO_TARGET_FRAME, frame_infos, now)
        if yaw is None:
            return None, None

        torpedo_quat = tft.quaternion_from_euler(
            _TORP_ROLL,
            _TORP_PITCH,
            yaw + _TORPEDO_TARGET_YAW_OFFSET,
        )
        return tft.quaternion_matrix(torpedo_quat)[:3, :3], stamp

    def _lookup_frame_yaw(self, frame_name, frame_infos, now):
        frame_info = frame_infos.get(frame_name)
        if frame_info is None or not self._is_frame_fresh(frame_info, now):
            return None, None

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.world_frame,
                frame_name,
                rospy.Time(0),
                self.tf_lookup_timeout,
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None, None

        quat = (
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z,
            tf_msg.transform.rotation.w,
        )
        _, _, yaw = tft.euler_from_quaternion(quat)
        return yaw, tf_msg.header.stamp

    @staticmethod
    def _combine_stamps(stamps):
        valid_stamps = []
        for stamp in stamps:
            if stamp is None or stamp == rospy.Time():
                continue
            valid_stamps.append(stamp)

        if not valid_stamps:
            return rospy.Time()

        return max(valid_stamps, key=lambda stamp: (stamp.secs, stamp.nsecs))

    @staticmethod
    def _rotation_between(v_from, v_to):
        a = np.array(v_from, dtype=float)
        b = np.array(v_to, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return None

        a_hat = a / norm_a
        b_hat = b / norm_b
        cross = np.cross(a_hat, b_hat)
        cross_norm_sq = float(np.dot(cross, cross))
        dot = float(np.clip(np.dot(a_hat, b_hat), -1.0, 1.0))

        if cross_norm_sq < 1e-12:
            if dot > 0.0:
                return np.eye(3)

            perp = np.array([1.0, 0.0, 0.0])
            if abs(a_hat[0]) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            axis = np.cross(a_hat, perp)
            axis = axis / np.linalg.norm(axis)
            return tft.rotation_matrix(np.pi, axis)[:3, :3]

        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ]
        )
        return np.eye(3) + skew + skew.dot(skew) * ((1.0 - dot) / cross_norm_sq)

    def _make_mesh_marker(
        self, model_key, mesh_uri, origin, rotation_matrix, instance_index, stamp
    ):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.world_frame
        marker.ns = "mesh_" + model_key
        marker.id = self._marker_id(instance_index)
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.mesh_resource = mesh_uri
        marker.mesh_use_embedded_materials = False
        marker.pose.position.x = origin[0]
        marker.pose.position.y = origin[1]
        marker.pose.position.z = origin[2]

        quat_matrix = np.eye(4)
        quat_matrix[:3, :3] = rotation_matrix
        qx, qy, qz, qw = tft.quaternion_from_matrix(quat_matrix)
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color = _c(0.6, 0.6, 0.6, 0.3)
        return marker

    def _make_primitive_marker(
        self, child_name, position, rotation, shape, color, scale, stamp
    ):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.world_frame
        marker.ns = "detection_" + child_name
        marker.id = 0
        marker.type = shape
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation = rotation
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color = color
        return marker

    def _append_delete_markers(self, markers, now):
        current_marker_stamps = {
            (marker.ns, marker.id): marker.header.stamp
            for marker in markers
            if marker.action == Marker.ADD
        }

        stale_marker_keys = set(self.published_marker_stamps) - set(
            current_marker_stamps
        )
        for namespace, marker_id in sorted(stale_marker_keys):
            delete_marker = Marker()
            delete_marker.header.stamp = self.published_marker_stamps.get(
                (namespace, marker_id), now
            )
            delete_marker.header.frame_id = self.world_frame
            delete_marker.ns = namespace
            delete_marker.id = marker_id
            delete_marker.action = Marker.DELETE
            markers.append(delete_marker)

        self.published_marker_stamps = current_marker_stamps

    def run(self):
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            self._publish_markers()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = MapObjectVisualizerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
