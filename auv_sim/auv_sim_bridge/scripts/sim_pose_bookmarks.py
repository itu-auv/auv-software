#!/usr/bin/env python3

import os

import rospy
import yaml
from auv_msgs.srv import PoseBookmark, PoseBookmarkResponse
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from std_srvs.srv import Trigger
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class SimPoseBookmarks:
    def __init__(self):
        rospy.init_node("sim_pose_bookmarks")

        self.robot_name = rospy.get_param("~robot_name", "taluy")
        self.reference_frame = rospy.get_param("~reference_frame", "world")
        self.preset_path = rospy.get_param("~preset_path")
        self.custom_path = rospy.get_param("~custom_path")
        self.latest_id = rospy.get_param("~latest_id", "latest")
        self.clear_object_transforms_service_name = rospy.get_param(
            "~clear_object_transforms_service", "map/clear_object_transforms"
        )
        self.sync_cmd_pose_service_name = rospy.get_param(
            "~sync_cmd_pose_service", "sync_cmd_pose"
        )
        self.post_teleport_sync_delay = float(
            rospy.get_param("~post_teleport_sync_delay", 0.2)
        )

        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.get_model_state = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        self.set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        self.clear_object_transforms = rospy.ServiceProxy(
            self.clear_object_transforms_service_name, Trigger
        )
        self.sync_cmd_pose = rospy.ServiceProxy(
            self.sync_cmd_pose_service_name, Trigger
        )

        self._ensure_yaml_file(self.custom_path)

        rospy.Service(
            "simulation/teleport_pose", PoseBookmark, self.handle_teleport_pose
        )
        rospy.Service("simulation/save_pose", PoseBookmark, self.handle_save_pose)

        rospy.loginfo(
            "[sim_pose_bookmarks] Ready. robot=%s preset_path=%s custom_path=%s",
            self.robot_name,
            self.preset_path,
            self.custom_path,
        )

    def _ensure_yaml_file(self, file_path):
        directory = os.path.dirname(file_path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump({}, handle, sort_keys=False)

    def _load_yaml(self, file_path, create_if_missing=False):
        if create_if_missing:
            self._ensure_yaml_file(file_path)
        elif not os.path.exists(file_path):
            rospy.logwarn("[sim_pose_bookmarks] YAML file not found: %s", file_path)
            return {}

        with open(file_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if data is None:
            return {}

        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level mapping in {file_path}")

        return data

    def _write_yaml(self, file_path, data):
        with open(file_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)

    def _normalize_pose_id(self, pose_id):
        pose_id = (pose_id or "").strip()
        return pose_id or self.latest_id

    def _bookmark_from_pose(self, pose):
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        return {
            "pose": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
            },
            "rpy": {
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            },
        }

    def _model_state_from_bookmark(self, pose_id, bookmark):
        pose_data = bookmark.get("pose")
        rpy_data = bookmark.get("rpy")
        if not isinstance(pose_data, dict) or not isinstance(rpy_data, dict):
            raise ValueError(f"Pose '{pose_id}' must contain 'pose' and 'rpy' mappings")

        quaternion = quaternion_from_euler(
            float(rpy_data["roll"]),
            float(rpy_data["pitch"]),
            float(rpy_data["yaw"]),
        )

        state = ModelState()
        state.model_name = self.robot_name
        state.reference_frame = self.reference_frame
        state.pose.position.x = float(pose_data["x"])
        state.pose.position.y = float(pose_data["y"])
        state.pose.position.z = float(pose_data["z"])
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        return state

    def _lookup_bookmark(self, pose_id):
        resolved_pose_id = self._normalize_pose_id(pose_id)
        custom_bookmarks = self._load_yaml(self.custom_path, create_if_missing=True)
        if resolved_pose_id in custom_bookmarks:
            return resolved_pose_id, custom_bookmarks[resolved_pose_id]

        preset_bookmarks = self._load_yaml(self.preset_path)
        if resolved_pose_id in preset_bookmarks:
            return resolved_pose_id, preset_bookmarks[resolved_pose_id]

        raise KeyError(f"Pose id '{resolved_pose_id}' was not found")

    def _sync_cmd_pose_best_effort(self):
        if self.post_teleport_sync_delay > 0.0:
            rospy.sleep(self.post_teleport_sync_delay)

        try:
            rospy.wait_for_service(self.sync_cmd_pose_service_name, timeout=0.5)
            self.sync_cmd_pose()
        except Exception as exc:
            rospy.logwarn(
                "[sim_pose_bookmarks] Failed to sync cmd_pose after teleport: %s", exc
            )

    def _clear_object_transforms_best_effort(self):
        try:
            rospy.wait_for_service(
                self.clear_object_transforms_service_name, timeout=0.5
            )
            self.clear_object_transforms()
        except Exception as exc:
            rospy.logwarn(
                "[sim_pose_bookmarks] Failed to clear object transforms after teleport: %s",
                exc,
            )

    def handle_teleport_pose(self, req):
        resolved_pose_id = self._normalize_pose_id(req.pose_id)
        try:
            _, bookmark = self._lookup_bookmark(req.pose_id)
            state = self._model_state_from_bookmark(resolved_pose_id, bookmark)
            response = self.set_model_state(state)
            if not response.success:
                return PoseBookmarkResponse(
                    success=False,
                    message=response.status_message,
                    resolved_pose_id=resolved_pose_id,
                )

            self._clear_object_transforms_best_effort()
            self._sync_cmd_pose_best_effort()
            message = f"Teleported '{self.robot_name}' to pose '{resolved_pose_id}'"
            rospy.loginfo("[sim_pose_bookmarks] %s", message)
            return PoseBookmarkResponse(
                success=True,
                message=message,
                resolved_pose_id=resolved_pose_id,
            )
        except Exception as exc:
            rospy.logerr("[sim_pose_bookmarks] Teleport failed: %s", exc)
            return PoseBookmarkResponse(
                success=False,
                message=str(exc),
                resolved_pose_id=resolved_pose_id,
            )

    def handle_save_pose(self, req):
        resolved_pose_id = self._normalize_pose_id(req.pose_id)
        try:
            custom_bookmarks = self._load_yaml(self.custom_path, create_if_missing=True)
            preset_bookmarks = self._load_yaml(self.preset_path)

            if (
                resolved_pose_id != self.latest_id
                and resolved_pose_id in preset_bookmarks
            ):
                raise ValueError(
                    f"Pose id '{resolved_pose_id}' already exists in presets and is read-only"
                )

            model_state = self.get_model_state(self.robot_name, self.reference_frame)
            if not model_state.success:
                raise RuntimeError(model_state.status_message)

            bookmark = self._bookmark_from_pose(model_state.pose)
            custom_bookmarks[resolved_pose_id] = bookmark
            custom_bookmarks[self.latest_id] = bookmark
            self._write_yaml(self.custom_path, custom_bookmarks)

            message = f"Saved current pose as '{resolved_pose_id}'"
            rospy.loginfo("[sim_pose_bookmarks] %s", message)
            return PoseBookmarkResponse(
                success=True,
                message=message,
                resolved_pose_id=resolved_pose_id,
            )
        except Exception as exc:
            rospy.logerr("[sim_pose_bookmarks] Save failed: %s", exc)
            return PoseBookmarkResponse(
                success=False,
                message=str(exc),
                resolved_pose_id=resolved_pose_id,
            )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    SimPoseBookmarks().run()
