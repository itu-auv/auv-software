#!/usr/bin/env python3

import os
import sys
import importlib
import rospy
from geometry_msgs.msg import TransformStamped
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import PropsYaw
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetDetectionFocus, SetDetectionFocusResponse
import tf2_ros

# Add scripts directory to path so we can import detection_utils and handlers
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from utils.detection_utils import CameraCalibration, load_config, build_id_tf_map


class CameraDetectionNode:
    def __init__(self):
        rospy.init_node("camera_detection_pose_estimator", anonymous=True)
        rospy.loginfo("Camera detection node started")

        # Load config
        config_file = rospy.get_param(
            "~config_file",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "config",
                "detection_objects.yaml",
            ),
        )
        self.config = load_config(config_file)
        self.props = self.config["props_objects"]
        self.object_groups = self.config.get("object_groups", {})
        self.bottom_object_groups = self.config.get("bottom_object_groups", {})

        # Shared resources
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.publishers = {
            "object_transform": rospy.Publisher(
                "object_transform_updates", TransformStamped, queue_size=10
            ),
            "props_yaw": rospy.Publisher("props_yaw", PropsYaw, queue_size=10),
        }

        # Shared state accessible by handlers
        self.shared_state = {
            "altitude": None,
            "pool_depth": rospy.get_param("/env/pool_depth"),
        }

        # Camera enable flags
        self.camera_enabled = {
            "front": True,
            "bottom": False,
            "torpedo": False,
        }

        # Create handlers for each camera
        self.handlers = {}
        for cam_key, cam_cfg in self.config["cameras"].items():
            # Build camera-specific calibration
            # cam_cfg["ns"] is like "taluy/cameras/cam_front"
            # CameraCalibration expects "cameras/cam_front"
            calib_ns = cam_cfg["ns"].split("/", 1)[1]  # remove "taluy/" prefix
            calibration = CameraCalibration(calib_ns)

            # Build id_tf_map for this camera
            id_tf_map = build_id_tf_map(cam_cfg)

            # Import handler module
            handler_module = importlib.import_module(f"handlers.{cam_cfg['handler']}")
            handler = handler_module.create_handler(
                cam_cfg,
                id_tf_map,
                self.props,
                calibration,
                self.tf_buffer,
                self.publishers,
                self.shared_state,
            )
            self.handlers[cam_key] = handler

            # Create subscriber
            rospy.Subscriber(
                cam_cfg["yolo_topic"],
                YoloResult,
                lambda msg, k=cam_key: self._dispatch(msg, k),
                queue_size=1,
            )

        # Odometry subscriber
        rospy.Subscriber("odometry", Odometry, self._odometry_callback)

        # Services
        rospy.Service(
            "enable_front_camera_detections",
            SetBool,
            self._handle_enable_front_camera,
        )
        rospy.Service(
            "enable_bottom_camera_detections",
            SetBool,
            self._handle_enable_bottom_camera,
        )
        rospy.Service(
            "enable_torpedo_camera_detections",
            SetBool,
            self._handle_enable_torpedo_camera,
        )
        rospy.Service(
            "set_bottom_camera_focus",
            SetDetectionFocus,
            self._handle_set_bottom_camera_focus,
        )
        rospy.Service(
            "set_front_camera_focus",
            SetDetectionFocus,
            self._handle_set_front_camera_focus,
        )

    def _dispatch(self, msg, cam_key):
        if not self.camera_enabled.get(cam_key, False):
            return
        self.handlers[cam_key].handle(msg)

    def _odometry_callback(self, msg: Odometry):
        depth = -msg.pose.pose.position.z
        self.shared_state["altitude"] = self.shared_state["pool_depth"] - depth
        rospy.loginfo_once(
            f"Calculated altitude from odometry Z: {self.shared_state['altitude']:.2f} m "
            f"(pool_depth={self.shared_state['pool_depth']})"
        )

    def _handle_enable_front_camera(self, req):
        self.camera_enabled["front"] = req.data
        message = "Front camera detections " + ("enabled" if req.data else "disabled")
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def _handle_enable_bottom_camera(self, req):
        self.camera_enabled["bottom"] = req.data
        message = "Bottom camera detections " + ("enabled" if req.data else "disabled")
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def _handle_enable_torpedo_camera(self, req):
        self.camera_enabled["torpedo"] = req.data
        message = "Torpedo camera detections " + ("enabled" if req.data else "disabled")
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def _handle_set_front_camera_focus(self, req):
        focus_objects = [
            obj.strip() for obj in req.focus_object.split(",") if obj.strip()
        ]

        if not focus_objects:
            message = f"Empty focus object provided. No changes made. Available options: {list(self.object_groups.keys())}"
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        unfound_objects = [
            obj for obj in focus_objects if obj not in self.object_groups
        ]

        if unfound_objects:
            message = f"Unknown focus object(s): '{', '.join(unfound_objects)}'. Available options: {list(self.object_groups.keys())}"
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        if "none" in focus_objects and len(focus_objects) > 1:
            message = "Cannot specify 'none' with other focus objects."
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        all_target_ids = []
        for focus_object in focus_objects:
            all_target_ids.extend(self.object_groups[focus_object])

        target_ids = list(set(all_target_ids))

        # Forward to front camera handler
        front_handler = self.handlers.get("front")
        if front_handler and hasattr(front_handler, "set_active_ids"):
            front_handler.set_active_ids(target_ids)

        if "none" in focus_objects:
            message = "Front camera focus set to none. Detections will be ignored."
        else:
            message = f"Front camera focus set to IDs: {target_ids}"

        rospy.loginfo(message)
        return SetDetectionFocusResponse(success=True, message=message)

    def _handle_set_bottom_camera_focus(self, req):
        focus_objects = [
            obj.strip() for obj in req.focus_object.split(",") if obj.strip()
        ]

        if not focus_objects:
            message = f"Empty focus object provided. No changes made. Available options: {list(self.bottom_object_groups.keys())}"
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        unfound_objects = [
            obj for obj in focus_objects if obj not in self.bottom_object_groups
        ]

        if unfound_objects:
            message = f"Unknown focus object(s): '{', '.join(unfound_objects)}'. Available options: {list(self.bottom_object_groups.keys())}"
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        if "none" in focus_objects and len(focus_objects) > 1:
            message = "Cannot specify 'none' with other focus objects."
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        all_target_ids = []
        for focus_object in focus_objects:
            all_target_ids.extend(self.bottom_object_groups[focus_object])

        target_ids = list(set(all_target_ids))

        # Forward to bottom camera handler
        bottom_handler = self.handlers.get("bottom")
        if bottom_handler and hasattr(bottom_handler, "set_active_ids"):
            bottom_handler.set_active_ids(target_ids)

        if "none" in focus_objects:
            message = "Bottom camera focus set to none. Detections will be ignored."
        else:
            message = f"Bottom camera focus set to IDs: {target_ids}"

        rospy.loginfo(message)
        return SetDetectionFocusResponse(success=True, message=message)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CameraDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
