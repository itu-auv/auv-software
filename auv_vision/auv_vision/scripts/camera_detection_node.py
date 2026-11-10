#!/usr/bin/env python3

import os
import sys
import importlib
import rclpy
from auv_common_lib.ros import get_param, service_client
from auv_common_lib import ros
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import TransformStamped
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import PropsYaw
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from auv_msgs.srv import SetDetectionFocus
import tf2_ros

# Add scripts directory to path so we can import detection_utils and handlers
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from utils.detection_utils import CameraCalibration, load_config, build_id_tf_map


DEFAULT_TRACKER_ENABLE_SERVICES = {
    "front": "/tracker_node_front/enable",
    "slalom": "/tracker_node_slalom/enable",
    "bottom": "/tracker_node_bottom/enable",
    "torpedo": "/tracker_node_torpedo/enable",
    "bottom_seg": "/tracker_node_segment/enable",
}


class CameraDetectionNode(Node):
    def __init__(self):
        super().__init__("camera_detection_pose_estimator")
        ros.set_node(self)
        self.get_logger().info("Camera detection node started")

        # Load config
        config_file = get_param(
            self,
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
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.object_publishers = {
            "object_transform": self.create_publisher(
                TransformStamped, "object_transform_updates", 10
            ),
            "props_yaw": self.create_publisher(PropsYaw, "props_yaw", 10),
        }

        # Shared state accessible by handlers
        self.shared_state = {
            "altitude": None,
            "pool_depth": get_param(self, "/env/pool_depth"),
        }

        # Camera enable flags
        self.camera_enabled = {
            "front": True,
            "front_kde": True,
            "slalom": False,
            "bottom": False,
            "torpedo": False,
            "bottom_seg": False,
        }

        tracker_enable_services = {
            cam_key: service_name
            for cam_key, service_name in DEFAULT_TRACKER_ENABLE_SERVICES.items()
            if cam_key in self.config["cameras"]
        }
        tracker_enable_services.update(get_param(self, "~tracker_enable_services", {}))
        self.tracker_enable_services = tracker_enable_services
        self.tracker_enable_proxies = {
            cam_key: service_client(self, SetBool, service_name)
            for cam_key, service_name in self.tracker_enable_services.items()
        }
        tracker_enabled_topics = {
            cam_key: self._enabled_topic_from_enable_service(service_name)
            for cam_key, service_name in self.tracker_enable_services.items()
        }
        tracker_enabled_topics.update(get_param(self, "~tracker_enabled_topics", {}))
        tracker_enabled_topics.update(
            get_param(self, "~tracker_enable_status_topics", {})
        )
        self.tracker_enabled_topics = tracker_enabled_topics
        self.tracker_enabled_status = {
            cam_key: None for cam_key in self.tracker_enable_services
        }
        self.tracker_enable_applied = {
            cam_key: None for cam_key in self.tracker_enable_services
        }
        self.tracker_enabled_subscribers = {
            cam_key: self.create_subscription(
                Bool,
                topic,
                lambda msg, k=cam_key: self._handle_tracker_enabled_status(k, msg),
                1,
            )
            for cam_key, topic in self.tracker_enabled_topics.items()
        }

        # Create handlers for each camera
        self.handlers = {}
        for cam_key, cam_cfg in self.config["cameras"].items():
            try:
                # Build camera-specific calibration
                # cam_cfg["ns"] is like "taluy/cameras/cam_front"
                # CameraCalibration expects "cameras/cam_front"
                calib_ns = cam_cfg["ns"].split("/", 1)[1]  # remove "taluy/" prefix
                calibration = CameraCalibration(calib_ns)

                # Build id_tf_map for this camera
                id_tf_map = build_id_tf_map(cam_cfg)

                # Import handler module
                handler_module = importlib.import_module(
                    f"handlers.{cam_cfg['handler']}"
                )
                handler = handler_module.create_handler(
                    cam_cfg,
                    id_tf_map,
                    self.props,
                    calibration,
                    self.tf_buffer,
                    self.object_publishers,
                    self.shared_state,
                )
                self.handlers[cam_key] = handler

                # Create subscriber
                self.create_subscription(
                    YoloResult,
                    cam_cfg["yolo_topic"],
                    lambda msg, k=cam_key: self._dispatch(msg, k),
                    1,
                )
            except Exception as e:
                self.get_logger().error(
                    f"Failed to initialize camera '{cam_key}': {e}. "
                    "Skipping this camera — other cameras will still work."
                )

        # Odometry subscriber
        self.create_subscription(Odometry, "odometry", self._odometry_callback, 10)

        # Services
        self.create_service(
            SetBool, "enable_front_camera_detections", self._handle_enable_front_camera
        )
        self.create_service(
            SetBool,
            "enable_bottom_camera_detections",
            self._handle_enable_bottom_camera,
        )
        self.create_service(
            SetBool,
            "enable_slalom_camera_detections",
            self._handle_enable_slalom_camera,
        )
        self.create_service(
            SetBool,
            "enable_torpedo_camera_detections",
            self._handle_enable_torpedo_camera,
        )
        self.create_service(
            SetDetectionFocus,
            "set_bottom_camera_focus",
            self._handle_set_bottom_camera_focus,
        )
        self.create_service(
            SetDetectionFocus,
            "set_front_camera_focus",
            self._handle_set_front_camera_focus,
        )
        self.create_service(
            SetBool,
            "enable_segment_camera_detections",
            self._handle_enable_segment_camera,
        )
        self.create_service(
            SetBool,
            "enable_front_kde_camera_detections",
            self._handle_enable_front_kde_camera,
        )
        self.create_service(
            SetBool,
            "enable_all_camera_detections",
            self._handle_enable_all_camera_detections,
        )

        self.tracker_enable_sync_timer = self.create_timer(
            1.0, self._sync_tracker_enable_states
        )

    @staticmethod
    def _enabled_topic_from_enable_service(service_name):
        service_name = service_name.rstrip("/")
        if service_name.endswith("/enable"):
            return service_name[: -len("/enable")] + "/enabled"
        if service_name.endswith("enable"):
            return service_name[: -len("enable")] + "enabled"
        return service_name + "/enabled"

    def _handle_tracker_enabled_status(self, cam_key, msg):
        self.tracker_enabled_status[cam_key] = bool(msg.data)

    def _dispatch(self, msg, cam_key):
        if not self.camera_enabled.get(cam_key, False):
            return

        camera_frame = self.config["cameras"][cam_key]["frame"]
        try:
            self.tf_buffer.lookup_transform(
                camera_frame,
                "odom",
                msg.header.stamp,
                Duration(seconds=1.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warning(
                f"Transform error: {e}", throttle_duration_sec=15.0
            )
            return

        self.handlers[cam_key].handle(msg)

    def _odometry_callback(self, msg: Odometry):
        depth = -msg.pose.pose.position.z
        self.shared_state["altitude"] = self.shared_state["pool_depth"] - depth
        self.get_logger().info(
            f"Calculated altitude from odometry Z: {self.shared_state['altitude']:.2f} m "
            f"(pool_depth={self.shared_state['pool_depth']})",
            once=True,
        )

    def _set_tracker_enabled(self, cam_key, enabled, warn=True):
        if cam_key not in self.tracker_enable_proxies:
            return True

        service_name = self.tracker_enable_services[cam_key]
        try:
            response = self.tracker_enable_proxies[cam_key](
                SetBool.Request(data=enabled)
            )
        except Exception as e:
            if warn:
                self.get_logger().warning(
                    f"Could not set YOLO tracker '{cam_key}' via {service_name}: {e}"
                )
            return False

        if not response.success:
            if warn:
                self.get_logger().warning(
                    f"YOLO tracker '{cam_key}' rejected enable={enabled}: {response.message}"
                )
            return False

        self.tracker_enable_applied[cam_key] = enabled
        return True

    def _sync_tracker_enable_states(self):
        for cam_key in self.tracker_enable_proxies:
            enabled = self.camera_enabled.get(cam_key, False)
            tracker_enabled = self.tracker_enabled_status.get(cam_key)
            if tracker_enabled == enabled:
                self.tracker_enable_applied[cam_key] = enabled
                continue
            if tracker_enabled is not None:
                self.get_logger().warning(
                    f"YOLO tracker '{cam_key}' enabled={tracker_enabled} "
                    f"differs from camera_detection enabled={enabled}; resyncing",
                    throttle_duration_sec=5.0,
                )
            elif self.tracker_enable_applied.get(cam_key) == enabled:
                continue

            self._set_tracker_enabled(cam_key, enabled, warn=False)

    def _handle_enable_camera(self, cam_key, enabled):
        self.camera_enabled[cam_key] = enabled
        tracker_updated = self._set_tracker_enabled(cam_key, enabled)
        message = f"{cam_key} camera detections " + (
            "enabled" if enabled else "disabled"
        )
        if not tracker_updated:
            message += "; YOLO tracker state was not updated"
        self.get_logger().info(message)
        return SetBool.Response(success=True, message=message)

    def _handle_enable_front_camera(self, req, response):
        return self._handle_enable_camera("front", req.data)

    def _handle_enable_bottom_camera(self, req, response):
        return self._handle_enable_camera("bottom", req.data)

    def _handle_enable_slalom_camera(self, req, response):
        return self._handle_enable_camera("slalom", req.data)

    def _handle_enable_torpedo_camera(self, req, response):
        return self._handle_enable_camera("torpedo", req.data)

    def _handle_enable_segment_camera(self, req, response):
        return self._handle_enable_camera("bottom_seg", req.data)

    def _handle_enable_front_kde_camera(self, req, response):
        return self._handle_enable_camera("front_kde", req.data)

    def _handle_enable_all_camera_detections(self, req, response):
        success = True
        messages = []
        for cam_key in list(self.camera_enabled.keys()):
            res = self._handle_enable_camera(cam_key, req.data)
            if not res.success:
                success = False
            messages.append(res.message)
        return SetBool.Response(success=success, message="; ".join(messages))

    def _handle_set_front_camera_focus(self, req, response):
        focus_objects = [
            obj.strip() for obj in req.focus_object.split(",") if obj.strip()
        ]

        if not focus_objects:
            message = f"Empty focus object provided. No changes made. Available options: {list(self.object_groups.keys())}"
            self.get_logger().warning(message)
            return SetDetectionFocus.Response(success=False, message=message)

        unfound_objects = [
            obj for obj in focus_objects if obj not in self.object_groups
        ]

        if unfound_objects:
            message = f"Unknown focus object(s): '{', '.join(unfound_objects)}'. Available options: {list(self.object_groups.keys())}"
            self.get_logger().warning(message)
            return SetDetectionFocus.Response(success=False, message=message)

        if "none" in focus_objects and len(focus_objects) > 1:
            message = "Cannot specify 'none' with other focus objects."
            self.get_logger().warning(message)
            return SetDetectionFocus.Response(success=False, message=message)

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

        self.get_logger().info(message)
        return SetDetectionFocus.Response(success=True, message=message)

    def _handle_set_bottom_camera_focus(self, req, response):
        focus_objects = [
            obj.strip() for obj in req.focus_object.split(",") if obj.strip()
        ]

        if not focus_objects:
            message = f"Empty focus object provided. No changes made. Available options: {list(self.bottom_object_groups.keys())}"
            self.get_logger().warning(message)
            return SetDetectionFocus.Response(success=False, message=message)

        unfound_objects = [
            obj for obj in focus_objects if obj not in self.bottom_object_groups
        ]

        if unfound_objects:
            message = f"Unknown focus object(s): '{', '.join(unfound_objects)}'. Available options: {list(self.bottom_object_groups.keys())}"
            self.get_logger().warning(message)
            return SetDetectionFocus.Response(success=False, message=message)

        if "none" in focus_objects and len(focus_objects) > 1:
            message = "Cannot specify 'none' with other focus objects."
            self.get_logger().warning(message)
            return SetDetectionFocus.Response(success=False, message=message)

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

        self.get_logger().info(message)
        return SetDetectionFocus.Response(success=True, message=message)


def main():
    ros.init()
    node = CameraDetectionNode()
    ros.spin(node)


if __name__ == "__main__":
    main()
