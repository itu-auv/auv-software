#!/usr/bin/env python3

import os
import time
import rospy
import numpy as np
import yaml
import threading
from datetime import datetime
from typing import Dict

from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, PoseStamped
from auv_msgs.srv import SetPremap, SetPremapResponse
import tf2_ros
import tf2_geometry_msgs


class PremapPublisher:
    def __init__(self):
        rospy.init_node("premap_publisher", anonymous=False)
        rospy.loginfo("Premap Publisher starting...")

        self.world_frame = rospy.get_param("~static_frame", "odom")
        self.rate_hz = rospy.get_param("~rate", 10.0)

        self.lock = threading.Lock()
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.premap: Dict[str, Dict] = {}
        premap_file = rospy.get_param("~premap_file", "")
        self.premap_yaml_path = premap_file
        if premap_file:
            self._load_premap(premap_file)

        self.set_premap_srv = rospy.Service(
            "set_premap", SetPremap, self.set_premap_handler
        )

        rospy.loginfo(
            f"Premap Publisher initialized. " f"premap_objects={len(self.premap)}"
        )

    def _load_premap(self, filepath: str):
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)

            if data is None:
                rospy.logwarn(f"Pre-map file is empty: {filepath}")
                return

            if "objects" not in data:
                rospy.logwarn(f"Pre-map file has no 'objects' key: {filepath}")
                return

            source_frame = data.get("reference_frame", self.world_frame)
            needs_transform = source_frame != self.world_frame
            transform_stamped = None

            if needs_transform:
                rospy.loginfo(
                    f"Premap reference_frame='{source_frame}' differs from "
                    f"world_frame='{self.world_frame}', waiting for TF..."
                )
                try:
                    transform_stamped = self.tf_buffer.lookup_transform(
                        self.world_frame,
                        source_frame,
                        rospy.Time(0),
                        rospy.Duration(10.0),
                    )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException,
                ) as e:
                    rospy.logerr(
                        f"Cannot transform premap from '{source_frame}' to "
                        f"'{self.world_frame}': {e}. Skipping premap load."
                    )
                    return

            with self.lock:
                for label, obj_data in data["objects"].items():
                    validated = self._validate_premap_object(label, obj_data)
                    if validated is None:
                        continue
                    pos, orient = validated

                    if transform_stamped:
                        p_stamped = PoseStamped()
                        p_stamped.header.frame_id = source_frame
                        p_stamped.pose.position.x = pos[0]
                        p_stamped.pose.position.y = pos[1]
                        p_stamped.pose.position.z = pos[2]
                        if len(orient) == 4:
                            p_stamped.pose.orientation.x = orient[0]
                            p_stamped.pose.orientation.y = orient[1]
                            p_stamped.pose.orientation.z = orient[2]
                            p_stamped.pose.orientation.w = orient[3]
                        else:
                            p_stamped.pose.orientation.w = 1.0

                        tf_pose = tf2_geometry_msgs.do_transform_pose(
                            p_stamped, transform_stamped
                        ).pose
                        pos = [
                            tf_pose.position.x,
                            tf_pose.position.y,
                            tf_pose.position.z,
                        ]
                        orient = [
                            tf_pose.orientation.x,
                            tf_pose.orientation.y,
                            tf_pose.orientation.z,
                            tf_pose.orientation.w,
                        ]

                    self.premap[label] = {
                        "position": np.array(pos),
                        "orientation": np.array(orient),
                    }

            frame_msg = f" (transformed from {source_frame})" if needs_transform else ""
            rospy.loginfo(
                f"Loaded pre-map with {len(self.premap)} objects from {filepath}{frame_msg}"
            )
        except Exception as e:
            rospy.logerr(f"Failed to load pre-map: {e}")

    def _validate_premap_object(self, label: str, obj_data: Dict):
        """Validate and normalize one premap object entry."""
        if not isinstance(obj_data, dict):
            rospy.logwarn(f"Skipping premap object '{label}': entry must be a dict")
            return None

        pos = obj_data.get("position", [0, 0, 0])
        orient = obj_data.get("orientation", [0, 0, 0, 1])

        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            rospy.logwarn(
                f"Skipping premap object '{label}': position must be length-3 list"
            )
            return None
        if not isinstance(orient, (list, tuple)) or len(orient) != 4:
            rospy.logwarn(
                f"Skipping premap object '{label}': orientation must be length-4 list"
            )
            return None

        try:
            pos = [float(pos[0]), float(pos[1]), float(pos[2])]
            orient = [
                float(orient[0]),
                float(orient[1]),
                float(orient[2]),
                float(orient[3]),
            ]
        except (TypeError, ValueError):
            rospy.logwarn(
                f"Skipping premap object '{label}': non-numeric position/orientation"
            )
            return None

        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(orient)):
            rospy.logwarn(
                f"Skipping premap object '{label}': non-finite position/orientation"
            )
            return None

        q_norm = np.linalg.norm(orient)
        if q_norm < 1e-8:
            rospy.logwarn(
                f"Skipping premap object '{label}': zero-norm orientation quaternion"
            )
            return None
        orient = [o / q_norm for o in orient]

        return pos, orient

    def set_premap_handler(self, req: SetPremap) -> SetPremapResponse:
        """Service handler to set pre-map."""
        try:
            target_frame = self.world_frame
            source_frame = req.reference_frame or target_frame

            new_premap_data, yaml_data_objects = (
                self._transform_and_parse_service_objects(
                    req.objects, source_frame, target_frame
                )
            )
            if new_premap_data is None:
                return SetPremapResponse(
                    success=False,
                    message=f"Failed to transform objects from {source_frame} to {target_frame}",
                )
            if not new_premap_data:
                return SetPremapResponse(
                    success=False,
                    message="No valid objects in request; pre-map unchanged",
                )

            self._atomic_update_and_save(
                new_premap_data, yaml_data_objects, target_frame
            )

            msg = f"Pre-map set with {len(self.premap)} objects: {list(self.premap.keys())}"
            rospy.loginfo(msg)
            return SetPremapResponse(success=True, message=msg)

        except Exception as e:
            msg = f"Failed to set pre-map: {e}"
            rospy.logerr(msg)
            return SetPremapResponse(success=False, message=msg)

    def _transform_and_parse_service_objects(self, objects, source_frame, target_frame):
        """Transform request objects to target frame."""
        new_premap_data = {}
        yaml_data_objects = {}
        transform_stamped = None

        if source_frame != target_frame:
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0)
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException,
            ) as e:
                rospy.logerr(f"Transform error: {e}")
                return None, None

        for obj_pose in objects:
            label = obj_pose.label
            p_stamped = PoseStamped()
            p_stamped.header.frame_id = source_frame
            p_stamped.pose = obj_pose.pose

            try:
                if transform_stamped:
                    tf_pose = tf2_geometry_msgs.do_transform_pose(
                        p_stamped, transform_stamped
                    ).pose
                else:
                    tf_pose = obj_pose.pose

                pos = tf_pose.position
                orient = tf_pose.orientation

                if label in new_premap_data:
                    rospy.logwarn(
                        f"[SetPremap] Duplicate label '{label}' in request. Overwriting."
                    )

                new_premap_data[label] = {
                    "position": np.array([pos.x, pos.y, pos.z]),
                    "orientation": np.array([orient.x, orient.y, orient.z, orient.w]),
                }

                yaml_data_objects[label] = {
                    "position": [float(pos.x), float(pos.y), float(pos.z)],
                    "orientation": [
                        float(orient.x),
                        float(orient.y),
                        float(orient.z),
                        float(orient.w),
                    ],
                }

            except Exception as e:
                rospy.logwarn(f"Failed to process object {label}: {e}")
                continue

        return new_premap_data, yaml_data_objects

    def _atomic_update_and_save(self, new_premap_data, yaml_data_objects, target_frame):
        """Update internal state and save to YAML."""
        with self.lock:
            self.premap = new_premap_data

        rospy.loginfo(
            f"Pre-map reset update complete. Loaded {len(self.premap)} objects."
        )
        if self.premap_yaml_path:
            try:
                yaml_dir = os.path.dirname(self.premap_yaml_path)
                if yaml_dir and not os.path.exists(yaml_dir):
                    os.makedirs(yaml_dir)

                if os.path.exists(self.premap_yaml_path):
                    try:
                        with open(self.premap_yaml_path, "r") as old_f:
                            old_data = yaml.safe_load(old_f)
                        old_timestamp = old_data.get("created_at", "")
                        if old_timestamp:
                            old_dt = datetime.fromisoformat(old_timestamp)
                            timestamp = old_dt.strftime("%Y%m%d_%H%M")
                        else:
                            mtime = os.path.getmtime(self.premap_yaml_path)
                            timestamp = datetime.fromtimestamp(mtime).strftime(
                                "%Y%m%d_%H%M"
                            )
                    except Exception:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

                    base, ext = os.path.splitext(self.premap_yaml_path)
                    backup_path = f"{base}_{timestamp}{ext}"
                    os.rename(self.premap_yaml_path, backup_path)
                    rospy.loginfo(f"Backed up old premap to {backup_path}")

                yaml_final_data = {
                    "reference_frame": target_frame,
                    "created_at": datetime.now().isoformat(),
                    "objects": yaml_data_objects,
                }

                with open(self.premap_yaml_path, "w") as f:
                    yaml.dump(
                        yaml_final_data, f, default_flow_style=False, sort_keys=False
                    )

                rospy.loginfo(f"Saved premap to {self.premap_yaml_path}")
            except Exception as yaml_err:
                rospy.logerr(f"Failed to save YAML: {yaml_err}")
        else:
            rospy.logwarn("No premap_file parameter set, not saving to disk")

    def broadcast_transforms(self):
        """Broadcast TF for all premap objects."""
        with self.lock:
            for label, data in self.premap.items():
                tf_msg = TransformStamped()
                tf_msg.header.stamp = rospy.Time.now()
                tf_msg.header.frame_id = self.world_frame
                # Prefix premap frames with p_ to match the original object_tracker behavior
                tf_msg.child_frame_id = f"p_{label}"

                pos = data["position"]
                tf_msg.transform.translation = Vector3(x=pos[0], y=pos[1], z=pos[2])

                q = data.get("orientation", np.array([0.0, 0.0, 0.0, 1.0]))
                tf_msg.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

                self.tf_broadcaster.sendTransform(tf_msg)

    def run(self):
        """Main loop: broadcast transforms at configured rate."""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.broadcast_transforms()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = PremapPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
