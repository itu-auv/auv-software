#!/usr/bin/env python3
"""Octagon Task 5 object manager — static spawn/despawn with fake gripper interaction.

Objects are spawned as static models (no physics). When the gripper closes,
the nearest object is despawned ("grabbed"). When the gripper opens, the held
object is re-spawned at the gripper position and animated downward toward the
nearest basket using kinematic position updates (no physics engine involved).
"""
import os
import math
import random
import threading

import rospy
import rospkg
import tf2_ros
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
import tf.transformations as tft


# ---------------------------------------------------------------------------
# Basket positions relative to robosub_octagon model origin.
# octagon world pose is configurable via params.
# ---------------------------------------------------------------------------
_BASKET_OFFSETS = {
    "redcross": (-0.02144, -0.431928, 0.56),
    "warning": (-0.02144, 0.431928, 0.56),
}

# Table surface z-offset relative to octagon origin (from octagon_table_link)
_TABLE_Z_OFFSET = 0.6245


class ModelSpawner:
    def __init__(self):
        rospy.init_node("spawn_model_server")

        # --- object key list ---
        self.keys = ["bandaid", "electric", "nutbolt", "pill"]

        # --- per-key model configuration ---
        self.configs = {}
        for key in self.keys:
            pkg = rospy.get_param(f"~{key}_model_package", "auv_sim_description")
            subdir = rospy.get_param(f"~{key}_model_subdir", f"models/{key}")
            fname = rospy.get_param(f"~{key}_model_file", "model.sdf")
            model_name = rospy.get_param(f"~{key}_model_name", f"{key}")
            service_name = rospy.get_param(
                f"~{key}_service_name", f"actuators/{key}/spawn"
            )
            self.configs[key] = {
                "model_package": pkg,
                "model_subdir": subdir,
                "model_file": fname,
                "model_name": model_name,
                "service_name": service_name,
            }

        # --- gripper / drop params ---
        self.gripper_frame = rospy.get_param("~gripper_frame", "taluy/gripper_link")
        self.robot_name = rospy.get_param("~robot_name", "taluy")
        self.base_frame = rospy.get_param("~base_frame", "taluy/base_link")
        self.drop_speed = float(rospy.get_param("~drop_speed", 0.3))  # m/s downward
        self.grab_radius = float(rospy.get_param("~grab_radius", 0.5))  # max grab dist

        # Octagon world pose (used to compute basket world positions)
        self.octagon_x = float(rospy.get_param("~octagon_x", 13.0))
        self.octagon_y = float(rospy.get_param("~octagon_y", -4.25))
        self.octagon_z = float(rospy.get_param("~octagon_z", -2.0))

        # Pre-compute table surface z in world frame
        self.table_z = self.octagon_z + _TABLE_Z_OFFSET

        # Pre-compute basket world positions
        self.baskets = {}
        for bname, (dx, dy, dz) in _BASKET_OFFSETS.items():
            self.baskets[bname] = (
                self.octagon_x + dx,
                self.octagon_y + dy,
                self.octagon_z + dz,
            )

        # --- state tracking ---
        self.spawn_counts = {}
        self.spawned_positions = []  # legacy list for min-dist check

        # Active models currently in the world: {model_name: {"key": str, "x","y","z": float}}
        self.active_models = {}
        self._active_lock = threading.Lock()

        # Currently held object (grabbed but not yet released)
        self.held_object = None  # {"key": str, "model_name": str} or None
        self._held_lock = threading.Lock()

        # Drop animation thread
        self._drop_thread = None

        # --- TF listener ---
        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)

        # --- Gazebo service proxies ---
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        rospy.wait_for_service("/gazebo/delete_model")
        self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_model_state = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )

        # --- ROS services: per-key manual spawn ---
        for key in self.keys:
            svc_name = self.configs[key]["service_name"]
            rospy.Service(
                svc_name, Trigger, lambda req, k=key: self.handle_spawn(req, k)
            )

        # --- ROS services: grab / release ---
        rospy.Service("actuators/gripper/grab", Trigger, self.handle_grab)
        rospy.Service("actuators/gripper/release", Trigger, self.handle_release)

        registered = ", ".join(
            [f"{k} -> {self.configs[k]['service_name']}" for k in self.keys]
        )
        rospy.loginfo(f"[spawn_model_server] Ready. Registered services: {registered}")
        rospy.loginfo(
            f"[spawn_model_server] Grab/release services: actuators/gripper/grab, actuators/gripper/release"
        )
        rospy.loginfo(
            f"[spawn_model_server] Baskets (world): {self.baskets}"
        )

        # Auto-spawn all 4 objects on startup
        self._auto_spawn_models()

    # -----------------------------------------------------------------------
    # Random position helpers
    # -----------------------------------------------------------------------
    def _get_random_table_pose(self, min_dist=0.15):
        """Pick a random (x, y) on the table that is at least min_dist from all
        previously spawned positions."""
        for _ in range(1000):
            x = random.uniform(12.8, 13.2)
            y = random.uniform(-4.45, -4.05)
            too_close = False
            for px, py in self.spawned_positions:
                if math.hypot(x - px, y - py) < min_dist:
                    too_close = True
                    break
            if not too_close:
                return x, y

        # Fallback
        fallback_offset = len(self.spawned_positions) * 0.05
        return 13.0 + (fallback_offset % 0.2), -4.25 + (fallback_offset % 0.2)

    def _get_unique_model_name(self, key):
        """Return a unique Gazebo model name for the given key."""
        base = self.configs[key]["model_name"]
        count = self.spawn_counts.get(key, 0) + 1
        self.spawn_counts[key] = count
        if count == 1:
            return base
        return f"{base}_{count}"

    # -----------------------------------------------------------------------
    # Auto-spawn at startup
    # -----------------------------------------------------------------------
    def _auto_spawn_models(self):
        """Automatically spawn the 4 task-5 models on the table at startup."""
        for key in self.keys:
            x, y = self._get_random_table_pose()
            self._spawn_single_model(key, x, y, self.table_z)

    # -----------------------------------------------------------------------
    # Spawn / Delete helpers
    # -----------------------------------------------------------------------
    def _read_model_xml(self, key):
        """Read the SDF XML for a given model key."""
        cfg = self.configs[key]
        rp = rospkg.RosPack()
        pkg_path = rp.get_path(cfg["model_package"])
        path = os.path.join(pkg_path, cfg["model_subdir"], cfg["model_file"])
        with open(path, "r") as f:
            return f.read()

    def _spawn_single_model(self, key, x, y, z):
        if key not in self.configs:
            rospy.logwarn(f"[spawn_model_server] Cannot spawn: '{key}' not configured")
            return False, f"'{key}' not configured"

        unique_name = self._get_unique_model_name(key)
        try:
            xml = self._read_model_xml(key)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z

            q = tft.quaternion_from_euler(0, 0, random.uniform(0, 2 * math.pi))
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

            resp = self.spawn_model(
                model_name=unique_name,
                model_xml=xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="world",
            )

            if resp.success:
                self.spawned_positions.append((x, y))
                with self._active_lock:
                    self.active_models[unique_name] = {
                        "key": key,
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                msg = (
                    f"[spawn_model_server] Spawned '{unique_name}' at "
                    f"{x:.2f}, {y:.2f}, {z:.2f}"
                )
                rospy.loginfo(msg)
                return True, msg
            else:
                msg = f"[spawn_model_server] Spawn failed: {resp.status_message}"
                rospy.logwarn(msg)
                return False, msg

        except Exception as e:
            err = f"[spawn_model_server] Error spawning {unique_name}: {e}"
            rospy.logerr(err)
            return False, err

    def _delete_gazebo_model(self, model_name):
        """Delete a model from Gazebo by name."""
        try:
            resp = self.delete_model(model_name=model_name)
            if resp.success:
                rospy.loginfo(
                    f"[spawn_model_server] Deleted model '{model_name}'"
                )
            else:
                rospy.logwarn(
                    f"[spawn_model_server] Delete failed for '{model_name}': "
                    f"{resp.status_message}"
                )
            return resp.success
        except Exception as e:
            rospy.logerr(f"[spawn_model_server] Delete error for '{model_name}': {e}")
            return False

    # -----------------------------------------------------------------------
    # Position helpers
    # -----------------------------------------------------------------------
    def _get_gripper_world_pos(self):
        """Get gripper link position in Gazebo world coordinates.

        Uses Gazebo get_model_state (immune to odom shifts) combined with
        the fixed URDF transform from base_link to gripper_link.
        """
        try:
            # 1. Get robot's actual Gazebo pose
            robot_state = self.get_model_state(self.robot_name, "world")
            if not robot_state.success:
                rospy.logwarn(
                    "[spawn_model_server] get_model_state failed for robot"
                )
                return None

            rp = robot_state.pose.position
            ro = robot_state.pose.orientation

            # 2. Get fixed TF offset: base_link -> gripper_link (from URDF, odom-independent)
            try:
                tf_offset = self.tf_buffer.lookup_transform(
                    self.base_frame, self.gripper_frame,
                    rospy.Time(0), rospy.Duration(2.0)
                )
                dx = tf_offset.transform.translation.x
                dy = tf_offset.transform.translation.y
                dz = tf_offset.transform.translation.z
            except Exception:
                # Fallback: use the known URDF offset directly
                # gripper_link is at mount_xyz="-0.101 0.0 -0.418" from base_link
                dx, dy, dz = -0.101, 0.0, -0.418

            # 3. Rotate the offset by the robot's world orientation
            q = [ro.x, ro.y, ro.z, ro.w]
            rot = tft.quaternion_matrix(q)[:3, :3]
            offset_world = rot.dot([dx, dy, dz])

            gx = rp.x + offset_world[0]
            gy = rp.y + offset_world[1]
            gz = rp.z + offset_world[2]

            rospy.logdebug(
                f"[spawn_model_server] Gripper world pos: "
                f"({gx:.3f}, {gy:.3f}, {gz:.3f})"
            )
            return (gx, gy, gz)

        except Exception as e:
            rospy.logwarn(f"[spawn_model_server] Gripper pos lookup failed: {e}")
            return None

    # -----------------------------------------------------------------------
    # Grab: find nearest active object and despawn it
    # -----------------------------------------------------------------------
    def _find_nearest_active(self, gx, gy, gz):
        """Find the active model nearest to (gx, gy, gz). Returns (name, info) or None."""
        best_name = None
        best_dist = float("inf")
        best_info = None

        with self._active_lock:
            for name, info in self.active_models.items():
                dist = math.sqrt(
                    (info["x"] - gx) ** 2
                    + (info["y"] - gy) ** 2
                    + (info["z"] - gz) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
                    best_info = info

        if best_name is None or best_dist > self.grab_radius:
            return None
        return best_name, best_info, best_dist

    def handle_grab(self, _req):
        """Service handler: grab the nearest object to the gripper."""
        with self._held_lock:
            if self.held_object is not None:
                msg = (
                    f"Already holding '{self.held_object['model_name']}', "
                    f"release first"
                )
                rospy.logwarn(f"[spawn_model_server] {msg}")
                return TriggerResponse(success=False, message=msg)

        gripper_pos = self._get_gripper_world_pos()
        if gripper_pos is None:
            return TriggerResponse(
                success=False, message="Cannot determine gripper position (TF failed)"
            )

        result = self._find_nearest_active(*gripper_pos)
        if result is None:
            return TriggerResponse(
                success=False,
                message=f"No object within grab radius ({self.grab_radius:.2f} m)",
            )

        model_name, info, dist = result

        # Delete from Gazebo
        ok = self._delete_gazebo_model(model_name)
        if not ok:
            return TriggerResponse(
                success=False, message=f"Failed to delete '{model_name}'"
            )

        # Update tracking
        with self._active_lock:
            self.active_models.pop(model_name, None)
        with self._held_lock:
            self.held_object = {"key": info["key"], "model_name": model_name}

        msg = (
            f"Grabbed '{model_name}' (key={info['key']}, dist={dist:.3f} m)"
        )
        rospy.loginfo(f"[spawn_model_server] {msg}")
        return TriggerResponse(success=True, message=msg)

    # -----------------------------------------------------------------------
    # Release: respawn at gripper, animate drop to nearest basket
    # -----------------------------------------------------------------------
    def _nearest_basket_xy(self, x, y):
        """Return the (bx, by, bz) of the basket nearest to (x, y)."""
        best = None
        best_dist = float("inf")
        for _bname, (bx, by, bz) in self.baskets.items():
            d = math.hypot(x - bx, y - by)
            if d < best_dist:
                best_dist = d
                best = (bx, by, bz)
        return best

    def handle_release(self, _req):
        """Service handler: release the held object at the gripper position."""
        with self._held_lock:
            if self.held_object is None:
                return TriggerResponse(
                    success=False, message="No object held to release"
                )
            held = self.held_object
            self.held_object = None

        gripper_pos = self._get_gripper_world_pos()
        if gripper_pos is None:
            # Fallback: respawn at table centre
            gripper_pos = (13.0, -4.25, self.table_z)
            rospy.logwarn(
                "[spawn_model_server] TF failed on release, using fallback position"
            )

        gx, gy, gz = gripper_pos

        # Spawn the object at the gripper position (static model)
        key = held["key"]
        model_name = self._get_unique_model_name(key)
        try:
            xml = self._read_model_xml(key)
            pose = Pose()
            pose.position.x = gx
            pose.position.y = gy
            pose.position.z = gz
            pose.orientation.w = 1.0

            resp = self.spawn_model(
                model_name=model_name,
                model_xml=xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="world",
            )
            if not resp.success:
                msg = f"Failed to respawn '{model_name}': {resp.status_message}"
                rospy.logwarn(f"[spawn_model_server] {msg}")
                return TriggerResponse(success=False, message=msg)

        except Exception as e:
            msg = f"Error respawning '{model_name}': {e}"
            rospy.logerr(f"[spawn_model_server] {msg}")
            return TriggerResponse(success=False, message=msg)

        # Determine target basket
        basket = self._nearest_basket_xy(gx, gy)
        if basket is None:
            target_z = self.octagon_z + 0.56  # fallback
        else:
            _bx, _by, bz = basket
            target_z = bz

        rospy.loginfo(
            f"[spawn_model_server] Released '{model_name}' at "
            f"({gx:.2f}, {gy:.2f}, {gz:.2f}), dropping to z={target_z:.2f}"
        )

        # Start drop animation in background thread
        self._drop_thread = threading.Thread(
            target=self._drop_animation,
            args=(model_name, key, gx, gy, gz, target_z),
            daemon=True,
        )
        self._drop_thread.start()

        return TriggerResponse(
            success=True,
            message=f"Released '{model_name}', animating drop to basket",
        )

    def _drop_animation(self, model_name, key, x, y, start_z, target_z):
        """Animate the model dropping straight down at constant speed.

        Uses /gazebo/set_model_state to update position each tick.
        No physics engine involvement — purely kinematic.
        """
        dt = 0.05  # 20 Hz update
        current_z = start_z

        rate = rospy.Rate(1.0 / dt)

        while not rospy.is_shutdown() and current_z > target_z:
            current_z -= self.drop_speed * dt
            if current_z < target_z:
                current_z = target_z

            state = ModelState()
            state.model_name = model_name
            state.reference_frame = "world"
            state.pose.position.x = x
            state.pose.position.y = y
            state.pose.position.z = current_z
            state.pose.orientation.w = 1.0

            try:
                self.set_model_state(state)
            except Exception as e:
                rospy.logwarn(
                    f"[spawn_model_server] Drop animation set_model_state error: {e}"
                )
                break

            rate.sleep()

        rospy.loginfo(
            f"[spawn_model_server] Drop complete for '{model_name}' at z={current_z:.3f}"
        )

        # Register as active at final position (resting in basket)
        with self._active_lock:
            self.active_models[model_name] = {
                "key": key,
                "x": x,
                "y": y,
                "z": current_z,
            }

    # -----------------------------------------------------------------------
    # Manual spawn service handler (legacy)
    # -----------------------------------------------------------------------
    def handle_spawn(self, req, key="bandaid"):
        """Handle a Trigger request and spawn the model identified by `key`."""
        if key not in self.configs:
            msg = f"Unknown model key: {key}"
            rospy.logerr(msg)
            return TriggerResponse(success=False, message=msg)

        x, y = self._get_random_table_pose()
        success, msg = self._spawn_single_model(key, x, y, self.table_z)
        return TriggerResponse(success=success, message=msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    ModelSpawner().run()
