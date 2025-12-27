#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest

from dynamic_reconfigure.server import Server
from auv_mapping.cfg import GpsTargetFrameConfig


class GpsTargetFramePublisher(object):
    """
    Converts two GPS coordinates (start, target) into a Cartesian delta (meters),
    rotates by a north reference yaw, and places a TF frame at that offset
    relative to the robot's pose in odom captured at trigger time.
    """

    def __init__(self):
        rospy.loginfo("Starting gps_target_frame_publisher...")

        # Frames
        self.parent_frame = rospy.get_param("~parent_frame", "odom")
        self.robot_frame = rospy.get_param("~robot_frame", "taluy/base_link")
        self.child_frame = rospy.get_param("~child_frame", "gps_target")
        self.anchor_frame = rospy.get_param("~anchor_frame", "anchor_robot")

        # Publishing control
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 2.0))

        # North reference, rotate ENU into odom by this yaw (rad)
        self.north_reference_yaw = float(
            rospy.get_param("~north_reference_yaw_rad", 0.0)
        )

        # Dynamic reconfigure (holds start/target GPS + placeholder cartesians)
        self.cfg_server = Server(GpsTargetFrameConfig, self._reconfig_cb)
        self.start_lat = None
        self.start_lon = None
        self.target_lat = None
        self.target_lon = None
        self.start_xyz = np.zeros(3, dtype=float)  # kept but unused
        self.target_xyz = np.zeros(3, dtype=float)  # kept but unused
        self.facing_exit = False

        # TF infra
        self.tf_broadcaster = (
            tf2_ros.TransformBroadcaster()
        )  # not used directly, reserved if service wants it
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._set_object_srv = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        rospy.loginfo("Waiting for 'set_object_transform' service")
        self._set_object_srv.wait_for_service()
        rospy.loginfo("Connected to 'set_object_transform'.")
        
        self.object_non_kalman_transform_pub = rospy.Publisher(
            "object_transform_non_kalman_create", TransformStamped, queue_size=10
        ) 

        self._active = False
        self._anchor_robot_at_start = None  # latched robot pose in odom at activation
        self._srv_toggle = rospy.Service(
            "gps_target_frame_trigger", SetBool, self._toggle_cb
        )

        # Timer for publishing
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate_hz), self._publish_loop
        )

        rospy.loginfo(
            "gps_target_frame_publisher ready. Call 'gps_target_frame_trigger' to start/stop."
        )

    # -------------------- Dynamic reconfigure --------------------
    def _reconfig_cb(self, config, level):
        self.start_lat = config.starting_latitude_deg
        self.start_lon = config.starting_longitude_deg

        self.target_lat = config.target_latitude_deg
        self.target_lon = config.target_longitude_deg

        self.start_xyz = np.array(
            [config.starting_x_m, config.starting_y_m, config.starting_z_m], dtype=float
        )
        self.target_xyz = np.array(
            [config.target_x_m, config.target_y_m, config.target_z_m], dtype=float
        )
        self.facing_exit = config.facing_exit
        self.facing_distance_m = config.facing_distance_m
        return config

    # -------------------- Toggle service --------------------
    def _toggle_cb(self, req):
        self._active = req.data
        if self._active:
            # Latch current robot pose in odom as the anchor
            try:
                t = self.tf_buffer.lookup_transform(
                    self.parent_frame,
                    self.robot_frame,
                    rospy.Time(0),
                    rospy.Duration(2.0),
                )
                self._anchor_robot_at_start = np.array(
                    [
                        t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z,
                    ],
                    dtype=float,
                )
            except Exception as e:
                self._anchor_robot_at_start = None
                self._active = False  # Failed to activate, so set back to false
                return SetBoolResponse(
                    success=False,
                    message=f"TF lookup (odom->{self.robot_frame}) failed: {e}",
                )

            rospy.loginfo("gps_target_frame_publisher: ACTIVATED.")
            return SetBoolResponse(success=True, message="Publishing activated.")
        else:
            rospy.loginfo("gps_target_frame_publisher: DEACTIVATED.")
            return SetBoolResponse(success=True, message="Publishing deactivated.")

    # -------------------- Main loop --------------------
    def _publish_loop(self, _evt):
        if not self._active:
            return
        if self._anchor_robot_at_start is None:
            rospy.logwarn_throttle(
                8.0, "Anchor position not cached; call toggle service again."
            )
            return
        if self.start_lat is None or self.start_lon is None:
            rospy.logwarn_throttle(
                8.0, "starting_latitude/longitude not set via dynamic_reconfigure."
            )
            return
        if self.target_lat is None or self.target_lon is None:
            rospy.logwarn_throttle(
                8.0, "target_latitude/longitude not set via dynamic_reconfigure."
            )
            return

        # 1) GPS delta (deg) -> local ENU meters at start latitude
        east_m, north_m = self._gps_delta_to_local_m(
            self.start_lat, self.start_lon, self.target_lat, self.target_lon
        )

        # 2) Rotate ENU -> odom using north reference yaw
        odom_dx, odom_dy = self._rotate_enu_to_odom(
            east_m, north_m, self.north_reference_yaw
        )

        gps_distance = np.linalg.norm([odom_dx, odom_dy])
        if gps_distance > 0.0:
            rospy.loginfo_once(f"GPS hedefi robotun {gps_distance:.2f} metre önünde.")

        # --- New: If facing_distance_m > 0, override target position to be that far in front of robot ---
        if hasattr(self, "facing_distance_m") and self.facing_distance_m > 0.0:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.parent_frame,
                    self.robot_frame,
                    rospy.Time(0),
                    rospy.Duration(2.0),
                )
                yaw = tf.transformations.euler_from_quaternion(
                    [
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w,
                    ]
                )[2]
                dx = self.facing_distance_m * math.cos(yaw)
                dy = self.facing_distance_m * math.sin(yaw)
                target_pos = np.array(
                    [
                        t.transform.translation.x + dx,
                        t.transform.translation.y + dy,
                        t.transform.translation.z,
                    ],
                    dtype=float,
                )
                q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
            except Exception as e:
                rospy.logerr_throttle(
                    8.0, "TF lookup for facing_distance_m failed: %s", e
                )
                return
        # --- facing_exit: always use robot's current heading, but distance is either GPS or facing_distance_m ---
        if self.facing_exit:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.parent_frame,
                    self.robot_frame,
                    rospy.Time(0),
                    rospy.Duration(2.0),
                )
                yaw = tf.transformations.euler_from_quaternion(
                    [
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w,
                    ]
                )[2]
                # If facing_distance_m > 0, use it; else use GPS distance
                if hasattr(self, "facing_distance_m") and self.facing_distance_m > 0.0:
                    distance = self.facing_distance_m
                else:
                    distance = np.linalg.norm([odom_dx, odom_dy])
                dx = distance * math.cos(yaw)
                dy = distance * math.sin(yaw)
                target_pos = np.array(
                    [
                        t.transform.translation.x + dx,
                        t.transform.translation.y + dy,
                        t.transform.translation.z,
                    ],
                    dtype=float,
                )
                q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
            except Exception as e:
                rospy.logerr_throttle(8.0, "TF lookup for facing_exit failed: %s", e)
                return
        else:
            # 3) Compose final position in odom (relative to anchor)
            target_pos = self._anchor_robot_at_start + np.array(
                [odom_dx, odom_dy, 0.0], dtype=float
            )
            # 4) Orientation: x-axis points from anchor to target
            vec = np.array([odom_dx, odom_dy], dtype=float)
            yaw = math.atan2(vec[1], vec[0]) if np.linalg.norm(vec) > 1e-6 else 0.0
            q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        # 5) Build and send GPS target frame via service
        ts = self._build_transform(self.child_frame, self.parent_frame, target_pos, q)
        try:
            self.send_transform(ts)
            """
            req = SetObjectTransformRequest(transform=ts)
            resp = self._set_object_srv.call(req)
            if not resp.success:
                rospy.logerr_throttle(
                    8.0,
                    "set_object_transform failed for %s: %s",
                    ts.child_frame_id,
                    resp.message,
                )
            """
        except Exception as e:
            rospy.logerr_throttle(8.0, "set_object_transform call error: %s", e)

        # 6) Also broadcast the anchor robot frame at the anchor position
        anchor_q = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0)
        anchor_ts = self._build_transform(
            self.anchor_frame, self.parent_frame, self._anchor_robot_at_start, anchor_q
        )
        try:
            send_transform(self, anchor_ts)
            """
            anchor_req = SetObjectTransformRequest(transform=anchor_ts)
            anchor_resp = self._set_object_srv.call(anchor_req)
            if not anchor_resp.success:
                rospy.logerr_throttle(
                    8.0,
                    "set_object_transform failed for %s: %s",
                    anchor_ts.child_frame_id,
                    anchor_resp.message,
                )
            """
        except Exception as e:
            rospy.logerr_throttle(
                8.0, "set_object_transform call error for anchor: %s", e
            )

    # -------------------- Helpers --------------------
    
    def send_transform(self, transform):
        self.object_non_kalman_transform_pub.publish(transform)
    
    @staticmethod
    def _build_transform(child_frame, parent_frame, pos_xyz, q_xyzw):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(pos_xyz[0])
        t.transform.translation.y = float(pos_xyz[1])
        t.transform.translation.z = float(pos_xyz[2])
        t.transform.rotation.x = float(q_xyzw[0])
        t.transform.rotation.y = float(q_xyzw[1])
        t.transform.rotation.z = float(q_xyzw[2])
        t.transform.rotation.w = float(q_xyzw[3])
        return t

    @staticmethod
    def _meters_per_deg(lat_deg):
        lat = math.radians(lat_deg)
        m_per_deg_lat = (
            111132.92
            - 559.82 * math.cos(2 * lat)
            + 1.175 * math.cos(4 * lat)
            - 0.0023 * math.cos(6 * lat)
        )
        m_per_deg_lon = (
            111319.488 * math.cos(lat)
            - 93.5 * math.cos(3 * lat)
            + 0.118 * math.cos(5 * lat)
        )
        return m_per_deg_lat, m_per_deg_lon

    def _gps_delta_to_local_m(self, lat0_deg, lon0_deg, lat1_deg, lon1_deg):
        dlat = float(lat1_deg - lat0_deg)
        dlon = float(lon1_deg - lon0_deg)
        m_per_deg_lat, m_per_deg_lon = self._meters_per_deg(lat0_deg)
        north_m = dlat * m_per_deg_lat
        east_m = dlon * m_per_deg_lon
        return east_m, north_m

    @staticmethod
    def _rotate_enu_to_odom(east_m, north_m, yaw_rad):
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        x = c * east_m - s * north_m
        y = s * east_m + c * north_m
        return x, y


if __name__ == "__main__":
    rospy.init_node("gps_target_frame_publisher", anonymous=True)
    node = GpsTargetFramePublisher()
    rospy.spin()
