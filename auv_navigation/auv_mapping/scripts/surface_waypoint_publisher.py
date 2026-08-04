#!/usr/bin/env python3

import math
import threading

import rospy
import tf2_ros
from auv_msgs.srv import (
    SetSurfaceMission,
    SetSurfaceMissionResponse,
)
from geometry_msgs.msg import TransformStamped


EXPECTED_WAYPOINT_COUNT = 3
WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
WGS84_ECCENTRICITY_SQUARED = 6.69437999014e-3


def geodetic_delta_to_odom(
    start_latitude_deg,
    start_longitude_deg,
    target_latitude_deg,
    target_longitude_deg,
):
    """Convert a short WGS84 displacement to odom with +x north and +y west."""
    mean_latitude_rad = math.radians(
        (float(start_latitude_deg) + float(target_latitude_deg)) / 2.0
    )
    sin_latitude = math.sin(mean_latitude_rad)
    prime_vertical_scale = math.sqrt(
        1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude * sin_latitude
    )

    meridional_radius_m = (
        WGS84_SEMI_MAJOR_AXIS_M
        * (1.0 - WGS84_ECCENTRICITY_SQUARED)
        / (prime_vertical_scale**3)
    )
    prime_vertical_radius_m = WGS84_SEMI_MAJOR_AXIS_M / prime_vertical_scale

    delta_latitude_rad = math.radians(
        float(target_latitude_deg) - float(start_latitude_deg)
    )
    delta_longitude_deg = (
        float(target_longitude_deg) - float(start_longitude_deg) + 180.0
    ) % 360.0 - 180.0
    delta_longitude_rad = math.radians(delta_longitude_deg)

    north_m = delta_latitude_rad * meridional_radius_m
    east_m = delta_longitude_rad * prime_vertical_radius_m * math.cos(mean_latitude_rad)

    # ROS uses a right-handed frame. With +x pointing north and +z pointing up,
    # +y therefore points west.
    return north_m, -east_m


class SurfaceWaypointPublisher:
    def __init__(self):
        self.parent_frame = "odom"
        self.start_frame = "surface_mission_start"
        self.frame_prefix = "surface_waypoint_"
        self.target_z_m = 0.0
        self.max_waypoint_distance_m = 10000.0
        self.config_rosparam = "surface_waypoint_mission/config"

        self._lock = threading.RLock()
        self._config = None
        self._broadcaster = tf2_ros.TransformBroadcaster()
        if rospy.has_param(self.config_rosparam):
            rospy.delete_param(self.config_rosparam)

        self._set_service = rospy.Service(
            "set_surface_mission",
            SetSurfaceMission,
            self._set_surface_mission,
        )
        self._timer = rospy.Timer(rospy.Duration(0.1), self._publish_transforms)

        rospy.loginfo(
            "[SurfaceWaypointPublisher] Ready: parent=%s, set_service=%s, "
            "config_param=%s",
            self.parent_frame,
            rospy.resolve_name("set_surface_mission"),
            rospy.resolve_name(self.config_rosparam),
        )

    @staticmethod
    def _validate_coordinate(latitude_deg, longitude_deg, label):
        latitude_deg = float(latitude_deg)
        longitude_deg = float(longitude_deg)

        if not math.isfinite(latitude_deg) or not math.isfinite(longitude_deg):
            raise ValueError(f"{label}: coordinates must be finite")
        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError(f"{label}: latitude must be in [-90, 90]")
        if not -180.0 <= longitude_deg <= 180.0:
            raise ValueError(f"{label}: longitude must be in [-180, 180]")

        return latitude_deg, longitude_deg

    def _build_config(
        self,
        start_latitude_deg,
        start_longitude_deg,
        waypoint_latitudes_deg,
        waypoint_longitudes_deg,
        camera_enabled,
        visit_nearest_first,
    ):
        start_latitude_deg, start_longitude_deg = self._validate_coordinate(
            start_latitude_deg,
            start_longitude_deg,
            "Start",
        )

        waypoint_latitudes_deg = list(waypoint_latitudes_deg)
        waypoint_longitudes_deg = list(waypoint_longitudes_deg)
        camera_enabled = list(camera_enabled)
        lengths = (
            len(waypoint_latitudes_deg),
            len(waypoint_longitudes_deg),
            len(camera_enabled),
        )
        if lengths != (
            EXPECTED_WAYPOINT_COUNT,
            EXPECTED_WAYPOINT_COUNT,
            EXPECTED_WAYPOINT_COUNT,
        ):
            raise ValueError(
                "Exactly three waypoint latitudes, longitudes, and camera "
                f"selections are required; received lengths={lengths}"
            )

        waypoints = []
        coordinate_pairs = set()
        for index, (latitude_deg, longitude_deg, capture_enabled) in enumerate(
            zip(
                waypoint_latitudes_deg,
                waypoint_longitudes_deg,
                camera_enabled,
            ),
            start=1,
        ):
            latitude_deg, longitude_deg = self._validate_coordinate(
                latitude_deg,
                longitude_deg,
                f"Waypoint {index}",
            )
            coordinate_key = (round(latitude_deg, 12), round(longitude_deg, 12))
            if coordinate_key in coordinate_pairs:
                raise ValueError("Waypoint coordinates must be distinct")
            coordinate_pairs.add(coordinate_key)

            x_m, y_m = geodetic_delta_to_odom(
                start_latitude_deg,
                start_longitude_deg,
                latitude_deg,
                longitude_deg,
            )
            distance_from_start_m = math.hypot(x_m, y_m)
            if distance_from_start_m > self.max_waypoint_distance_m:
                raise ValueError(
                    f"Waypoint {index} is {distance_from_start_m:.1f} m from the "
                    f"start; the configured local-mission limit is "
                    f"{self.max_waypoint_distance_m:.1f} m"
                )
            waypoints.append(
                {
                    "index": index,
                    "frame_id": f"{self.frame_prefix}{index}",
                    "latitude_deg": latitude_deg,
                    "longitude_deg": longitude_deg,
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": self.target_z_m,
                    "distance_from_start_m": distance_from_start_m,
                    "camera_enabled": bool(capture_enabled),
                }
            )

        ordered_waypoints = self._order_waypoints(
            waypoints,
            bool(visit_nearest_first),
        )
        previous_x_m = 0.0
        previous_y_m = 0.0
        for waypoint in ordered_waypoints:
            waypoint["yaw_rad"] = math.atan2(
                waypoint["y_m"] - previous_y_m,
                waypoint["x_m"] - previous_x_m,
            )
            previous_x_m = waypoint["x_m"]
            previous_y_m = waypoint["y_m"]

        return {
            "start": {
                "latitude_deg": start_latitude_deg,
                "longitude_deg": start_longitude_deg,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": self.target_z_m,
            },
            "waypoints": ordered_waypoints,
            "visit_nearest_first": bool(visit_nearest_first),
        }

    @staticmethod
    def _order_waypoints(waypoints, visit_nearest_first):
        if not visit_nearest_first:
            return list(waypoints)

        remaining = list(waypoints)
        ordered = []
        current_x_m = 0.0
        current_y_m = 0.0
        while remaining:
            next_waypoint = min(
                remaining,
                key=lambda waypoint: (
                    math.hypot(
                        waypoint["x_m"] - current_x_m,
                        waypoint["y_m"] - current_y_m,
                    ),
                    waypoint["index"],
                ),
            )
            ordered.append(next_waypoint)
            remaining.remove(next_waypoint)
            current_x_m = next_waypoint["x_m"]
            current_y_m = next_waypoint["y_m"]
        return ordered

    def _set_surface_mission(self, request):
        try:
            with self._lock:
                config = self._build_config(
                    request.start_latitude_deg,
                    request.start_longitude_deg,
                    request.waypoint_latitudes_deg,
                    request.waypoint_longitudes_deg,
                    request.camera_enabled,
                    request.visit_nearest_first,
                )
                self._commit_config(config)
            self._log_config(config)

            response = SetSurfaceMissionResponse()
            response.success = True
            response.message = "Three surface waypoints were loaded successfully."
            response.waypoint_x_m = [wp["x_m"] for wp in config["waypoints"]]
            response.waypoint_y_m = [wp["y_m"] for wp in config["waypoints"]]
            response.waypoint_distances_m = [
                wp["distance_from_start_m"] for wp in config["waypoints"]
            ]
            response.waypoint_frame_ids = [wp["frame_id"] for wp in config["waypoints"]]
            return response
        except (TypeError, ValueError, rospy.ROSException) as exc:
            rospy.logwarn("[SurfaceWaypointPublisher] Rejected configuration: %s", exc)
            return SetSurfaceMissionResponse(success=False, message=str(exc))

    def _commit_config(self, config):
        self._config = config
        rospy.set_param(self.config_rosparam, config)

    def _log_config(self, config):
        start = config["start"]
        rospy.loginfo(
            "[SurfaceWaypointPublisher] Loaded: start=(%.8f N, %.8f E), "
            "odom convention=(+x north, +y west)",
            start["latitude_deg"],
            start["longitude_deg"],
        )
        for visit_number, waypoint in enumerate(config["waypoints"], start=1):
            rospy.loginfo(
                "[SurfaceWaypointPublisher] Visit %d: WP%d [%s], x=%.3f m north, "
                "y=%.3f m west, distance=%.3f m, yaw=%.1f deg, camera=%s",
                visit_number,
                waypoint["index"],
                waypoint["frame_id"],
                waypoint["x_m"],
                waypoint["y_m"],
                waypoint["distance_from_start_m"],
                math.degrees(waypoint["yaw_rad"]),
                "ON" if waypoint["camera_enabled"] else "OFF",
            )

    def _publish_transforms(self, _event):
        with self._lock:
            config = self._config
        if config is None:
            return

        stamp = rospy.Time.now()
        transforms = [
            self._make_transform(
                self.parent_frame,
                self.start_frame,
                0.0,
                0.0,
                self.target_z_m,
                0.0,
                stamp,
            )
        ]
        transforms.extend(
            self._make_transform(
                self.parent_frame,
                waypoint["frame_id"],
                waypoint["x_m"],
                waypoint["y_m"],
                waypoint["z_m"],
                waypoint["yaw_rad"],
                stamp,
            )
            for waypoint in config["waypoints"]
        )
        self._broadcaster.sendTransform(transforms)

    @staticmethod
    def _make_transform(parent_frame, child_frame, x_m, y_m, z_m, yaw_rad, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(x_m)
        transform.transform.translation.y = float(y_m)
        transform.transform.translation.z = float(z_m)
        transform.transform.rotation.z = math.sin(yaw_rad / 2.0)
        transform.transform.rotation.w = math.cos(yaw_rad / 2.0)
        return transform


def main():
    rospy.init_node("surface_waypoint_publisher", anonymous=False)
    SurfaceWaypointPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
