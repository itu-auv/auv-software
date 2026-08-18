#!/usr/bin/env python3

import math
import threading

import rospy
import tf2_ros
from auv_msgs.srv import SetBuoyWaypoints, SetBuoyWaypointsResponse
from geometry_msgs.msg import TransformStamped


WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
WGS84_ECCENTRICITY_SQUARED = 6.69437999014e-3


def geodetic_delta_to_odom(
    origin_latitude_deg,
    origin_longitude_deg,
    target_latitude_deg,
    target_longitude_deg,
):
    """Return a local WGS84 displacement with +x north and +y west."""
    mean_latitude_rad = math.radians(
        (float(origin_latitude_deg) + float(target_latitude_deg)) / 2.0
    )
    sin_latitude = math.sin(mean_latitude_rad)
    scale = math.sqrt(1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude * sin_latitude)
    meridional_radius_m = (
        WGS84_SEMI_MAJOR_AXIS_M * (1.0 - WGS84_ECCENTRICITY_SQUARED) / (scale**3)
    )
    prime_vertical_radius_m = WGS84_SEMI_MAJOR_AXIS_M / scale

    delta_latitude_rad = math.radians(
        float(target_latitude_deg) - float(origin_latitude_deg)
    )
    delta_longitude_deg = (
        float(target_longitude_deg) - float(origin_longitude_deg) + 180.0
    ) % 360.0 - 180.0
    delta_longitude_rad = math.radians(delta_longitude_deg)

    north_m = delta_latitude_rad * meridional_radius_m
    east_m = delta_longitude_rad * prime_vertical_radius_m * math.cos(mean_latitude_rad)
    return north_m, -east_m


class BuoyWaypointPublisher:
    """Validate one buoy/surface configuration and continuously publish its TFs."""

    def __init__(self):
        self.parent_frame = rospy.get_param("~parent_frame", "odom")
        self.north_frame = rospy.get_param("~north_frame", "north")
        self.buoy_frame = rospy.get_param("~buoy_frame", "buoy")
        self.surface_frame = rospy.get_param("~surface_frame", "surface")
        self.target_z_m = float(rospy.get_param("~target_z_m", 0.0))
        self.max_local_distance_m = float(
            rospy.get_param("~max_local_distance_m", 10000.0)
        )
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))
        self.config_rosparam = rospy.get_param(
            "~config_rosparam", "buoy_waypoint_mission/config"
        )
        if self.publish_rate_hz <= 0.0:
            raise ValueError("~publish_rate_hz must be positive")
        if self.max_local_distance_m <= 0.0:
            raise ValueError("~max_local_distance_m must be positive")

        self._lock = threading.RLock()
        self._config = None
        self._broadcaster = tf2_ros.TransformBroadcaster()
        if rospy.has_param(self.config_rosparam):
            rospy.delete_param(self.config_rosparam)

        self._set_service = rospy.Service(
            "set_buoy_waypoints",
            SetBuoyWaypoints,
            self._set_waypoints,
        )
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate_hz),
            self._publish_transforms,
        )
        rospy.loginfo(
            "[BuoyWaypoints] Ready: service=%s, frames=%s,%s, parent=%s",
            rospy.resolve_name("set_buoy_waypoints"),
            self.buoy_frame,
            self.surface_frame,
            self.parent_frame,
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

    @staticmethod
    def _validate_xy(x_m, y_m, label):
        x_m = float(x_m)
        y_m = float(y_m)
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError(f"{label}: metre coordinates must be finite")
        return x_m, y_m

    def _build_config(self, request):
        start_latitude_deg, start_longitude_deg = self._validate_coordinate(
            request.start_latitude_deg,
            request.start_longitude_deg,
            "Start",
        )
        odom_heading_from_north_deg = float(request.odom_heading_from_north_deg)
        if not math.isfinite(odom_heading_from_north_deg):
            raise ValueError("Odom heading must be finite")
        north_yaw_rad = math.radians(odom_heading_from_north_deg)

        if request.use_geodetic:
            buoy_latitude_deg, buoy_longitude_deg = self._validate_coordinate(
                request.buoy_latitude_deg,
                request.buoy_longitude_deg,
                "Buoy",
            )
            surface_latitude_deg, surface_longitude_deg = self._validate_coordinate(
                request.surface_latitude_deg,
                request.surface_longitude_deg,
                "Surface",
            )
            if (
                round(buoy_latitude_deg, 12),
                round(buoy_longitude_deg, 12),
            ) == (
                round(surface_latitude_deg, 12),
                round(surface_longitude_deg, 12),
            ):
                raise ValueError("Buoy and surface coordinates must be distinct")

            buoy_x_m, buoy_y_m = geodetic_delta_to_odom(
                start_latitude_deg,
                start_longitude_deg,
                buoy_latitude_deg,
                buoy_longitude_deg,
            )
            surface_x_m, surface_y_m = geodetic_delta_to_odom(
                start_latitude_deg,
                start_longitude_deg,
                surface_latitude_deg,
                surface_longitude_deg,
            )
            geodetic = {
                "buoy_latitude_deg": buoy_latitude_deg,
                "buoy_longitude_deg": buoy_longitude_deg,
                "surface_latitude_deg": surface_latitude_deg,
                "surface_longitude_deg": surface_longitude_deg,
            }
        else:
            buoy_x_m, buoy_y_m = self._validate_xy(
                request.buoy_x_m, request.buoy_y_m, "Buoy"
            )
            surface_x_m, surface_y_m = self._validate_xy(
                request.surface_x_m, request.surface_y_m, "Surface"
            )
            # ROS parameters are transported over XML-RPC, which cannot marshal
            # Python None unless allow_none is enabled. Keep the schema stable
            # and use an empty mapping when metre debug mode has no GPS data.
            geodetic = {}

        if math.hypot(surface_x_m - buoy_x_m, surface_y_m - buoy_y_m) <= 1e-6:
            raise ValueError("Buoy and surface positions must be distinct")

        buoy_distance_m = math.hypot(buoy_x_m, buoy_y_m)
        surface_distance_m = math.hypot(surface_x_m, surface_y_m)
        buoy_to_surface_yaw_rad = math.atan2(
            surface_y_m - buoy_y_m,
            surface_x_m - buoy_x_m,
        )
        for label, distance_m in (
            ("Buoy", buoy_distance_m),
            ("Surface", surface_distance_m),
        ):
            if distance_m > self.max_local_distance_m:
                raise ValueError(
                    f"{label} is {distance_m:.1f} m from the start; "
                    f"limit is {self.max_local_distance_m:.1f} m"
                )

        points = [
            self._point_config(self.buoy_frame, buoy_x_m, buoy_y_m, buoy_distance_m),
            self._point_config(
                self.surface_frame,
                surface_x_m,
                surface_y_m,
                surface_distance_m,
                yaw_rad=buoy_to_surface_yaw_rad,
            ),
        ]
        return {
            "mode": "geodetic" if request.use_geodetic else "metres",
            "axis_convention": "+x north, +y west, +z up",
            "start": {
                "latitude_deg": start_latitude_deg,
                "longitude_deg": start_longitude_deg,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": self.target_z_m,
            },
            "geodetic": geodetic,
            "odom_heading_from_north_deg": odom_heading_from_north_deg,
            "north_yaw_rad": north_yaw_rad,
            "points": points,
        }

    def _point_config(
        self,
        frame_id,
        x_m,
        y_m,
        distance_from_start_m,
        yaw_rad=0.0,
    ):
        return {
            "frame_id": frame_id,
            "x_m": float(x_m),
            "y_m": float(y_m),
            "z_m": self.target_z_m,
            "yaw_rad": float(yaw_rad),
            "distance_from_start_m": float(distance_from_start_m),
        }

    def _set_waypoints(self, request):
        try:
            config = self._build_config(request)
            with self._lock:
                rospy.set_param(self.config_rosparam, config)
                self._config = config
            self._log_config(config)
            response = SetBuoyWaypointsResponse()
            response.success = True
            response.message = "Buoy and surface frames loaded successfully."
            response.frame_ids = [point["frame_id"] for point in config["points"]]
            response.x_m = [point["x_m"] for point in config["points"]]
            response.y_m = [point["y_m"] for point in config["points"]]
            response.distances_from_start_m = [
                point["distance_from_start_m"] for point in config["points"]
            ]
            return response
        except (TypeError, ValueError, rospy.ROSException) as exc:
            rospy.logwarn("[BuoyWaypoints] Rejected configuration: %s", exc)
            return SetBuoyWaypointsResponse(success=False, message=str(exc))

    @staticmethod
    def _log_config(config):
        rospy.loginfo("[BuoyWaypoints] Loaded in %s mode", config["mode"])
        rospy.loginfo(
            "[BuoyWaypoints] Odom heading from north=%.1f deg",
            config["odom_heading_from_north_deg"],
        )
        for point in config["points"]:
            rospy.loginfo(
                "[BuoyWaypoints] %s: x=%.3f m north, y=%.3f m west, "
                "yaw=%.3f rad, distance_from_start=%.3f m",
                point["frame_id"],
                point["x_m"],
                point["y_m"],
                point["yaw_rad"],
                point["distance_from_start_m"],
            )

    def _publish_transforms(self, _event):
        with self._lock:
            config = self._config
        if config is None:
            return

        stamp = rospy.Time.now()
        transforms = [self.build_north_transform(config["north_yaw_rad"], stamp)]
        transforms.extend(
            self.build_transform_message(point, stamp) for point in config["points"]
        )
        self.send_transforms(transforms)

    def build_north_transform(self, yaw_rad, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.north_frame
        transform.transform.rotation.z = math.sin(yaw_rad / 2.0)
        transform.transform.rotation.w = math.cos(yaw_rad / 2.0)
        return transform

    def build_transform_message(self, point, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.north_frame
        transform.child_frame_id = point["frame_id"]
        transform.transform.translation.x = point["x_m"]
        transform.transform.translation.y = point["y_m"]
        transform.transform.translation.z = point["z_m"]
        half_yaw_rad = point["yaw_rad"] / 2.0
        transform.transform.rotation.z = math.sin(half_yaw_rad)
        transform.transform.rotation.w = math.cos(half_yaw_rad)
        return transform

    def send_transforms(self, transforms):
        self._broadcaster.sendTransform(transforms)


def main():
    rospy.init_node("buoy_waypoint_publisher", anonymous=False)
    try:
        BuoyWaypointPublisher()
    except ValueError as exc:
        rospy.logfatal("[BuoyWaypoints] Invalid configuration: %s", exc)
        return
    rospy.spin()


if __name__ == "__main__":
    main()
