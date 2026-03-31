#!/usr/bin/env python3

from typing import Optional, Tuple

import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, SetBoolResponse

import numpy as np

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class StationTrajectoryPublisherNode:
    @staticmethod
    def _get_float_param(name: str, default: float) -> float:
        value = rospy.get_param(name, default)
        if isinstance(value, (int, float)):
            return float(value)

        rospy.logwarn("Invalid value for %s. Using default %s", name, default)
        return default

    def __init__(self):
        rospy.init_node("station_trajectory_publisher")

        self.enable = False

        self.odom_frame = str(rospy.get_param("~odom_frame", "odom"))
        self.robot_frame = str(rospy.get_param("~robot_frame", "taluy/base_link"))
        self.octagon_frame = str(rospy.get_param("~octagon_frame", "octagon_link"))
        self.torpedo_frame = str(
            rospy.get_param("~torpedo_frame", "torpedo_map_link")
        )
        self.pinger_frame = str(rospy.get_param("~pinger_frame", "pinger_link"))
        self.station_frame = str(rospy.get_param("~station_frame", "station_frame"))
        self.publish_rate = self._get_float_param("~publish_rate", 5.0)
        self.station_yaw_override = self._get_float_param("~station_yaw_override", 0.0)
        self.tie_tolerance_xy = self._get_float_param(
            "~tie_tolerance_xy", 0.25
        )
        self.selection_topic = str(
            rospy.get_param("~selection_topic", "station_target_selection")
        )
        self.selection_metrics_topic = str(
            rospy.get_param("~selection_metrics_topic", "station_selection_metrics")
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.toggle_service = rospy.Service(
            "toggle_station_trajectory",
            SetBool,
            self._handle_toggle_service,
        )
        self.selection_publisher = rospy.Publisher(
            self.selection_topic,
            String,
            queue_size=1,
            latch=True,
        )
        self.selection_metrics_publisher = rospy.Publisher(
            self.selection_metrics_topic,
            Float32MultiArray,
            queue_size=1,
        )

        self.last_station_success_stamp = rospy.Time(0)
        self.last_station_message = "station frame not computed yet"
        self.last_selection = ""

    def _handle_toggle_service(self, req):
        self.enable = req.data
        message = (
            "Station trajectory publishing enabled"
            if self.enable
            else "Station trajectory publishing disabled"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def _lookup_transform(
        self, target_frame: str, source_frame: str
    ) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration.from_sec(4.0),
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                3.0,
                "TF lookup failed %s <- %s: %s",
                target_frame,
                source_frame,
                str(exc),
            )
            return None

    @staticmethod
    def _xy_from_transform(transform: TransformStamped) -> Tuple[float, float]:
        return transform.transform.translation.x, transform.transform.translation.y

    @staticmethod
    def find_circumcenter(point_1,point_2,point_3):
        def create_circle_equation(x,y):
            powr_x=x**2
            powr_y=y**2
            x_scaler=-2*x
            y_scaler=-2*y
            return powr_x,powr_y, x_scaler, y_scaler

        power_x_1,power_y_1,x_scaler_1,y_scaler_1=create_circle_equation(point_1[0],point_1[1])
        power_x_2,power_y_2,x_scaler_2,y_scaler_2=create_circle_equation(point_2[0],point_2[1])
        power_x_3,power_y_3,x_scaler_3,y_scaler_3=create_circle_equation(point_3[0],point_3[1])
        eq_1_x_scaler=x_scaler_1-x_scaler_2
        eq_1_y_scaler=y_scaler_1-y_scaler_2
        eq_2_y_scaler=y_scaler_1-y_scaler_3
        eq_2_x_scaler=x_scaler_1-x_scaler_3
        eq_1_c_scaler=power_x_1-power_x_2+power_y_1-power_y_2
        eq_2_c_scaler=power_x_1-power_x_3+power_y_1-power_y_3

        a=np.array([[eq_1_x_scaler,eq_1_y_scaler],[eq_2_x_scaler,eq_2_y_scaler]])
        b=np.array([-eq_1_c_scaler,-eq_2_c_scaler])
        try:
            solution = np.linalg.solve(a,b)
            return float(solution[0]), float(solution[1])
        except np.linalg.LinAlgError:
            rospy.logwarn("The points are collinear, a circle cannot be formed.")
            return None
    
    def _compute_station_xy_placeholder(
        self,
        octagon_xy: Tuple[float, float],
        torpedo_xy: Tuple[float, float],
        robot_xy: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        #circle center calculation using selected 3 frames
        station_xy=self.find_circumcenter(octagon_xy, torpedo_xy, robot_xy)
        return station_xy

    @staticmethod
    def _build_yaw_only_orientation(yaw: float):
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        orientation = Pose().orientation
        orientation.x = q[0]
        orientation.y = q[1]
        orientation.z = q[2]
        orientation.w = q[3]
        return orientation

    def _build_station_pose(
        self,
        station_xy: Tuple[float, float],
        robot_tf: TransformStamped,
    ) -> Pose:
        station_pose = Pose()
        station_pose.position.x = station_xy[0]
        station_pose.position.y = station_xy[1]
        station_pose.position.z = robot_tf.transform.translation.z

        station_pose.orientation = self._build_yaw_only_orientation(
            self.station_yaw_override
        )
        return station_pose

    @staticmethod
    def _distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _publish_selection(self, selection: str, d_torpedo: float, d_octagon: float):
        self.last_selection = selection
        self.selection_publisher.publish(String(data=selection))

        metrics = Float32MultiArray()
        metrics.data = [d_torpedo, d_octagon]
        self.selection_metrics_publisher.publish(metrics)

    def _compute_and_publish_selection(
        self,
        pinger_tf: TransformStamped,
        torpedo_tf: TransformStamped,
        octagon_tf: TransformStamped,
    ):
        pinger_xy = self._xy_from_transform(pinger_tf)
        torpedo_xy = self._xy_from_transform(torpedo_tf)
        octagon_xy = self._xy_from_transform(octagon_tf)

        d_torpedo = self._distance_xy(pinger_xy, torpedo_xy)
        d_octagon = self._distance_xy(pinger_xy, octagon_xy)

        if abs(d_torpedo - d_octagon) <= self.tie_tolerance_xy:
            selection = "NAVIGATE_TO_TORPEDO_TASK"
        elif d_torpedo < d_octagon:
            selection = "NAVIGATE_TO_TORPEDO_TASK"
        else:
            selection = "NAVIGATE_TO_OCTAGON_TASK"

        self._publish_selection(selection, d_torpedo, d_octagon)
        rospy.loginfo_throttle(
            1.0,
            "station selection=%s d_torpedo=%.3f d_octagon=%.3f eps=%.3f",
            selection,
            d_torpedo,
            d_octagon,
            self.tie_tolerance_xy,
        )

    def _build_transform_message(self, child_frame_id: str, pose: Pose):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation = pose.orientation
        return transform

    def _send_transform(self, transform: TransformStamped):
        req = SetObjectTransformRequest()
        req.transform = transform
        try:
            resp = self.set_object_transform_service.call(req)
            if not resp.success:
                rospy.logwarn(
                    "Failed to set transform for %s: %s",
                    transform.child_frame_id,
                    resp.message,
                )
        except rospy.ServiceException as exc:
            rospy.logerr("set_object_transform service call failed: %s", str(exc))

    def create_station_frame(self):
        octagon_tf = self._lookup_transform(self.odom_frame, self.octagon_frame)
        torpedo_tf = self._lookup_transform(self.odom_frame, self.torpedo_frame)
        robot_tf = self._lookup_transform(self.odom_frame, self.robot_frame)
        pinger_tf = self._lookup_transform(self.odom_frame, self.pinger_frame)

        if (
            octagon_tf is None
            or torpedo_tf is None
            or robot_tf is None
            or pinger_tf is None
        ):
            self.last_station_message = "missing required TF(s)"
            return

        octagon_xy = self._xy_from_transform(octagon_tf)
        torpedo_xy = self._xy_from_transform(torpedo_tf)
        robot_xy = self._xy_from_transform(robot_tf)

        station_xy = self._compute_station_xy_placeholder(
            octagon_xy,
            torpedo_xy,
            robot_xy,
        )

        if station_xy is None:
            self.last_station_message = "station XY calculation failed"
            rospy.logwarn_throttle(
                3.0,
                "Station XY is not computed yet. Implement geometric function in station_trajectory_publisher.py",
            )
            return

        station_pose = self._build_station_pose(
            station_xy,
            robot_tf,
        )
        station_transform = self._build_transform_message(
            self.station_frame, station_pose
        )
        self._send_transform(station_transform)
        self.last_station_success_stamp = rospy.Time.now()
        self.last_station_message = "station frame updated"
        self._compute_and_publish_selection(pinger_tf, torpedo_tf, octagon_tf)

    def spin(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            if self.enable:
                self.create_station_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = StationTrajectoryPublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
