#!/usr/bin/env python3

import math
import numpy as np
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_msgs.msg import Float32, String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from dynamic_reconfigure.server import Server
from auv_mapping.cfg import PingerConfig


class PingerFramePublisher:
    def __init__(self):
        rospy.init_node("pinger_frame_publisher_node")

        self.is_collecting = False
        self.samples = []
        self.current_leg_samples = []
        self.pinger_pose = None
        self.close_pose = None

        self.pinger_frame = rospy.get_param("~pinger_frame", "pinger_frame")
        self.close_frame = rospy.get_param("~close_frame", "pinger_close_approach")
        self.close_distance = float(rospy.get_param("~close_distance", 2.0))
        self.waypoint_frame = rospy.get_param("~waypoint_frame", "pinger_waypoint")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "taluy/base_link")
        self.topic_name = rospy.get_param(
            "~topic_name", "/taluy/acoustic/hydrophone/base_angle"
        )
        self.search_distance = rospy.get_param("~search_distance", 2.0)
        self.outlier_threshold = rospy.get_param("~outlier_threshold", 0.26)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        rospy.loginfo(
            "PingerFramePublisher: Waiting for set_object_transform service..."
        )
        self.set_object_transform_service.wait_for_service()

        try:
            self.dyn_srv = Server(PingerConfig, self.reconfigure_callback)
        except Exception as e:
            rospy.logwarn(f"Dynamic reconfigure server could not be started: {e}")

        self.toggle_collection_srv = rospy.Service(
            "toggle_pinger_collection", SetBool, self.handle_toggle_collection
        )
        self.compute_position_srv = rospy.Service(
            "compute_pinger_position", Trigger, self.handle_compute_position
        )
        self.clear_data_srv = rospy.Service(
            "clear_pinger_data", Trigger, self.handle_clear_data
        )

        self.sub = rospy.Subscriber(self.topic_name, Float32, self.angle_callback)
        rospy.Subscriber("pinger_waypoint_direction", String, self.direction_callback)

        rospy.loginfo("PingerFramePublisher initialized successfully.")

    def direction_callback(self, msg):
        direction = msg.data.strip().lower()

        offsets = {
            "forward": (self.search_distance, 0.0),
            "backward": (-self.search_distance, 0.0),
            "left": (0.0, self.search_distance),
            "right": (0.0, -self.search_distance),
        }

        if direction not in offsets:
            rospy.logerr(
                f"Invalid pinger direction '{direction}'. Use: forward, backward, left, right."
            )
            return

        dx, dy = offsets[direction]

        try:
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "mission_start_link"
            t.child_frame_id = self.waypoint_frame
            t.transform.translation.x = dx
            t.transform.translation.y = dy
            t.transform.translation.z = 0.0
            t.transform.rotation.w = 1.0

            req_trans = SetObjectTransformRequest()
            req_trans.transform = t
            resp = self.set_object_transform_service.call(req_trans)

            if resp.success:
                rospy.loginfo(
                    f"Published waypoint '{self.waypoint_frame}' direction={direction} (dx={dx:.2f}, dy={dy:.2f})"
                )
            else:
                rospy.logwarn(f"set_object_transform failed: {resp.message}")
        except Exception as e:
            rospy.logerr(f"Failed to publish waypoint frame: {e}")

    def handle_toggle_collection(self, req):
        if req.data:
            self.current_leg_samples = []
            self.is_collecting = True
            msg = "Started pinger data collection."
            rospy.loginfo(msg)
        else:
            self.is_collecting = False
            if self.current_leg_samples:
                angles = [s["angle_world"] for s in self.current_leg_samples]
                angles_np = np.array(angles)
                diffs = np.arctan2(
                    np.sin(angles_np[:, None] - angles_np[None, :]),
                    np.cos(angles_np[:, None] - angles_np[None, :]),
                )
                sum_abs_diffs = np.sum(np.abs(diffs), axis=1)
                median_idx = np.argmin(sum_abs_diffs)
                median_angle = angles_np[median_idx]

                filtered_samples = []
                for sample in self.current_leg_samples:
                    diff = abs(
                        np.arctan2(
                            np.sin(sample["angle_world"] - median_angle),
                            np.cos(sample["angle_world"] - median_angle),
                        )
                    )
                    if diff <= self.outlier_threshold:
                        filtered_samples.append(sample)

                num_removed = len(self.current_leg_samples) - len(filtered_samples)
                self.samples.extend(filtered_samples)
                self.current_leg_samples = []
                msg = (
                    f"Stopped pinger data collection. Collected {len(angles)} samples, "
                    f"kept {len(filtered_samples)} (removed {num_removed} outliers, "
                    f"median angle: {math.degrees(median_angle):.1f}°)."
                )
                rospy.loginfo(msg)
            else:
                msg = "Stopped pinger data collection. No samples collected."
                rospy.loginfo(msg)
        return SetBoolResponse(success=True, message=msg)

    def handle_clear_data(self, req):
        self.samples = []
        self.current_leg_samples = []
        self.pinger_pose = None
        self.close_pose = None
        msg = "Cleared all collected pinger samples and reset pinger position."
        rospy.loginfo(msg)
        return TriggerResponse(success=True, message=msg)

    def angle_callback(self, msg):
        if not self.is_collecting:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                rospy.Time(0),
                rospy.Duration(0.5),
            )

            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            rx = transform.transform.rotation.x
            ry = transform.transform.rotation.y
            rz = transform.transform.rotation.z
            rw = transform.transform.rotation.w

            _, _, yaw = tf.transformations.euler_from_quaternion([rx, ry, rz, rw])

            angle_body = msg.data
            angle_world = yaw + angle_body

            self.current_leg_samples.append(
                {"pos": (tx, ty), "angle_world": angle_world}
            )

            rospy.loginfo_throttle(
                2.0,
                f"Collected sample #{len(self.current_leg_samples)}: pos=({tx:.2f}, {ty:.2f}), "
                f"angle_body={math.degrees(angle_body):.1f}°, "
                f"angle_world={math.degrees(angle_world):.1f}°",
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(2.0, f"TF lookup failed: {e}")

    def handle_compute_position(self, req):
        if len(self.samples) < 2:
            return TriggerResponse(
                success=False,
                message=f"Not enough samples to compute position. Count: {len(self.samples)}",
            )

        A = np.zeros((2, 2))
        b = np.zeros(2)

        for sample in self.samples:
            theta = sample["angle_world"]
            nx = -math.sin(theta)
            ny = math.cos(theta)
            n = np.array([nx, ny])
            r = np.array(sample["pos"])

            A += np.outer(n, n)
            b += np.dot(n, r) * n

        try:
            p = np.linalg.solve(A, b)
            self.pinger_pose = Pose()
            self.pinger_pose.position.x = p[0]
            self.pinger_pose.position.y = p[1]
            self.pinger_pose.position.z = 0.0
            self.pinger_pose.orientation.w = 1.0

            msg = f"Computed pinger position: x={p[0]:.3f}, y={p[1]:.3f}"
            rospy.loginfo(msg)

            if not self.send_pinger_transforms():
                rospy.logwarn("Pinger position was computed but frame publication was incomplete")

            return TriggerResponse(success=True, message=msg)

        except np.linalg.LinAlgError:
            return TriggerResponse(
                success=False, message="Intersection failed: singular matrix"
            )

    def _publish_pose(self, frame_name, pose):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = frame_name
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation

        req = SetObjectTransformRequest(transform=t)
        try:
            resp = self.set_object_transform_service.call(req)
            if not resp.success:
                rospy.logwarn(f"Failed to publish {frame_name}: {resp.message}")
                return False
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to set_object_transform failed: {e}")
            return False

    def _compute_close_pose(self):
        try:
            robot_transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn("Unable to compute pinger close frame: %s", e)
            return None

        px = self.pinger_pose.position.x
        py = self.pinger_pose.position.y
        rx = robot_transform.transform.translation.x
        ry = robot_transform.transform.translation.y
        vx = rx - px
        vy = ry - py
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            rospy.logwarn("Robot and pinger positions are coincident; close frame unavailable")
            return None

        close_pose = Pose()
        close_pose.position.x = px + self.close_distance * vx / norm
        close_pose.position.y = py + self.close_distance * vy / norm
        close_pose.position.z = self.pinger_pose.position.z
        yaw = math.atan2(close_pose.position.y - ry, close_pose.position.x - rx)
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        close_pose.orientation.x = quaternion[0]
        close_pose.orientation.y = quaternion[1]
        close_pose.orientation.z = quaternion[2]
        close_pose.orientation.w = quaternion[3]
        return close_pose

    def send_pinger_transforms(self):
        if self.pinger_pose is None:
            return False

        if self.close_pose is None:
            self.close_pose = self._compute_close_pose()
        pinger_ok = self._publish_pose(self.pinger_frame, self.pinger_pose)
        close_ok = self.close_pose is not None and self._publish_pose(
            self.close_frame, self.close_pose
        )
        return pinger_ok and close_ok

    def send_pinger_transform(self):
        return self.send_pinger_transforms()

    def reconfigure_callback(self, config, level):
        try:
            self.search_distance = config.get("search_distance", self.search_distance)
            new_close_distance = config.get("close_distance", self.close_distance)
            if new_close_distance != self.close_distance:
                self.close_pose = None
            self.close_distance = new_close_distance
            rospy.loginfo_throttle(
                5.0, f"Dynamic reconfigure: search_distance={self.search_distance:.3f}"
            )
        except Exception as e:
            rospy.logwarn(f"Reconfigure callback error: {e}")
        return config

    def spin(self):
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            if self.pinger_pose is not None:
                self.send_pinger_transforms()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = PingerFramePublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass
