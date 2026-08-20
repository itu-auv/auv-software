#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rospy
import tf2_ros
import tf.transformations

from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from auv_mapping.cfg import PingerConfig


class PingerFramePublisher:
    def __init__(self):
        rospy.init_node("pinger_frame_publisher_node")

        self.is_collecting = False

        # Valid samples used for final pinger position.
        self.samples = []

        # Samples from the currently active collection leg.
        self.current_leg_samples = []

        # Rejected samples are kept only for debug visualization.
        self.rejected_samples = []

        self.pinger_pose = None
        self.close_pose = None

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------

        self.pinger_frame = rospy.get_param(
            "~pinger_frame",
            "pinger_frame",
        )

        self.close_frame = rospy.get_param(
            "~close_frame",
            "pinger_close_approach",
        )

        self.close_distance = float(
            rospy.get_param(
                "~close_distance",
                2.0,
            )
        )

        self.waypoint_frame = rospy.get_param(
            "~waypoint_frame",
            "pinger_waypoint",
        )

        self.odom_frame = rospy.get_param(
            "~odom_frame",
            "odom",
        )

        self.robot_base_frame = rospy.get_param(
            "~robot_base_frame",
            "taluy/base_link",
        )

        self.topic_name = rospy.get_param(
            "~topic_name",
            "/taluy/acoustic/hydrophone/base_angle",
        )

        self.search_distance = float(
            rospy.get_param(
                "~search_distance",
                2.0,
            )
        )

        # Radian.
        # 0.26 rad ~= 14.9 degrees.
        self.outlier_threshold = float(
            rospy.get_param(
                "~outlier_threshold",
                0.26,
            )
        )

        # --------------------------------------------------------------
        # Debug image parameters
        # --------------------------------------------------------------

        self.debug_image_size = int(
            rospy.get_param(
                "~debug_image_size",
                900,
            )
        )

        self.debug_ray_length = float(
            rospy.get_param(
                "~debug_ray_length",
                6.0,
            )
        )

        self.debug_max_pixels_per_meter = float(
            rospy.get_param(
                "~debug_max_pixels_per_meter",
                100.0,
            )
        )

        self.debug_max_samples = int(
            rospy.get_param(
                "~debug_max_samples",
                250,
            )
        )

        # --------------------------------------------------------------
        # TF
        # --------------------------------------------------------------

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

        # --------------------------------------------------------------
        # set_object_transform service
        # --------------------------------------------------------------

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform",
            SetObjectTransform,
        )

        rospy.loginfo(
            "PingerFramePublisher: "
            "Waiting for set_object_transform service..."
        )

        self.set_object_transform_service.wait_for_service()

        # --------------------------------------------------------------
        # Dynamic reconfigure
        # --------------------------------------------------------------

        try:
            self.dyn_srv = Server(
                PingerConfig,
                self.reconfigure_callback,
            )

        except Exception as e:
            rospy.logwarn(
                f"Dynamic reconfigure server could not be started: {e}"
            )

        # --------------------------------------------------------------
        # Services
        # --------------------------------------------------------------

        self.toggle_collection_srv = rospy.Service(
            "toggle_pinger_collection",
            SetBool,
            self.handle_toggle_collection,
        )

        self.compute_position_srv = rospy.Service(
            "compute_pinger_position",
            Trigger,
            self.handle_compute_position,
        )

        self.clear_data_srv = rospy.Service(
            "clear_pinger_data",
            Trigger,
            self.handle_clear_data,
        )

        # --------------------------------------------------------------
        # Subscribers
        # --------------------------------------------------------------

        self.sub = rospy.Subscriber(
            self.topic_name,
            Float32,
            self.angle_callback,
        )

        self.direction_sub = rospy.Subscriber(
            "pinger_waypoint_direction",
            String,
            self.direction_callback,
        )

        # --------------------------------------------------------------
        # Debug image
        # --------------------------------------------------------------

        self.cv_bridge = CvBridge()

        self.debug_image_pub = rospy.Publisher(
            "~debug_image",
            Image,
            queue_size=1,
        )

        rospy.loginfo(
            "PingerFramePublisher initialized successfully."
        )

    # ==================================================================
    # ANGLE FILTER
    # ==================================================================

    @staticmethod
    def _angle_difference(angle_a, angle_b):
        """
        Circular angular difference in range [-pi, pi].
        """
        return math.atan2(
            math.sin(angle_a - angle_b),
            math.cos(angle_a - angle_b),
        )

    def _filter_angle_samples(self, samples):
        """
        Find the dominant group of mutually similar angles.

        Important:
        We DO NOT reduce the valid samples into one median/mean angle.

        Example:

            29, 30, 31, 32, 78, 120 deg

        With ~15 deg threshold:

            valid:
                29, 30, 31, 32

            rejected:
                78, 120

        All valid samples keep their original angle_world values.
        """

        if not samples:
            return [], []

        if len(samples) == 1:
            return list(samples), []

        angles = np.array(
            [
                sample["angle_world"]
                for sample in samples
            ],
            dtype=float,
        )

        # Pairwise circular angle difference.
        diffs = np.arctan2(
            np.sin(
                angles[:, None]
                - angles[None, :]
            ),
            np.cos(
                angles[:, None]
                - angles[None, :]
            ),
        )

        abs_diffs = np.abs(diffs)

        # True if two samples are sufficiently close.
        close_matrix = (
            abs_diffs
            <= self.outlier_threshold
        )

        # Every sample sees itself as True, so subtract one.
        support_counts = (
            np.sum(
                close_matrix,
                axis=1,
            )
            - 1
        )

        # Sample with the largest number of nearby angles.
        anchor_idx = int(
            np.argmax(support_counts)
        )

        anchor_angle = angles[anchor_idx]

        # Keep every sample close to the dominant anchor.
        valid_mask = close_matrix[anchor_idx]

        filtered_samples = []
        rejected_samples = []

        for sample, valid in zip(
            samples,
            valid_mask,
        ):
            if valid:
                filtered_samples.append(sample)
            else:
                rejected_samples.append(sample)

        rospy.loginfo(
            "Angle filter: raw=%d valid=%d rejected=%d "
            "anchor=%.1f deg threshold=%.1f deg",
            len(samples),
            len(filtered_samples),
            len(rejected_samples),
            math.degrees(anchor_angle),
            math.degrees(self.outlier_threshold),
        )

        if filtered_samples:
            valid_angles = [
                round(
                    math.degrees(
                        sample["angle_world"]
                    ),
                    1,
                )
                for sample in filtered_samples
            ]

            rospy.loginfo(
                "Valid pinger angles: %s",
                valid_angles,
            )

        if rejected_samples:
            rejected_angles = [
                round(
                    math.degrees(
                        sample["angle_world"]
                    ),
                    1,
                )
                for sample in rejected_samples
            ]

            rospy.loginfo(
                "Rejected pinger angles: %s",
                rejected_angles,
            )

        return (
            filtered_samples,
            rejected_samples,
        )

    # ==================================================================
    # WAYPOINT
    # ==================================================================

    def direction_callback(self, msg):
        direction = msg.data.strip().lower()

        offsets = {
            "c0": (
                3.0,
                0.0,
            ),
            "r1": (
                3.0,
                -self.search_distance,
            ),
            "r2": (
                3.0,
                -self.search_distance * 2,
            ),
            "r3": (
                3.0,
                -self.search_distance * 3,
            ),
            "l1": (
                3.0,
                self.search_distance,
            ),
            "l2": (
                3.0,
                self.search_distance * 2,
            ),
            "l3": (
                3.0,
                self.search_distance * 3,
            ),
        }

        if direction not in offsets:
            rospy.logerr(
                f"Invalid pinger direction '{direction}'. "
                "Use: c0, r1, r2, r3, l1, l2, l3."
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

            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            req = SetObjectTransformRequest()
            req.transform = t

            resp = self.set_object_transform_service.call(
                req
            )

            if resp.success:
                rospy.loginfo(
                    f"Published waypoint "
                    f"'{self.waypoint_frame}' "
                    f"direction={direction} "
                    f"(dx={dx:.2f}, dy={dy:.2f})"
                )

            else:
                rospy.logwarn(
                    f"set_object_transform failed: "
                    f"{resp.message}"
                )

        except Exception as e:
            rospy.logerr(
                f"Failed to publish waypoint frame: {e}"
            )

    # ==================================================================
    # COLLECTION
    # ==================================================================

    def handle_toggle_collection(self, req):
        if req.data:
            # New leg.
            self.current_leg_samples = []

            self.is_collecting = True

            msg = (
                "Started pinger data collection."
            )

            rospy.loginfo(msg)

            return SetBoolResponse(
                success=True,
                message=msg,
            )

        # --------------------------------------------------------------
        # Stop collection
        # --------------------------------------------------------------

        self.is_collecting = False

        if not self.current_leg_samples:
            msg = (
                "Stopped pinger data collection. "
                "No samples collected."
            )

            rospy.loginfo(msg)

            return SetBoolResponse(
                success=True,
                message=msg,
            )

        raw_count = len(
            self.current_leg_samples
        )

        (
            filtered_samples,
            rejected_samples,
        ) = self._filter_angle_samples(
            self.current_leg_samples
        )

        # IMPORTANT:
        # Every valid sample keeps its own angle.
        self.samples.extend(
            filtered_samples
        )

        # Rejected samples are retained only for visualization.
        self.rejected_samples.extend(
            rejected_samples
        )

        valid_count = len(
            filtered_samples
        )

        rejected_count = len(
            rejected_samples
        )

        self.current_leg_samples = []

        msg = (
            f"Stopped pinger data collection. "
            f"Collected {raw_count}, "
            f"valid {valid_count}, "
            f"rejected {rejected_count}. "
            f"Total valid samples: "
            f"{len(self.samples)}."
        )

        rospy.loginfo(msg)

        return SetBoolResponse(
            success=True,
            message=msg,
        )

    def handle_clear_data(self, req):
        self.samples = []
        self.current_leg_samples = []
        self.rejected_samples = []

        self.pinger_pose = None
        self.close_pose = None

        msg = (
            "Cleared all collected pinger samples "
            "and reset pinger position."
        )

        rospy.loginfo(msg)

        return TriggerResponse(
            success=True,
            message=msg,
        )

    # ==================================================================
    # ANGLE CALLBACK
    # ==================================================================

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

            tx = (
                transform
                .transform
                .translation
                .x
            )

            ty = (
                transform
                .transform
                .translation
                .y
            )

            rx = (
                transform
                .transform
                .rotation
                .x
            )

            ry = (
                transform
                .transform
                .rotation
                .y
            )

            rz = (
                transform
                .transform
                .rotation
                .z
            )

            rw = (
                transform
                .transform
                .rotation
                .w
            )

            _, _, yaw = (
                tf.transformations.euler_from_quaternion(
                    [
                        rx,
                        ry,
                        rz,
                        rw,
                    ]
                )
            )

            # Hydrophone angle relative to robot body.
            angle_body = float(
                msg.data
            )

            # Convert to odom/world direction.
            angle_world = (
                yaw
                + angle_body
            )

            # Normalize to [-pi, pi].
            angle_world = math.atan2(
                math.sin(angle_world),
                math.cos(angle_world),
            )

            self.current_leg_samples.append(
                {
                    "pos": (
                        tx,
                        ty,
                    ),
                    "angle_world": angle_world,
                }
            )

            rospy.loginfo_throttle(
                2.0,
                f"Collected sample "
                f"#{len(self.current_leg_samples)}: "
                f"pos=({tx:.2f}, {ty:.2f}), "
                f"angle_body="
                f"{math.degrees(angle_body):.1f} deg, "
                f"angle_world="
                f"{math.degrees(angle_world):.1f} deg",
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                2.0,
                f"TF lookup failed: {e}",
            )

    # ==================================================================
    # PINGER POSITION
    # ==================================================================

    def handle_compute_position(self, req):
        if len(self.samples) < 2:
            return TriggerResponse(
                success=False,
                message=(
                    "Not enough valid samples "
                    "to compute position. "
                    f"Count: {len(self.samples)}"
                ),
            )

        A = np.zeros(
            (2, 2),
            dtype=float,
        )

        b = np.zeros(
            2,
            dtype=float,
        )

        # --------------------------------------------------------------
        # Least-squares intersection of ALL valid bearing lines.
        #
        # No median/average angle is used here.
        # Every valid sample contributes separately.
        # --------------------------------------------------------------

        for sample in self.samples:
            theta = sample[
                "angle_world"
            ]

            # Unit normal vector of bearing line.
            nx = -math.sin(theta)
            ny = math.cos(theta)

            n = np.array(
                [
                    nx,
                    ny,
                ],
                dtype=float,
            )

            r = np.array(
                sample["pos"],
                dtype=float,
            )

            A += np.outer(
                n,
                n,
            )

            b += (
                np.dot(n, r)
                * n
            )

        # Check matrix conditioning before solving.
        condition_number = np.linalg.cond(A)

        if not np.isfinite(condition_number):
            return TriggerResponse(
                success=False,
                message=(
                    "Intersection failed: "
                    "invalid matrix condition."
                ),
            )

        if condition_number > 1e8:
            rospy.logwarn(
                "Pinger intersection geometry is poorly "
                "conditioned. condition=%.3e",
                condition_number,
            )

        try:
            p = np.linalg.solve(
                A,
                b,
            )

        except np.linalg.LinAlgError:
            return TriggerResponse(
                success=False,
                message=(
                    "Intersection failed: "
                    "singular matrix."
                ),
            )

        self.pinger_pose = Pose()

        self.pinger_pose.position.x = float(
            p[0]
        )

        self.pinger_pose.position.y = float(
            p[1]
        )

        self.pinger_pose.position.z = 0.0

        self.pinger_pose.orientation.x = 0.0
        self.pinger_pose.orientation.y = 0.0
        self.pinger_pose.orientation.z = 0.0
        self.pinger_pose.orientation.w = 1.0

        # Recalculate because pinger position changed.
        self.close_pose = None

        msg = (
            f"Computed pinger position "
            f"using {len(self.samples)} valid samples: "
            f"x={p[0]:.3f}, "
            f"y={p[1]:.3f}, "
            f"condition={condition_number:.2e}"
        )

        rospy.loginfo(msg)

        if not self.send_pinger_transforms():
            rospy.logwarn(
                "Pinger position was computed "
                "but frame publication was incomplete."
            )

        return TriggerResponse(
            success=True,
            message=msg,
        )

    # ==================================================================
    # TF PUBLISHING
    # ==================================================================

    def _publish_pose(
        self,
        frame_name,
        pose,
    ):
        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = frame_name

        t.transform.translation.x = (
            pose.position.x
        )

        t.transform.translation.y = (
            pose.position.y
        )

        t.transform.translation.z = (
            pose.position.z
        )

        t.transform.rotation = (
            pose.orientation
        )

        req = SetObjectTransformRequest(
            transform=t
        )

        try:
            resp = (
                self.set_object_transform_service.call(
                    req
                )
            )

            if not resp.success:
                rospy.logwarn(
                    f"Failed to publish "
                    f"{frame_name}: "
                    f"{resp.message}"
                )

                return False

            return True

        except rospy.ServiceException as e:
            rospy.logerr(
                "Service call to "
                f"set_object_transform failed: {e}"
            )

            return False

    def _compute_close_pose(self):
        if self.pinger_pose is None:
            return None

        try:
            robot_transform = (
                self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.robot_base_frame,
                    rospy.Time(0),
                    rospy.Duration(1.0),
                )
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(
                "Unable to compute "
                "pinger close frame: %s",
                e,
            )

            return None

        px = (
            self.pinger_pose
            .position
            .x
        )

        py = (
            self.pinger_pose
            .position
            .y
        )

        rx = (
            robot_transform
            .transform
            .translation
            .x
        )

        ry = (
            robot_transform
            .transform
            .translation
            .y
        )

        # Vector from pinger toward robot.
        vx = rx - px
        vy = ry - py

        norm = math.hypot(
            vx,
            vy,
        )

        if norm < 1e-6:
            rospy.logwarn(
                "Robot and pinger positions are coincident; "
                "close frame unavailable."
            )

            return None

        close_pose = Pose()

        close_pose.position.x = (
            px
            + self.close_distance
            * vx
            / norm
        )

        close_pose.position.y = (
            py
            + self.close_distance
            * vy
            / norm
        )

        close_pose.position.z = (
            self.pinger_pose
            .position
            .z
        )

        # Orientation points from robot toward close point.
        yaw = math.atan2(
            close_pose.position.y - ry,
            close_pose.position.x - rx,
        )

        quaternion = (
            tf.transformations.quaternion_from_euler(
                0.0,
                0.0,
                yaw,
            )
        )

        close_pose.orientation.x = quaternion[0]
        close_pose.orientation.y = quaternion[1]
        close_pose.orientation.z = quaternion[2]
        close_pose.orientation.w = quaternion[3]

        return close_pose

    def send_pinger_transforms(self):
        if self.pinger_pose is None:
            return False

        if self.close_pose is None:
            self.close_pose = (
                self._compute_close_pose()
            )

        pinger_ok = self._publish_pose(
            self.pinger_frame,
            self.pinger_pose,
        )

        close_ok = (
            self.close_pose is not None
            and self._publish_pose(
                self.close_frame,
                self.close_pose,
            )
        )

        return (
            pinger_ok
            and close_ok
        )

    def send_pinger_transform(self):
        return self.send_pinger_transforms()

    # ==================================================================
    # DEBUG IMAGE
    # ==================================================================

    def _get_robot_position(self):
        try:
            transform = (
                self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.robot_base_frame,
                    rospy.Time(0),
                    rospy.Duration(0.05),
                )
            )

            return (
                transform
                .transform
                .translation
                .x,
                transform
                .transform
                .translation
                .y,
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None

    @staticmethod
    def _nice_grid_step(
        target_step,
    ):
        """
        Select nice grid sizes:
        1, 2, 5, 10, 20, 50...
        """

        if target_step <= 0.0:
            return 1.0

        magnitude = (
            10.0
            ** math.floor(
                math.log10(
                    target_step
                )
            )
        )

        normalized = (
            target_step
            / magnitude
        )

        if normalized <= 1.0:
            nice = 1.0

        elif normalized <= 2.0:
            nice = 2.0

        elif normalized <= 5.0:
            nice = 5.0

        else:
            nice = 10.0

        return (
            nice
            * magnitude
        )

    def publish_debug_image(self):
        size = self.debug_image_size
        margin = 70

        image = np.zeros(
            (
                size,
                size,
                3,
            ),
            dtype=np.uint8,
        )

        # --------------------------------------------------------------
        # Limit number of rendered samples.
        # Calculation still uses ALL self.samples.
        # --------------------------------------------------------------

        valid_samples = (
            self.samples[
                -self.debug_max_samples:
            ]
        )

        rejected_samples = (
            self.rejected_samples[
                -self.debug_max_samples:
            ]
        )

        current_samples = (
            self.current_leg_samples[
                -self.debug_max_samples:
            ]
        )

        robot_position = (
            self._get_robot_position()
        )

        world_points = []

        def add_sample_bounds(sample):
            x, y = sample[
                "pos"
            ]

            theta = sample[
                "angle_world"
            ]

            end_x = (
                x
                + self.debug_ray_length
                * math.cos(theta)
            )

            end_y = (
                y
                + self.debug_ray_length
                * math.sin(theta)
            )

            world_points.append(
                (
                    x,
                    y,
                )
            )

            world_points.append(
                (
                    end_x,
                    end_y,
                )
            )

        for sample in valid_samples:
            add_sample_bounds(sample)

        for sample in rejected_samples:
            add_sample_bounds(sample)

        for sample in current_samples:
            add_sample_bounds(sample)

        if robot_position is not None:
            world_points.append(
                robot_position
            )

        if self.pinger_pose is not None:
            world_points.append(
                (
                    self.pinger_pose.position.x,
                    self.pinger_pose.position.y,
                )
            )

        if self.close_pose is not None:
            world_points.append(
                (
                    self.close_pose.position.x,
                    self.close_pose.position.y,
                )
            )

        if not world_points:
            world_points.append(
                (
                    0.0,
                    0.0,
                )
            )

        xs = [
            p[0]
            for p in world_points
        ]

        ys = [
            p[1]
            for p in world_points
        ]

        min_x = min(xs)
        max_x = max(xs)

        min_y = min(ys)
        max_y = max(ys)

        center_x = (
            0.5
            * (
                min_x
                + max_x
            )
        )

        center_y = (
            0.5
            * (
                min_y
                + max_y
            )
        )

        range_x = max(
            max_x - min_x,
            2.0,
        )

        range_y = max(
            max_y - min_y,
            2.0,
        )

        usable_size = (
            size
            - 2 * margin
        )

        scale_x = (
            usable_size
            / range_x
        )

        scale_y = (
            usable_size
            / range_y
        )

        pixels_per_meter = min(
            scale_x,
            scale_y,
            self.debug_max_pixels_per_meter,
        )

        pixels_per_meter = max(
            pixels_per_meter,
            1.0,
        )

        # --------------------------------------------------------------
        # World -> pixel
        # --------------------------------------------------------------

        def world_to_pixel(x, y):
            u = int(
                size / 2
                + (
                    x
                    - center_x
                )
                * pixels_per_meter
            )

            # Image coordinates grow downward.
            v = int(
                size / 2
                - (
                    y
                    - center_y
                )
                * pixels_per_meter
            )

            return (
                u,
                v,
            )

        # --------------------------------------------------------------
        # Grid
        # --------------------------------------------------------------

        visible_width_m = (
            size
            / pixels_per_meter
        )

        target_grid_step = max(
            1.0,
            60.0
            / pixels_per_meter,
        )

        grid_step = (
            self._nice_grid_step(
                target_grid_step
            )
        )

        view_min_x = (
            center_x
            - visible_width_m / 2
        )

        view_max_x = (
            center_x
            + visible_width_m / 2
        )

        view_min_y = (
            center_y
            - visible_width_m / 2
        )

        view_max_y = (
            center_y
            + visible_width_m / 2
        )

        gx = (
            math.floor(
                view_min_x
                / grid_step
            )
            * grid_step
        )

        while gx <= view_max_x:
            p1 = world_to_pixel(
                gx,
                view_min_y,
            )

            p2 = world_to_pixel(
                gx,
                view_max_y,
            )

            cv2.line(
                image,
                p1,
                p2,
                (35, 35, 35),
                1,
            )

            gx += grid_step

        gy = (
            math.floor(
                view_min_y
                / grid_step
            )
            * grid_step
        )

        while gy <= view_max_y:
            p1 = world_to_pixel(
                view_min_x,
                gy,
            )

            p2 = world_to_pixel(
                view_max_x,
                gy,
            )

            cv2.line(
                image,
                p1,
                p2,
                (35, 35, 35),
                1,
            )

            gy += grid_step

        # --------------------------------------------------------------
        # Odom axes
        # --------------------------------------------------------------

        cv2.line(
            image,
            world_to_pixel(
                view_min_x,
                0.0,
            ),
            world_to_pixel(
                view_max_x,
                0.0,
            ),
            (70, 70, 70),
            1,
        )

        cv2.line(
            image,
            world_to_pixel(
                0.0,
                view_min_y,
            ),
            world_to_pixel(
                0.0,
                view_max_y,
            ),
            (70, 70, 70),
            1,
        )

        # --------------------------------------------------------------
        # Bearing drawing helper
        # --------------------------------------------------------------

        def draw_sample(
            sample,
            color,
            thickness=1,
            radius=3,
        ):
            x, y = sample[
                "pos"
            ]

            theta = sample[
                "angle_world"
            ]

            end_x = (
                x
                + self.debug_ray_length
                * math.cos(theta)
            )

            end_y = (
                y
                + self.debug_ray_length
                * math.sin(theta)
            )

            origin_px = world_to_pixel(
                x,
                y,
            )

            end_px = world_to_pixel(
                end_x,
                end_y,
            )

            cv2.line(
                image,
                origin_px,
                end_px,
                color,
                thickness,
                cv2.LINE_AA,
            )

            cv2.circle(
                image,
                origin_px,
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------
        # Rejected bearings - RED
        # --------------------------------------------------------------

        for sample in rejected_samples:
            draw_sample(
                sample,
                (0, 0, 200),
                thickness=1,
                radius=2,
            )

        # --------------------------------------------------------------
        # Valid bearings - GREEN
        # --------------------------------------------------------------

        for sample in valid_samples:
            draw_sample(
                sample,
                (0, 220, 0),
                thickness=2,
                radius=3,
            )

        # --------------------------------------------------------------
        # Current/unfiltered bearings - YELLOW
        # --------------------------------------------------------------

        for sample in current_samples:
            draw_sample(
                sample,
                (0, 220, 220),
                thickness=1,
                radius=2,
            )

        # --------------------------------------------------------------
        # Close approach - CYAN
        # --------------------------------------------------------------

        if self.close_pose is not None:
            cx = (
                self.close_pose
                .position
                .x
            )

            cy = (
                self.close_pose
                .position
                .y
            )

            close_px = world_to_pixel(
                cx,
                cy,
            )

            cv2.circle(
                image,
                close_px,
                9,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "CLOSE",
                (
                    close_px[0] + 12,
                    close_px[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------
        # Pinger - MAGENTA
        # --------------------------------------------------------------

        if self.pinger_pose is not None:
            px = (
                self.pinger_pose
                .position
                .x
            )

            py = (
                self.pinger_pose
                .position
                .y
            )

            pinger_px = world_to_pixel(
                px,
                py,
            )

            cv2.circle(
                image,
                pinger_px,
                10,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.line(
                image,
                (
                    pinger_px[0] - 14,
                    pinger_px[1],
                ),
                (
                    pinger_px[0] + 14,
                    pinger_px[1],
                ),
                (255, 0, 255),
                2,
            )

            cv2.line(
                image,
                (
                    pinger_px[0],
                    pinger_px[1] - 14,
                ),
                (
                    pinger_px[0],
                    pinger_px[1] + 14,
                ),
                (255, 0, 255),
                2,
            )

            cv2.putText(
                image,
                (
                    f"PINGER "
                    f"({px:.2f}, {py:.2f})"
                ),
                (
                    pinger_px[0] + 15,
                    pinger_px[1] - 15,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------
        # Robot - WHITE
        # --------------------------------------------------------------

        if robot_position is not None:
            rx, ry = (
                robot_position
            )

            robot_px = world_to_pixel(
                rx,
                ry,
            )

            cv2.circle(
                image,
                robot_px,
                7,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "ROBOT",
                (
                    robot_px[0] + 10,
                    robot_px[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------
        # Information box
        # --------------------------------------------------------------

        cv2.rectangle(
            image,
            (10, 10),
            (355, 165),
            (15, 15, 15),
            -1,
        )

        cv2.putText(
            image,
            f"VALID: {len(self.samples)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            (
                f"REJECTED: "
                f"{len(self.rejected_samples)}"
            ),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 220),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            (
                f"COLLECTING: "
                f"{len(self.current_leg_samples)}"
            ),
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 220),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            (
                f"threshold: "
                f"{math.degrees(self.outlier_threshold):.1f} deg"
            ),
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            (
                f"grid: "
                f"{grid_step:.1f} m"
            ),
            (20, 133),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            (
                f"scale: "
                f"{pixels_per_meter:.1f} px/m"
            ),
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------
        # Publish ROS Image
        # --------------------------------------------------------------

        try:
            image_msg = (
                self.cv_bridge.cv2_to_imgmsg(
                    image,
                    encoding="bgr8",
                )
            )

            image_msg.header.stamp = (
                rospy.Time.now()
            )

            image_msg.header.frame_id = (
                self.odom_frame
            )

            self.debug_image_pub.publish(
                image_msg
            )

        except Exception as e:
            rospy.logwarn_throttle(
                2.0,
                f"Debug image publish failed: {e}",
            )

    # ==================================================================
    # DYNAMIC RECONFIGURE
    # ==================================================================

    def reconfigure_callback(
        self,
        config,
        level,
    ):
        try:
            self.search_distance = float(
                config.get(
                    "search_distance",
                    self.search_distance,
                )
            )

            new_close_distance = float(
                config.get(
                    "close_distance",
                    self.close_distance,
                )
            )

            if (
                new_close_distance
                != self.close_distance
            ):
                self.close_pose = None

            self.close_distance = (
                new_close_distance
            )

            # Works only if outlier_threshold exists in PingerConfig.
            if "outlier_threshold" in config:
                self.outlier_threshold = float(
                    config[
                        "outlier_threshold"
                    ]
                )

            rospy.loginfo_throttle(
                5.0,
                (
                    "Dynamic reconfigure: "
                    f"search_distance="
                    f"{self.search_distance:.3f}, "
                    f"close_distance="
                    f"{self.close_distance:.3f}, "
                    f"outlier_threshold="
                    f"{math.degrees(self.outlier_threshold):.1f} deg"
                ),
            )

        except Exception as e:
            rospy.logwarn(
                f"Reconfigure callback error: {e}"
            )

        return config

    # ==================================================================
    # MAIN LOOP
    # ==================================================================

    def spin(self):
        rate = rospy.Rate(5.0)

        while not rospy.is_shutdown():
            if self.pinger_pose is not None:
                self.send_pinger_transforms()

            self.publish_debug_image()

            rate.sleep()


if __name__ == "__main__":
    try:
        node = PingerFramePublisher()
        node.spin()

    except rospy.ROSInterruptException:
        pass
