#!/usr/bin/env python3
import math
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger, TriggerResponse

from auv_control.cfg import VisualServoingConfig
from auv_msgs.msg import PropsYaw
from auv_msgs.srv import VisualServoing, VisualServoingResponse
from auv_vision.slalom_segmentation import PipeDetection, SlalomSegmentor

from auv_control.vs_types import (
    ControllerConfig,
    ControllerState,
    SlalomState,
)
from auv_control.vs_slalom import compute_slalom_control
from auv_vision.slalom_debug import create_slalom_debug
import auv_common_lib.vision.camera_calibrations as camera_calibrations

REFERENCE_DEPTH_M = 2.0
PIPE_REAL_HEIGHT = 0.9


class DepthEstimationMethod(Enum):
    SEGMENTATION_HEIGHT = "segmentation_height"
    DEPTH_IMAGE = "depth_image"


DEPTH_ESTIMATION_METHOD = DepthEstimationMethod.DEPTH_IMAGE


def scale_depths(
    pipe_1: PipeDetection, pipe_2: PipeDetection, reference_m: float = REFERENCE_DEPTH_M
) -> Tuple[float, float]:
    """Scale relative depths so the nearest pipe is at reference_m."""
    min_depth = min(pipe_1.depth, pipe_2.depth)
    if min_depth <= 0:
        return reference_m, reference_m
    scale = reference_m / min_depth
    return pipe_1.depth * scale, pipe_2.depth * scale


def pipe_to_camera_position_depth_method(
    pipe: PipeDetection, fx: float, fy: float, depth_image: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute pipe position in camera frame using depth from PipeDetection.

    pipe.depth is already metric (meters) - computed as median depth
    in SlalomSegmentor from depth_anything output.
    """
    yaw = pipe.centroid[0]
    depth = pipe.depth

    x = depth
    y = -depth * math.tan(yaw)
    z = 0.0
    return (x, y, z)


def pipe_to_camera_position_seg_method(
    pipe: PipeDetection, fx: float, fy: float, real_height: float
) -> Tuple[float, float, float]:
    yaw = pipe.centroid[0]

    depth = (real_height * fy) / max(pipe.length, 1.0)

    x = depth
    y = -depth * math.tan(yaw)
    z = 0.0
    return (x, y, z)


def pipe_to_camera_position(
    pipe: PipeDetection, fx: float, fy: float, depth_image: np.ndarray
) -> Tuple[float, float, float]:
    if DEPTH_ESTIMATION_METHOD == DepthEstimationMethod.DEPTH_IMAGE:
        return pipe_to_camera_position_depth_method(pipe, fx, fy, depth_image)
    else:
        return pipe_to_camera_position_seg_method(pipe, fx, fy, PIPE_REAL_HEIGHT)


class VisualServoingNode:
    def __init__(self):
        rospy.init_node("visual_servoing_controller", anonymous=True)

        self._load_config()
        self._init_state()
        self._setup_ros()
        self._setup_dynamic_reconfigure()

        rospy.loginfo("[VS] Node ready")

    def _load_config(self):
        self.config = ControllerConfig(
            kp_gain=rospy.get_param("~kp_gain", 0.8),
            kd_gain=rospy.get_param("~kd_gain", 0.4),
            v_x_desired=rospy.get_param("~v_x_desired", 0.3),
            rate_hz=rospy.get_param("~rate_hz", 10.0),
            overall_timeout_s=rospy.get_param("~overall_timeout_s", 1500.0),
            navigation_timeout_s=rospy.get_param("~navigation_timeout_s", 12.0),
            max_angular_velocity=rospy.get_param("~max_angular_velocity", 1.0),
            slalom_mode=rospy.get_param("~slalom_mode", False),
            slalom_side=rospy.get_param("~slalom_side", "left"),
        )
        self.rgb_topic = rospy.get_param("~rgb_topic", "cameras/cam_front/image_raw")
        self.camera_frame = rospy.get_param(
            "~camera_frame", "taluy/base_link/front_camera_link"
        )
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.reference_depth_m = rospy.get_param(
            "~reference_depth_m", REFERENCE_DEPTH_M
        )

    def _init_state(self):
        self.state = ControllerState.IDLE
        self.target_prop = ""
        self.service_start_time = None
        self.last_detection_time = None
        self.debug_stream_enabled = False
        self.seg_debug_enabled = False
        self.rgb_buffer: Dict[float, Tuple[np.ndarray, Header]] = {}
        self.rgb_buffer_lock = threading.Lock()
        self.rgb_buffer_max_age = 3.6
        self.debug_thread = None
        self.depth_thread = None
        self.depth_lock = threading.Lock()
        self.depth_cond = threading.Condition(self.depth_lock)
        self.latest_depth_msg: Optional[Image] = None
        self.debug_lock = threading.Lock()
        self.debug_cond = threading.Condition(self.debug_lock)
        self.latest_debug_payload = None

        self.slalom_state = SlalomState()

        camera_ns = self.rgb_topic.rsplit("/", 1)[0]
        try:
            calib_fetcher = camera_calibrations.CameraCalibrationFetcher(
                camera_ns, True
            )
            calib = calib_fetcher.get_camera_info()
            self.fx = calib.K[0]
            self.fy = calib.K[4]
            self.cx = calib.K[2]
            rospy.loginfo(
                f"[VS] Camera calibration: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}"
            )
        except Exception as e:
            rospy.logwarn(
                f"[VS] Camera calibration failed: {e}, using linear normalization"
            )
            self.fx, self.fy, self.cx = None, None, None

        self.segmentor = SlalomSegmentor(fx=self.fx, cx=self.cx)
        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def _setup_ros(self):
        self.transform_pub = rospy.Publisher(
            "map/object_transform_updates", TransformStamped, queue_size=10
        )

        rospy.Subscriber("props_yaw", PropsYaw, self._on_prop, queue_size=1)
        rospy.Subscriber(
            self.rgb_topic + "/compressed", CompressedImage, self._on_rgb, queue_size=1
        )
        rospy.Subscriber(
            "depth_anything/raw_depth", Image, self._on_depth, queue_size=1
        )
        self.debug_pub = rospy.Publisher(
            "visual_servoing/debug/compressed", CompressedImage, queue_size=1
        )
        self.seg_components_pub = rospy.Publisher(
            "seg_debug/components", CompressedImage, queue_size=1
        )
        self.seg_pairing_pub = rospy.Publisher(
            "seg_debug/pairing", CompressedImage, queue_size=1
        )
        self.seg_detections_pub = rospy.Publisher(
            "seg_debug/detections", CompressedImage, queue_size=1
        )
        rospy.Service("visual_servoing/start_servoing", VisualServoing, self._srv_start)
        rospy.Service("visual_servoing/cancel_servoing", Trigger, self._srv_cancel)
        rospy.Service("visual_servoing/navigate", Trigger, self._srv_navigate)
        rospy.Service(
            "visual_servoing/toggle_debug_stream",
            Trigger,
            self._srv_toggle_debug_stream,
        )
        rospy.Service(
            "visual_servoing/toggle_seg_debug",
            Trigger,
            self._srv_toggle_seg_debug,
        )

    def _setup_dynamic_reconfigure(self):
        self.dyn_srv = Server(VisualServoingConfig, self._on_reconfig)
        self._start_debug_thread()
        self._start_depth_thread()

    def _start_debug_thread(self):
        if self.debug_thread is not None:
            return
        self.debug_thread = threading.Thread(
            target=self._debug_worker, name="vs_debug_worker", daemon=True
        )
        self.debug_thread.start()

    def _start_depth_thread(self):
        if self.depth_thread is not None:
            return
        self.depth_thread = threading.Thread(
            target=self._depth_worker, name="vs_depth_worker", daemon=True
        )
        self.depth_thread.start()

    def _debug_worker(self):
        while not rospy.is_shutdown():
            with self.debug_cond:
                while self.latest_debug_payload is None and not rospy.is_shutdown():
                    self.debug_cond.wait(timeout=0.2)
                payload = self.latest_debug_payload
                self.latest_debug_payload = None

            if payload is None:
                continue

            try:
                rgb = payload["rgb"]
                mask = payload["mask"]
                detections = payload["detections"]
                pair = payload["pair"]
                heading = payload["heading"]
                lateral = payload["lateral"]
                header = payload["header"]

                debug_vis = create_slalom_debug(
                    rgb=rgb,
                    mask=mask,
                    all_detections=detections,
                    selected_pair=pair,
                    alpha=0.7,
                    depth_shape=payload.get("depth_shape"),
                )
                mode = self.config.slalom_side
                if heading is not None and lateral is not None:
                    info_text = (
                        f"Mode: {mode} | Heading: {heading:.2f} | LatErr: {lateral:.2f}"
                    )
                else:
                    info_text = f"Mode: {mode} | No Pair"
                cv2.putText(
                    debug_vis,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                debug_msg = CompressedImage()
                debug_msg.header = header
                debug_msg.format = "jpeg"
                debug_msg.data = cv2.imencode(
                    ".jpg", debug_vis, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )[1].tobytes()
                self.debug_pub.publish(debug_msg)

            except Exception as e:
                rospy.logerr_throttle(5.0, f"[VS] Debug worker error: {e}")

    def _publish_debug_img(self, publisher, img, header):
        if img is None:
            return
        try:
            msg = CompressedImage()
            msg.header = header
            msg.format = "jpeg"
            msg.data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[
                1
            ].tobytes()
            publisher.publish(msg)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"[VS] Debug img publish error: {e}")

    def _publish_target_transform(self):
        try:
            pipe1_tf = self.tf_buffer.lookup_transform(
                "odom", "pipe_1", rospy.Time(0), rospy.Duration(0.1)
            )
            pipe2_tf = self.tf_buffer.lookup_transform(
                "odom", "pipe_2", rospy.Time(0), rospy.Duration(0.1)
            )
            robot_tf = self.tf_buffer.lookup_transform(
                "odom", "taluy/base_link", rospy.Time(0), rospy.Duration(0.1)
            )

            p1 = np.array(
                [pipe1_tf.transform.translation.x, pipe1_tf.transform.translation.y]
            )
            p2 = np.array(
                [pipe2_tf.transform.translation.x, pipe2_tf.transform.translation.y]
            )
            robot = np.array(
                [robot_tf.transform.translation.x, robot_tf.transform.translation.y]
            )

            midpoint = 0.5 * (p1 + p2)

            pipe_dir = p1 - p2
            pipe_len = np.linalg.norm(pipe_dir)
            if pipe_len < 0.01:
                return
            pipe_dir /= pipe_len
            normal = np.array([-pipe_dir[1], pipe_dir[0]])
            if np.dot(robot - midpoint, normal) < 0:
                normal *= -1

            yaw = math.atan2(normal[1], normal[0])

            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = "slalom_position_target"
            t.transform.translation.x = midpoint[0]
            t.transform.translation.y = midpoint[1]
            t.transform.translation.z = -1.3
            t.transform.rotation.z = math.sin(yaw / 2)
            t.transform.rotation.w = math.cos(yaw / 2)
            self.transform_pub.publish(t)

        except tf2_ros.TransformException:
            pass

    def _publish_pipe_transforms(
        self,
        pair: Tuple[PipeDetection, PipeDetection],
        stamp: rospy.Time,
        depth_image: np.ndarray,
    ):
        pipe_1, pipe_2 = pair

        pos_1 = pipe_to_camera_position(pipe_1, self.fx, self.fy, depth_image)
        pos_2 = pipe_to_camera_position(pipe_2, self.fx, self.fy, depth_image)

        dist_1 = math.sqrt(pos_1[0] ** 2 + pos_1[1] ** 2)
        dist_2 = math.sqrt(pos_2[0] ** 2 + pos_2[1] ** 2)

        rospy.loginfo_throttle(
            1.0,
            f"[VS] Pair: p1(yaw={pipe_1.centroid[0]:.3f}rad, len={pipe_1.length:.1f}px, dist={dist_1:.2f}m) "
            f"p2(yaw={pipe_2.centroid[0]:.3f}rad, len={pipe_2.length:.1f}px, dist={dist_2:.2f}m)",
        )

        for name, pos in [("pipe_1", pos_1), ("pipe_2", pos_2)]:
            try:
                # Create point in camera frame
                pt = PointStamped()
                pt.header.stamp = stamp
                pt.header.frame_id = self.camera_frame
                pt.point.x = pos[0]
                pt.point.y = pos[1]
                pt.point.z = pos[2]

                try:
                    # Transform point to odom frame
                    odom_pt = self.tf_buffer.transform(pt, self.odom_frame)

                    # Build TransformStamped for publishing
                    t = TransformStamped()
                    t.header.stamp = stamp
                    t.header.frame_id = self.odom_frame
                    t.child_frame_id = name
                    t.transform.translation.x = odom_pt.point.x
                    t.transform.translation.y = odom_pt.point.y
                    t.transform.translation.z = odom_pt.point.z
                    # Identity rotation - pipe aligned with odom frame
                    t.transform.rotation.w = 1.0

                    self.transform_pub.publish(t)
                    rospy.logdebug(
                        f"[VS] Published {name}: ({t.transform.translation.x:.2f}, "
                        f"{t.transform.translation.y:.2f})"
                    )
                except tf2_ros.TransformException as e:
                    rospy.logwarn_throttle(2.0, f"[VS] TF transform failed: {e}")

                    self.transform_pub.publish(t)

            except Exception as e:
                rospy.logerr_throttle(5.0, f"[VS] Pipe transform error: {e}")

    def _depth_worker(self):
        while not rospy.is_shutdown():
            with self.depth_cond:
                while self.latest_depth_msg is None and not rospy.is_shutdown():
                    self.depth_cond.wait(timeout=0.2)
                msg = self.latest_depth_msg
                self.latest_depth_msg = None

            if msg is None:
                continue

            should_process = (
                self.state != ControllerState.IDLE or self.debug_stream_enabled
            )
            if not should_process:
                continue

            try:
                depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                result = self.segmentor.process(
                    depth, return_debug=self.seg_debug_enabled
                )
                detections = result.get("detections", [])
                mask = result.get("mask")

                if self.seg_debug_enabled:
                    header = msg.header
                    self._publish_debug_img(
                        self.seg_components_pub, result.get("debug_components"), header
                    )
                    self._publish_debug_img(
                        self.seg_pairing_pub, result.get("debug_pairing"), header
                    )
                    self._publish_debug_img(
                        self.seg_detections_pub, result.get("debug_detections"), header
                    )

                # TODO remove - debug pipe detections
                if detections:
                    pipe_info = [
                        f"P{i}:d={d.depth:.2f}" for i, d in enumerate(detections)
                    ]
                    rospy.loginfo_throttle(
                        2.0,
                        f"[VS DEBUG] {len(detections)} pipes: {', '.join(pipe_info)}",
                    )

                heading = None
                lateral = None
                pair = None

                compute_for_debug = self.debug_stream_enabled or self.config.slalom_mode
                if compute_for_debug and detections:
                    heading, lateral, pair = compute_slalom_control(
                        detections,
                        mode=self.config.slalom_side,
                    )

                # Publish pipe transforms if pair selected
                if pair is not None:
                    self._publish_pipe_transforms(pair, msg.header.stamp, depth)
                self._publish_target_transform()
                if self.config.slalom_mode:
                    self.slalom_state.detections = detections
                    self.slalom_state.selected_pair = pair
                    self.slalom_state.lateral_error = lateral or 0.0

                    if self.state != ControllerState.IDLE and detections:
                        self.last_detection_time = (
                            rospy.Time.now()
                        )  # TODO doğru mu now?

                depth_stamp = msg.header.stamp.to_sec()
                rgb_data = None

                with self.rgb_buffer_lock:
                    if depth_stamp in self.rgb_buffer:
                        rgb_data = self.rgb_buffer[depth_stamp]
                    else:
                        closest_stamp = None
                        min_diff = 0.05

                        for s in self.rgb_buffer:
                            diff = abs(s - depth_stamp)
                            if diff < min_diff:
                                min_diff = diff
                                closest_stamp = s

                        if closest_stamp is not None:
                            rgb_data = self.rgb_buffer[closest_stamp]

                if rgb_data is None:
                    rospy.logwarn_throttle(
                        1.0,
                        f"[VS] No RGB for stamp {depth_stamp:.3f} (buffer empty or too far)",
                    )
                else:
                    rgb_img, rgb_header = rgb_data
                    payload = {
                        "rgb": rgb_img.copy(),
                        "mask": mask.copy() if mask is not None else None,
                        "detections": detections,
                        "pair": pair,
                        "heading": heading,
                        "lateral": lateral,
                        "header": rgb_header,
                        "depth_shape": depth.shape[:2],
                    }
                    with self.debug_cond:
                        self.latest_debug_payload = payload
                        self.debug_cond.notify()

            except Exception as e:
                rospy.logerr_throttle(5.0, f"[VS] Depth error: {e}")

    def _on_prop(self, msg: PropsYaw):
        if self.state == ControllerState.IDLE or msg.object != self.target_prop:
            return
        if self.config.slalom_mode:
            return
        self.last_detection_time = rospy.Time.now()

    def _on_rgb(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if rgb.shape[0] > 480:
                scale = 480.0 / rgb.shape[0]
                rgb = cv2.resize(
                    rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )

            stamp = msg.header.stamp.to_sec()
            now = rospy.Time.now().to_sec()

            with self.rgb_buffer_lock:
                self.rgb_buffer[stamp] = (rgb, msg.header)
                stale = [
                    k for k in self.rgb_buffer if now - k > self.rgb_buffer_max_age
                ]
                for k in stale:
                    del self.rgb_buffer[k]
        except Exception as e:
            rospy.logerr_throttle(5.0, f"[VS] RGB error: {e}")

    def _on_depth(self, msg: Image):
        with self.depth_cond:
            self.latest_depth_msg = msg
            self.depth_cond.notify()

    def _on_reconfig(self, config, level):
        self.config.kp_gain = config.get("kp_gain", self.config.kp_gain)
        self.config.kd_gain = config.get("kd_gain", self.config.kd_gain)
        self.config.max_angular_velocity = config.get(
            "max_angular_velocity", self.config.max_angular_velocity
        )
        self.config.v_x_desired = config.get("v_x_desired", self.config.v_x_desired)
        return config

    def _srv_start(self, req) -> VisualServoingResponse:
        if self.state != ControllerState.IDLE:
            return VisualServoingResponse(success=False, message="Already active")

        self.target_prop = req.target_prop
        self.state = ControllerState.CENTERING
        self.service_start_time = rospy.Time.now()
        self.last_detection_time = None
        self.slalom_state.reset()

        rospy.loginfo(f"[VS] Started: {self.target_prop}")
        return VisualServoingResponse(success=True, message="Activated")

    def _srv_cancel(self, req) -> TriggerResponse:
        if self.state == ControllerState.IDLE:
            return TriggerResponse(success=False, message="Not active")
        self._stop("cancelled")
        return TriggerResponse(success=True, message="Stopped")

    def _srv_navigate(self, req) -> TriggerResponse:
        if self.state != ControllerState.CENTERING:
            return TriggerResponse(success=False, message="Not in centering mode")
        self.state = ControllerState.NAVIGATING
        return TriggerResponse(success=True, message="Navigating")

    def _srv_toggle_debug_stream(self, req) -> TriggerResponse:
        self.debug_stream_enabled = not self.debug_stream_enabled
        msg = (
            "Debug stream enabled"
            if self.debug_stream_enabled
            else "Debug stream disabled"
        )
        rospy.loginfo(f"[VS] {msg}")
        return TriggerResponse(success=True, message=msg)

    def _srv_toggle_seg_debug(self, req) -> TriggerResponse:
        self.seg_debug_enabled = not self.seg_debug_enabled
        msg = (
            "Segmentation debug enabled"
            if self.seg_debug_enabled
            else "Segmentation debug disabled"
        )
        rospy.loginfo(f"[VS] {msg}")
        return TriggerResponse(success=True, message=msg)

    def _stop(self, reason: str):
        rospy.loginfo(f"[VS] Stopped: {reason}")
        self.state = ControllerState.IDLE

    def spin(self):
        rate = rospy.Rate(self.config.rate_hz)
        while not rospy.is_shutdown():
            if self.state == ControllerState.IDLE:
                rate.sleep()
                continue

            elapsed = (rospy.Time.now() - self.service_start_time).to_sec()
            if elapsed > self.config.overall_timeout_s:
                self._stop("timeout")
                continue

            if self.last_detection_time:
                time_since = (rospy.Time.now() - self.last_detection_time).to_sec()
            else:
                time_since = float("inf")

            if self.state == ControllerState.NAVIGATING:
                if time_since > self.config.navigation_timeout_s:
                    self.state = ControllerState.CENTERING

            rate.sleep()


if __name__ == "__main__":
    try:
        node = VisualServoingNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
