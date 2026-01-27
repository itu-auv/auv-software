#!/usr/bin/env python3

import cv2
import numpy as np
import threading
from typing import Dict, Optional, Tuple
from std_msgs.msg import Header
import rospy
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Trigger, TriggerResponse

from auv_control.cfg import VisualServoingConfig
from auv_msgs.msg import PropsYaw
from auv_msgs.srv import VisualServoing, VisualServoingResponse
from auv_vision.slalom_segmentation import SlalomSegmentor

from auv_control.vs_types import (
    ControllerConfig,
    ControllerState,
    ErrorState,
    SlalomState,
)
from auv_control.vs_pd_controller import PDController
from auv_control.vs_slalom import compute_slalom_control
from auv_vision.slalom_debug import create_slalom_debug
import auv_common_lib.vision.camera_calibrations as camera_calibrations


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

    def _init_state(self):
        self.state = ControllerState.IDLE
        self.target_prop = ""
        self.service_start_time = None
        self.last_detection_time = None
        self.debug_stream_enabled = False
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

        self.error_state = ErrorState()
        self.slalom_state = SlalomState()
        self.pd = PDController(config=self.config, error_state=self.error_state)

        camera_ns = self.rgb_topic.rsplit("/", 1)[0]
        try:
            calib_fetcher = camera_calibrations.CameraCalibrationFetcher(
                camera_ns, True
            )
            calib = calib_fetcher.get_camera_info()
            fx = calib.K[0]
            cx = calib.K[2]
            rospy.loginfo(f"[VS] Camera calibration: fx={fx:.2f}, cx={cx:.2f}")
        except Exception as e:
            rospy.logwarn(
                f"[VS] Camera calibration failed: {e}, using linear normalization"
            )
            fx, cx = None, None

        self.segmentor = SlalomSegmentor(fx=fx, cx=cx)
        self.bridge = CvBridge()

    def _setup_ros(self):
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)
        self.error_pub = rospy.Publisher("visual_servoing/error", Float64, queue_size=1)

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

        rospy.Service("visual_servoing/start_servoing", VisualServoing, self._srv_start)
        rospy.Service("visual_servoing/cancel_servoing", Trigger, self._srv_cancel)
        rospy.Service("visual_servoing/navigate", Trigger, self._srv_navigate)
        rospy.Service(
            "visual_servoing/toggle_debug_stream",
            Trigger,
            self._srv_toggle_debug_stream,
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
                result = self.segmentor.process(depth, return_debug=False)
                detections = result.get("detections", [])
                mask = result.get("mask")
                rospy.loginfo_throttle(
                    5.0,
                    f"[VS DEBUG] depth={depth.shape}, mask={mask.shape if mask is not None else None}",
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

                if self.config.slalom_mode:
                    self.slalom_state.detections = detections
                    self.slalom_state.selected_pair = pair
                    self.slalom_state.lateral_error = lateral or 0.0

                    if self.state != ControllerState.IDLE:
                        if detections:
                            self.last_detection_time = rospy.Time.now()
                        if heading is not None:
                            self.pd.update_error(heading, rospy.Time.now().to_sec())
                            rospy.loginfo_throttle(
                                1.0, f"[SLALOM] heading={heading:.2f}"
                            )

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
                    rospy.loginfo_throttle(5.0, f"[VS DEBUG] rgb={rgb_img.shape}")
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
        self.pd.update_error(msg.angle, rospy.Time.now().to_sec())

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
        self.error_state.reset()
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

    def _stop(self, reason: str):
        rospy.loginfo(f"[VS] Stopped: {reason}")
        self.state = ControllerState.IDLE
        self.cmd_vel_pub.publish(Twist())
        self.enable_pub.publish(Bool(data=False))

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

            twist = Twist()
            twist.angular.z = self.pd.compute_angular(self.state)
            twist.linear.x = self.pd.compute_linear(self.state, time_since)

            self.enable_pub.publish(Bool(data=True))
            self.cmd_vel_pub.publish(twist)
            self.error_pub.publish(Float64(self.error_state.error))

            rate.sleep()


if __name__ == "__main__":
    try:
        node = VisualServoingNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
