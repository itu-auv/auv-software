#!/usr/bin/env python3
"""ViTPose process node: VitposeResult -> configured operations + debug overlay.

Operation framework (auv_vision/VITPOSE_PLAN.md §5). At startup every object
YAML under config/vitpose/ is loaded and its `process:` section built into an
object pipeline: camera calibration, an operation list, and a debug overlay
publisher. Incoming VitposeResult messages dispatch on msg.object — the
process node therefore never needs config switching; only the detection node
(which holds the one GPU model) switches.

Operations are modules under scripts/vitpose_ops/, each exporting
create_op(params, ctx) (same dynamic-import factory convention as
scripts/handlers/). An op implements:

    process(frame)          required; frame is a FrameData
    draw(image_bgr, frame)  optional; paint on the debug overlay, called only
                            while the overlay has subscribers. An op with its
                            own image topic (tetra_unfold) omits this so the
                            overlay stays pure model output.

and receives an OpContext with:

    ctx.object_name         the config object name
    ctx.camera_frame        camera optical frame id
    ctx.calibration()       -> (K 3x3, D) or None until camera_info arrives
    ctx.publish_tf(child_frame_id, xyz, stamp, rotation_quat=None)
                            routed through the object map TF server via
                            object_transform_updates — never a raw broadcast
    ctx.publisher(topic, msg_type, queue_size=1, latch=False)
                            lazily created, cached, for op-specific outputs

FrameData gives ids/pixels/scores for all K keypoints with raw confidences
(gate on them yourself), mask_probs (C, H, W) in [0, 1] with binary_masks()
at the calibrated threshold, keypoint_names, mask_classes, bbox and stamp.
Per-op init failures are contained: a bad op is disabled, the pipeline lives.

Debug overlay per object on vitpose_process_image_<object>/compressed:
keypoints (confidence-coloured, named), skeleton, bbox, translucent mask
tints, then each op's draw() layer. All overlay work is gated on subscriber
count.
"""

import importlib
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
import tf2_ros

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage, Image

import auv_common_lib.vision.camera_calibrations as camera_calibrations
from auv_msgs.msg import VitposeResult

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from utils.detection_utils import transform_to_odom_and_publish  # noqa: E402
from utils.vitpose_utils import all_object_configs  # noqa: E402

# Debug palette (BGR), indexed by mask class position. Deliberately matches
# the tetra channel order (red, green, blue) for the first three classes.
_MASK_COLORS = [
    (60, 60, 235),
    (90, 210, 60),
    (255, 120, 70),
    (60, 220, 220),
    (220, 60, 220),
    (220, 220, 60),
]
_COLOR_KP = (0, 220, 0)
_COLOR_KP_LOWCONF = (140, 140, 140)
_COLOR_SKELETON = (0, 160, 0)
_COLOR_BBOX = (0, 200, 255)
_COLOR_TEXT = (255, 255, 255)


# ─────────────────────────────────────────────── data passed to ops


@dataclass
class FrameData:
    """Everything one VitposeResult carries, unpacked for op consumption."""

    object_name: str
    stamp: rospy.Time
    ids: np.ndarray  # (N,) int32, 0-indexed keypoint ids
    pixels: np.ndarray  # (N, 2) float64 source-image px
    scores: np.ndarray  # (N,) float64 raw heatmap peak values
    keypoint_names: List[str]
    mask_probs: Optional[np.ndarray]  # (C, H, W) float32 in [0, 1], or None
    mask_classes: List[str]
    mask_threshold: float
    bbox: Optional[Tuple[float, float, float, float]]  # None = full frame

    def binary_masks(self) -> Optional[np.ndarray]:
        """(C, H, W) bool masks at the calibrated threshold."""
        if self.mask_probs is None:
            return None
        return self.mask_probs >= self.mask_threshold


class OpContext:
    """Per-object services handed to each op at construction."""

    def __init__(self, object_name, camera_cfg, tf_buffer, transform_pub):
        self.object_name = object_name
        self.camera_frame = camera_cfg["frame"]
        self.tf_buffer = tf_buffer
        self._transform_pub = transform_pub
        self._calibration = camera_calibrations.CameraCalibrationFetcher(
            camera_cfg["calibration_ns"], wait_for_camera_info=False
        )
        self._publishers: Dict[str, rospy.Publisher] = {}

    def calibration(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(K 3x3, D) from camera_info, or None until the first message."""
        info = (
            self._calibration.get_camera_info()
            if self._calibration.is_received()
            else None
        )
        if info is None:
            return None
        K = np.array(info.K, dtype=np.float64).reshape(3, 3)
        D = np.array(info.D, dtype=np.float64)
        return K, D

    def publish_tf(self, child_frame_id, xyz, stamp, rotation_quat=None):
        """Publish a pose through the object map TF server.

        xyz is (x, y, z) in the camera optical frame; rotation_quat is an
        optional (x, y, z, w) camera-frame orientation (identity when None —
        the 2D-detection convention).
        """
        transform_to_odom_and_publish(
            self.camera_frame,
            child_frame_id,
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            stamp,
            self.tf_buffer,
            self._transform_pub,
            rotation_quat=rotation_quat,
        )

    def publisher(self, topic, msg_type, queue_size=1, latch=False) -> rospy.Publisher:
        """Lazily created, cached publisher for op-specific outputs."""
        if topic not in self._publishers:
            self._publishers[topic] = rospy.Publisher(
                topic, msg_type, queue_size=queue_size, latch=latch
            )
        return self._publishers[topic]


# ─────────────────────────────────────────────── per-object pipeline


@dataclass
class ObjectPipeline:
    object_name: str
    config: dict
    ops: list
    ctx: OpContext
    debug_pub: rospy.Publisher
    image_topic: str
    skeleton: List[List[int]] = field(default_factory=list)
    viz_conf_threshold: float = 0.5


class VitposeProcessNode:
    def __init__(self):
        rospy.init_node("vitpose_process_node")

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )

        # Ring buffers of recent raw frames, one per image topic, so the
        # overlay is drawn on the exact frame the result came from (matched
        # by header stamp). Filled only while someone watches a debug topic.
        self._image_lock = threading.Lock()
        self._image_bufs: Dict[str, Deque[Tuple[float, np.ndarray]]] = {}

        self.pipelines: Dict[str, ObjectPipeline] = {}
        result_topics = set()
        for name, config in all_object_configs().items():
            try:
                pipeline = self._build_pipeline(name, config)
            except Exception as exc:
                rospy.logerr(
                    f"vitpose object '{name}' failed to initialize: {exc}. "
                    "Skipping it — other objects still work."
                )
                continue
            self.pipelines[name] = pipeline
            result_topics.add(config["process"]["result_topic"])

        if not self.pipelines:
            raise RuntimeError("no vitpose object configs could be initialized")

        for topic in sorted(result_topics):
            rospy.Subscriber(
                topic, VitposeResult, self._result_cb, queue_size=1, buff_size=2**24
            )
        for topic in sorted({p.image_topic for p in self.pipelines.values()}):
            self._image_bufs[topic] = deque(maxlen=60)
            rospy.Subscriber(
                topic,
                Image,
                lambda msg, t=topic: self._image_cb(msg, t),
                queue_size=1,
                buff_size=2**24,
            )

        total_ops = sum(len(p.ops) for p in self.pipelines.values())
        rospy.loginfo(
            f"vitpose_process_node ready — objects: "
            f"{sorted(self.pipelines)} ({total_ops} ops, "
            f"{len(result_topics)} result topics)"
        )

    # ------------------------------------------------------------- building

    def _build_pipeline(self, name, config) -> ObjectPipeline:
        process_cfg = config["process"]
        ctx = OpContext(name, process_cfg["camera"], self.tf_buffer, self.transform_pub)
        ops = []
        for op_cfg in process_cfg.get("operations", []):
            op_type = op_cfg["type"]
            params = op_cfg.get("params") or {}
            try:
                module = importlib.import_module(f"vitpose_ops.{op_type}")
                op = module.create_op(params, ctx)
            except Exception as exc:
                rospy.logerr(
                    f"op '{op_type}' for object '{name}' failed to load: {exc}. "
                    "Disabling this op — the rest of the pipeline still works."
                )
                continue
            ops.append(op)
        debug_pub = rospy.Publisher(
            f"vitpose_process_image_{name}/compressed", CompressedImage, queue_size=1
        )
        return ObjectPipeline(
            object_name=name,
            config=config,
            ops=ops,
            ctx=ctx,
            debug_pub=debug_pub,
            image_topic=process_cfg["image_topic"],
            skeleton=[list(pair) for pair in (process_cfg.get("skeleton") or [])],
            viz_conf_threshold=float(process_cfg.get("viz_conf_threshold", 0.5)),
        )

    # ------------------------------------------------------------- callbacks

    def _debug_wanted(self, image_topic) -> bool:
        return any(
            p.image_topic == image_topic and p.debug_pub.get_num_connections() > 0
            for p in self.pipelines.values()
        )

    def _image_cb(self, msg, topic):
        # Ring buffer feeds only the debug overlay — skip decode + copy
        # entirely while nobody is watching (house style).
        if not self._debug_wanted(topic):
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, f"[{topic}] image decode failed: {exc}")
            return
        with self._image_lock:
            self._image_bufs[topic].append((msg.header.stamp.to_sec(), img))

    def _nearest_image(self, topic, stamp) -> Optional[np.ndarray]:
        target = stamp.to_sec()
        with self._image_lock:
            buf = self._image_bufs.get(topic)
            if not buf:
                return None
            _, img = min(buf, key=lambda entry: abs(entry[0] - target))
        return img

    def _result_cb(self, msg: VitposeResult):
        pipeline = self.pipelines.get(msg.object)
        if pipeline is None:
            rospy.logwarn_throttle(
                10.0,
                f"VitposeResult for unknown object '{msg.object}' "
                f"(configured: {sorted(self.pipelines)})",
            )
            return

        frame = self._unpack(msg)
        for op in pipeline.ops:
            try:
                op.process(frame)
            except Exception as exc:
                rospy.logerr_throttle(
                    5.0,
                    f"op '{getattr(op, 'name', type(op).__name__)}' on "
                    f"'{msg.object}' raised: {exc}",
                )
        self._publish_debug(pipeline, msg, frame)

    # ------------------------------------------------------------- unpack

    def _unpack(self, msg: VitposeResult) -> FrameData:
        n = len(msg.keypoints)
        ids = np.array([kp.id for kp in msg.keypoints], dtype=np.int32)
        pixels = np.array(
            [(kp.x, kp.y) for kp in msg.keypoints], dtype=np.float64
        ).reshape(n, 2)
        scores = np.array([kp.confidence for kp in msg.keypoints], dtype=np.float64)
        mask_probs = None
        if msg.masks:
            planes = []
            for image in msg.masks:
                mono = self.bridge.imgmsg_to_cv2(image, desired_encoding="mono8")
                planes.append(mono.astype(np.float32) / 255.0)
            mask_probs = np.stack(planes)
        return FrameData(
            object_name=msg.object,
            stamp=msg.header.stamp,
            ids=ids,
            pixels=pixels,
            scores=scores,
            keypoint_names=list(msg.keypoint_names),
            mask_probs=mask_probs,
            mask_classes=list(msg.mask_classes),
            mask_threshold=float(msg.mask_threshold),
            bbox=tuple(msg.bbox) if len(msg.bbox) == 4 else None,
        )

    # ------------------------------------------------------------- overlay

    def _publish_debug(self, pipeline, msg, frame: FrameData):
        if pipeline.debug_pub.get_num_connections() == 0:
            return
        vis = self._nearest_image(pipeline.image_topic, msg.header.stamp)
        if vis is None:
            rospy.logwarn_throttle(
                10.0,
                f"[{pipeline.object_name}] no raw frame for overlay yet "
                f"(subscribed {pipeline.image_topic})",
            )
            return
        vis = vis.copy()

        # Mask tints (translucent, one colour per class).
        binary = frame.binary_masks()
        if binary is not None:
            for index, mask in enumerate(binary):
                if not mask.any():
                    continue
                color = _MASK_COLORS[index % len(_MASK_COLORS)]
                tint = np.zeros_like(vis)
                tint[...] = color
                vis[mask] = cv2.addWeighted(vis[mask], 0.6, tint[mask], 0.4, 0)

        # Bbox.
        if frame.bbox is not None:
            x, y, w, h = frame.bbox
            cv2.rectangle(
                vis, (int(x), int(y)), (int(x + w), int(y + h)), _COLOR_BBOX, 2
            )

        # Skeleton between confident keypoints.
        pt_by_id = {
            int(i): (float(px), float(py), float(s))
            for i, (px, py), s in zip(frame.ids, frame.pixels, frame.scores)
        }
        for a, b in pipeline.skeleton:
            if a in pt_by_id and b in pt_by_id:
                if (
                    pt_by_id[a][2] >= pipeline.viz_conf_threshold
                    and pt_by_id[b][2] >= pipeline.viz_conf_threshold
                ):
                    cv2.line(
                        vis,
                        (int(pt_by_id[a][0]), int(pt_by_id[a][1])),
                        (int(pt_by_id[b][0]), int(pt_by_id[b][1])),
                        _COLOR_SKELETON,
                        2,
                    )

        # Keypoints, confidence-coloured, named.
        for kp_id, (px, py, score) in pt_by_id.items():
            confident = score >= pipeline.viz_conf_threshold
            color = _COLOR_KP if confident else _COLOR_KP_LOWCONF
            point = (int(px), int(py))
            cv2.circle(vis, point, 5 if confident else 4, color, -1 if confident else 1)
            name = (
                frame.keypoint_names[kp_id]
                if kp_id < len(frame.keypoint_names)
                else str(kp_id)
            )
            cv2.putText(
                vis,
                f"{name} ({score:.2f})",
                (point[0] + 6, point[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

        # Header line.
        cv2.putText(
            vis,
            f"{pipeline.object_name}  kps {len(frame.ids)}  "
            f"masks {0 if binary is None else len(binary)}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _COLOR_TEXT,
            2,
        )

        # Op layers.
        for op in pipeline.ops:
            draw = getattr(op, "draw", None)
            if draw is None:
                continue
            try:
                draw(vis, frame)
            except Exception as exc:
                rospy.logwarn_throttle(
                    5.0,
                    f"op '{getattr(op, 'name', type(op).__name__)}' "
                    f"draw() raised: {exc}",
                )

        out = CompressedImage()
        out.header = msg.header
        out.format = "jpeg"
        ok, encoded = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        out.data = encoded.tobytes()
        pipeline.debug_pub.publish(out)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        VitposeProcessNode().run()
    except rospy.ROSInterruptException:
        pass
