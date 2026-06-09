#!/usr/bin/env python3

"""Valve keypoint producer: image → VITTrack bbox → ViTPose → KeypointResult.

Detector-free pipeline. The seed bounding box comes from a one-shot whole-image
ViTPose pass at start-up; thereafter a generic visual tracker (cv2.TrackerVit)
maintains the bbox per frame and ViTPose runs only on the tight tracked crop.
See VITTRACK_VALVE.md for the design rationale.

State machine:

    UNINIT  ──seed ok──▶  LOCKED  ──update ok──▶ LOCKED
       ▲                    │
       │                    └─update fail / conf collapse──▶ RECOVERING
       └────────────────── whole-image ViTPose ◀────────────────┘

The slow whole-image seed runs on a worker thread so it never blocks the
image callback; the tracker.update + tight-bbox ViTPose path stays on the
ROS subscriber thread.
"""

import os

# Must run before torch/ultralytics import (PyTorch ApproximateClock workaround,
# https://github.com/pytorch/pytorch/issues/91516).
os.environ.setdefault("KINETO_DISABLED", "1")

import sys
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospkg
import rospy

from cv_bridge import CvBridge

from auv_msgs.msg import Keypoint, KeypointResult
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import SetBool, SetBoolResponse, Empty, EmptyResponse

# vitpose_inference sits next to this file but isn't an installable module yet.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from utils.vitpose_inference import (  # noqa: E402
    SKELETON_10,
    SKELETON_11,
    load_valve_pose,
)


SKELETON_COLOR = (0, 255, 0)
KP_COLOR = (0, 80, 255)
LOW_CONF_COLOR = (100, 100, 100)
BBOX_COLOR = (0, 200, 255)
STATE_COLOR = (255, 255, 255)

# State machine
STATE_UNINIT = "UNINIT"
STATE_LOCKED = "LOCKED"
STATE_RECOVERING = "RECOVERING"


class ValveKeypointNode:
    def __init__(self):
        rospy.init_node("valve_keypoint_node")

        self.image_topic = rospy.get_param(
            "~image_topic", "/taluy/cameras/cam_front/image_raw"
        )
        self.result_topic = rospy.get_param("~result_topic", "/keypoint_result_front")
        self.image_out_topic = rospy.get_param(
            "~image_out_topic", "/keypoint_image_front"
        )

        det_pkg = rospkg.RosPack().get_path("auv_detection")
        self.vitpose_model_path = rospy.get_param(
            "~vitpose_model_path", os.path.join(det_pkg, "models", "valve_vitpose.pth")
        )
        # vitTracker.onnx (~700 KB) lives at
        # opencv/opencv_extra/testdata/dnn/onnx/models/vitTracker.onnx.
        # Vendor it into auv_detection/models/ and load by absolute path.
        self.vit_tracker_model_path = rospy.get_param(
            "~vit_tracker_model_path",
            os.path.join(det_pkg, "models", "vitTracker.onnx"),
        )
        self.device = rospy.get_param("~device", "cpu")
        # Keypoint count for the TensorRT (.engine) backend, where the output
        # channel count can't be auto-detected. Ignored by the PyTorch (.pth)
        # backend, which reads it straight from the checkpoint.
        self.vitpose_num_kps = int(rospy.get_param("~vitpose_num_kps", 11))

        # Confidence gate for publishing individual keypoints downstream.
        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.5))
        # Confidence gate for accepting a keypoint into the seed bbox extent.
        self.seed_kp_conf_threshold = float(
            rospy.get_param("~seed_kp_conf_threshold", 0.3)
        )
        # Min keypoints passing seed_kp_conf_threshold before we accept a seed.
        # If fewer pass, we either fall back to top-K or skip this seed attempt.
        self.seed_min_kps = int(rospy.get_param("~seed_min_kps", 3))
        self.seed_topk_fallback = int(rospy.get_param("~seed_topk_fallback", 5))
        # Mean confidence required to call a seed acceptable.
        self.seed_mean_conf_min = float(rospy.get_param("~seed_mean_conf_min", 0.35))
        # Seed bbox is tight (extent of high-conf keypoints) with a small pad
        # — the tracker handles inter-frame margin itself.
        self.seed_bbox_pad = float(rospy.get_param("~seed_bbox_pad", 0.05))

        # Recovery gates while LOCKED.
        # Min bbox side in px; below this we declare collapse.
        self.min_bbox_side = float(rospy.get_param("~min_bbox_side", 30.0))
        # Mean keypoint conf inside the tracked bbox required to stay locked.
        self.lock_mean_conf_min = float(rospy.get_param("~lock_mean_conf_min", 0.40))
        # When running ViTPose inside the tracked bbox, expand the bbox by this
        # fraction so the keypoint head sees a little context around the valve.
        self.track_bbox_pad = float(rospy.get_param("~track_bbox_pad", 0.10))
        # Once the tracked bbox covers at least this fraction of the frame area,
        # the valve is close enough that a tight crop clips keypoints — feed
        # ViTPose the whole image instead. Tracking keeps running; only a
        # confidence collapse (lock_mean_conf_min) drops us to RECOVERING.
        self.full_image_bbox_area_frac = float(
            rospy.get_param("~full_image_bbox_area_frac", 0.5)
        )
        # How often (in successful LOCKED frames) to re-init VITTrack from the
        # crop's keypoint extent. 0 disables. Keeps the tracker anchored
        # without leaning on the slow whole-image ViTPose seed, which yields
        # an over-large bbox.
        self.lock_reseed_interval = int(rospy.get_param("~lock_reseed_interval", 30))
        # Frame stride between whole-image ViTPose seed attempts while in
        # UNINIT/RECOVERING. Default 5 — the whole-image pass is expensive,
        # no need to retry on every single frame.
        self.recover_seed_interval = max(
            1, int(rospy.get_param("~recover_seed_interval", 5))
        )

        # Producer emits 1-indexed keypoint ids; the consumer YAML
        # (valve_keypoints.yaml) maps those ids to model points.
        self.object_name = rospy.get_param("~object_name", "valve")

        rospy.loginfo(
            f"Loading ViTPose from {self.vitpose_model_path} (device={self.device})"
        )
        self.vp = load_valve_pose(
            self.vitpose_model_path,
            device=self.device,
            num_kps=self.vitpose_num_kps,
        )
        rospy.loginfo(f"ViTPose ready ({self.vp.num_kps} keypoints).")

        if not os.path.isfile(self.vit_tracker_model_path):
            raise FileNotFoundError(
                f"vitTracker.onnx not found at {self.vit_tracker_model_path}. "
                "Vendor it from opencv/opencv_extra/testdata/dnn/onnx/models/."
            )
        if not hasattr(cv2, "TrackerVit_create"):
            raise RuntimeError(
                "cv2.TrackerVit is not available. Install opencv-contrib-python "
                "(>=4.8); the plain opencv-python build does not include it."
            )
        rospy.loginfo(f"VITTrack ONNX: {self.vit_tracker_model_path}")

        self.bridge = CvBridge()
        self._debug_thread: Optional[threading.Thread] = None
        self._debug_lock = threading.Lock()

        # Tracker / state machine.
        self._lock = threading.Lock()
        self._state = STATE_UNINIT
        self._tracker = None
        self._tracked_bbox: Optional[Tuple[float, float, float, float]] = None
        self._seed_thread: Optional[threading.Thread] = None
        self._last_image_bgr: Optional[np.ndarray] = None
        # Successful-LOCKED-frame counter for periodic VITTrack reseeding.
        self._lock_frame_count = 0
        # Frame counter while in UNINIT/RECOVERING, used to stride seed attempts.
        self._recover_frame_count = 0

        self.enabled = bool(rospy.get_param("~enabled", True))
        # Private (~) so multiple instances (front + bottom camera) can coexist
        # under the same namespace without colliding on service names — callers
        # use `<node_name>/set_enabled` and `<node_name>/reseed`.
        self.set_enabled_service = rospy.Service(
            "~set_enabled", SetBool, self._handle_set_enabled
        )
        # Optional explicit re-seed trigger (mirrors the integration sketch in
        # VITTRACK_VALVE.md).
        self.reseed_service = rospy.Service("~reseed", Empty, self._handle_reseed)

        self.result_pub = rospy.Publisher(
            self.result_topic, KeypointResult, queue_size=1
        )
        self.image_pub = rospy.Publisher(
            self.image_out_topic + "/compressed", CompressedImage, queue_size=1
        )

        rospy.Subscriber(
            self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2**24
        )

        rospy.loginfo(
            f"valve_keypoint_node ready (image={self.image_topic}, "
            f"result={self.result_topic}, image_out={self.image_out_topic})"
        )

    # ─────────────────────────────────────────────── services

    def _handle_set_enabled(self, req):
        self.enabled = bool(req.data)
        message = f"valve_keypoint_node enabled set to: {self.enabled}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def _handle_reseed(self, _req):
        """Force a re-seed: drop the current tracker and re-run whole-image ViTPose."""
        with self._lock:
            self._state = STATE_RECOVERING
            self._tracker = None
            self._tracked_bbox = None
            self._lock_frame_count = 0
            self._recover_frame_count = 0
        rospy.loginfo("valve_keypoint_node: re-seed requested, dropping tracker.")
        return EmptyResponse()

    # ─────────────────────────────────────────────── image callback

    def _image_cb(self, msg: Image):
        if not self.enabled:
            return
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr_throttle(5.0, f"image decode failed: {e}")
            return

        # Snapshot state under lock; perform expensive work (tracker.update,
        # ViTPose) without holding the lock.
        with self._lock:
            state = self._state
            tracker = self._tracker
            self._last_image_bgr = img_bgr  # for seed worker

        bbox_xywh: Optional[Tuple[float, float, float, float]] = None
        kps, scores = None, None

        if state in (STATE_UNINIT, STATE_RECOVERING):
            # Whole-image ViTPose is expensive; only attempt a seed every
            # recover_seed_interval frames. The worker also self-guards
            # against pile-up, but the stride lets the camera breathe between
            # back-to-back failed seeds.
            self._recover_frame_count += 1
            if self._recover_frame_count % self.recover_seed_interval == 0:
                self._maybe_start_seed_worker()
            # No tracked bbox yet → publish an empty result so downstream nodes
            # see we're alive but not locked.
            self._publish_result(msg.header, [])
        elif state == STATE_LOCKED and tracker is not None:
            ok, raw_bbox = tracker.update(img_bgr)
            if not ok or not self._bbox_is_sane(raw_bbox, img_bgr.shape):
                rospy.logwarn_throttle(
                    2.0,
                    "VITTrack lost (ok=%s bbox=%s); switching to RECOVERING"
                    % (ok, raw_bbox),
                )
                self._enter_recovering()
                self._publish_result(msg.header, [])
            else:
                # When the tracked valve fills a large fraction of the frame, a
                # tight crop buys nothing and risks clipping keypoints right when
                # the valve is closest — feed ViTPose the whole image instead.
                # The tracker still ran above, so tracking continuity is intact;
                # only a confidence collapse below drops us to RECOVERING.
                # Trigger off the RAW tracker extent (not the in-frame clamped
                # box) so a valve bigger than the frame reliably trips it — the
                # ratio can exceed 1.0.
                img_h, img_w = img_bgr.shape[:2]
                raw_area_frac = (raw_bbox[2] * raw_bbox[3]) / float(img_w * img_h)
                if raw_area_frac >= self.full_image_bbox_area_frac:
                    infer_bbox = (0.0, 0.0, float(img_w), float(img_h))
                else:
                    infer_bbox = self._pad_bbox(
                        raw_bbox, img_bgr.shape, self.track_bbox_pad
                    )
                # Clamped tracker box for bookkeeping; infer_bbox is what we fed.
                tracked_bbox = self._pad_bbox(
                    raw_bbox, img_bgr.shape, self.track_bbox_pad
                )
                # Draw what we actually fed to ViTPose in the debug viz.
                bbox_xywh = infer_bbox

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                kps, scores = self.vp.predict(img_rgb, infer_bbox)

                mean_conf = (
                    float(np.mean(scores))
                    if scores is not None and len(scores)
                    else 0.0
                )
                if mean_conf < self.lock_mean_conf_min:
                    rospy.logwarn_throttle(
                        2.0,
                        "ViTPose-on-bbox mean conf %.2f < %.2f; switching to RECOVERING"
                        % (mean_conf, self.lock_mean_conf_min),
                    )
                    self._enter_recovering()
                    # Still publish best-effort keypoints for this frame.
                    self._tracked_bbox = tracked_bbox
                else:
                    with self._lock:
                        self._tracked_bbox = tracked_bbox
                        self._lock_frame_count += 1
                        do_reseed = (
                            self.lock_reseed_interval > 0
                            and self._lock_frame_count >= self.lock_reseed_interval
                        )
                    if do_reseed:
                        self._reseed_from_crop_kps(img_bgr, kps, scores)

                keypoints = self._build_keypoints(kps, scores)
                self._publish_result(msg.header, keypoints)

        if self.image_pub.get_num_connections() > 0:
            self._start_debug_publish(
                img_bgr, state, bbox_xywh, kps, scores, msg.header
            )

    # ─────────────────────────────────────────────── seed worker

    def _maybe_start_seed_worker(self):
        """Spawn the whole-image ViTPose seed thread if one isn't already running."""
        with self._lock:
            if self._seed_thread is not None and self._seed_thread.is_alive():
                return
            if self._last_image_bgr is None:
                return
            frame = self._last_image_bgr.copy()
            self._seed_thread = threading.Thread(
                target=self._seed_worker, args=(frame,), daemon=True
            )
            self._seed_thread.start()

    def _seed_worker(self, frame_bgr: np.ndarray):
        """Whole-image ViTPose → seed bbox → init cv2.TrackerVit."""
        try:
            h, w = frame_bgr.shape[:2]
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Whole-image bbox — ViTPose's affine crop handles the resize.
            whole = (0.0, 0.0, float(w), float(h))
            kps, scores = self.vp.predict(img_rgb, whole)

            seed_bbox = self._compute_seed_bbox(kps, scores, frame_bgr.shape)
            if seed_bbox is None:
                rospy.logwarn_throttle(
                    2.0,
                    "Seed attempt rejected: too few high-confidence keypoints "
                    "(mean=%.2f)" % float(np.mean(scores)),
                )
                return

            try:
                params = cv2.TrackerVit_Params()
                params.net = self.vit_tracker_model_path
                tracker = cv2.TrackerVit_create(params)
                # init expects (x, y, w, h) ints.
                x, y, bw, bh = seed_bbox
                tracker.init(frame_bgr, (int(x), int(y), int(bw), int(bh)))
            except Exception as e:
                rospy.logerr(f"VITTrack init failed: {e}")
                return

            with self._lock:
                self._tracker = tracker
                self._tracked_bbox = seed_bbox
                self._state = STATE_LOCKED
            rospy.loginfo(
                "VITTrack seeded: bbox=(%.0f, %.0f, %.0f, %.0f) mean_conf=%.2f"
                % (
                    seed_bbox[0],
                    seed_bbox[1],
                    seed_bbox[2],
                    seed_bbox[3],
                    float(np.mean(scores)),
                )
            )
        except Exception as e:
            rospy.logerr_throttle(5.0, f"seed worker failed: {e}")

    def _reseed_from_crop_kps(self, frame_bgr, kps, scores):
        """Re-init VITTrack from the bbox extent of the current crop's keypoints.

        Runs no extra inference — uses the kps/scores already produced by
        ViTPose-on-bbox for this frame. Keeps the tracker anchored without
        falling back to the slow whole-image seed (which also yields a much
        looser bbox).
        """
        try:
            seed_bbox = self._compute_seed_bbox(kps, scores, frame_bgr.shape)
            if seed_bbox is None:
                # Low-conf frame: don't poison a working tracker with a bad seed.
                return

            params = cv2.TrackerVit_Params()
            params.net = self.vit_tracker_model_path
            new_tracker = cv2.TrackerVit_create(params)
            x, y, bw, bh = seed_bbox
            new_tracker.init(frame_bgr, (int(x), int(y), int(bw), int(bh)))
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"VITTrack periodic reseed failed: {e}")
            return

        with self._lock:
            # Only swap in if we're still LOCKED — avoid clobbering a
            # concurrent recovery transition or service-triggered reseed.
            if self._state != STATE_LOCKED:
                return
            self._tracker = new_tracker
            self._tracked_bbox = seed_bbox
            self._lock_frame_count = 0
        rospy.loginfo_throttle(
            10.0,
            "VITTrack reseeded from crop keypoints: bbox=(%.0f, %.0f, %.0f, %.0f)"
            % seed_bbox,
        )

    def _compute_seed_bbox(
        self, kps: np.ndarray, scores: np.ndarray, img_shape
    ) -> Optional[Tuple[float, float, float, float]]:
        """Tight axis-aligned bbox over high-confidence keypoints.

        Strategy per VITTRACK_VALVE.md:
          - keep keypoints with conf >= seed_kp_conf_threshold (0.3 default)
          - if fewer than seed_min_kps pass, fall back to top-K by confidence
          - reject if mean conf over the kept set is below seed_mean_conf_min
        """
        h_img, w_img = img_shape[:2]
        flat_scores = scores.reshape(-1)

        keep = np.where(flat_scores >= self.seed_kp_conf_threshold)[0]
        if len(keep) < self.seed_min_kps:
            # Top-K by confidence fallback.
            k = min(self.seed_topk_fallback, len(flat_scores))
            keep = np.argsort(flat_scores)[::-1][:k]

        if len(keep) < 2:
            return None

        kept_scores = flat_scores[keep]
        if float(np.mean(kept_scores)) < self.seed_mean_conf_min:
            return None

        pts = kps[keep]
        x0 = float(np.min(pts[:, 0]))
        y0 = float(np.min(pts[:, 1]))
        x1 = float(np.max(pts[:, 0]))
        y1 = float(np.max(pts[:, 1]))
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)

        # Small pad — bbox stays intentionally tight (doc §"Seeding from ViTPose").
        pad_x = bw * self.seed_bbox_pad
        pad_y = bh * self.seed_bbox_pad
        x = max(0.0, x0 - pad_x)
        y = max(0.0, y0 - pad_y)
        bw = min(w_img - x, bw + 2 * pad_x)
        bh = min(h_img - y, bh + 2 * pad_y)

        if bw < self.min_bbox_side or bh < self.min_bbox_side:
            return None
        return (x, y, bw, bh)

    # ─────────────────────────────────────────────── helpers

    def _enter_recovering(self):
        with self._lock:
            self._state = STATE_RECOVERING
            self._tracker = None
            self._tracked_bbox = None
            self._lock_frame_count = 0
            self._recover_frame_count = 0

    def _bbox_is_sane(self, bbox, img_shape) -> bool:
        # A close valve legitimately overruns the frame, so a partially
        # off-screen bbox is NOT a failure — we clamp it for cropping instead
        # (see _pad_bbox). The only real "tracker lost it" signals are: no box,
        # a non-finite/degenerate box, or a box that has collapsed below
        # min_bbox_side (the tracker shrinking to nothing). Position is
        # deliberately not checked.
        if bbox is None:
            return False
        x, y, bw, bh = bbox
        if not all(np.isfinite(v) for v in (x, y, bw, bh)):
            return False
        if bw <= 0 or bh <= 0:
            return False
        # Collapse check uses the raw tracker extent, so a large but
        # off-screen valve never trips it.
        if min(bw, bh) < self.min_bbox_side:
            return False
        return True

    def _pad_bbox(
        self,
        bbox,
        img_shape,
        pad_frac: float,
    ) -> Tuple[float, float, float, float]:
        """Pad the bbox then intersect it with the image, so an origin or
        extent that lies outside the frame (normal for a close valve) still
        yields a valid in-frame crop."""
        x, y, bw, bh = bbox
        h_img, w_img = img_shape[:2]
        pad_x = bw * pad_frac
        pad_y = bh * pad_frac
        # Padded corners in image coords, then clamp into [0, w] x [0, h].
        x0 = max(0.0, x - pad_x)
        y0 = max(0.0, y - pad_y)
        x1 = min(float(w_img), x + bw + pad_x)
        y1 = min(float(h_img), y + bh + pad_y)
        bw_p = max(1.0, x1 - x0)
        bh_p = max(1.0, y1 - y0)
        return (float(x0), float(y0), float(bw_p), float(bh_p))

    def _build_keypoints(self, kps: np.ndarray, scores: np.ndarray) -> List[Keypoint]:
        out: List[Keypoint] = []
        for i in range(len(kps)):
            conf = float(scores[i, 0])
            if conf < self.conf_threshold:
                continue
            msg = Keypoint()
            msg.id = i + 1
            msg.x = float(kps[i, 0])
            msg.y = float(kps[i, 1])
            msg.confidence = conf
            msg.object = self.object_name
            out.append(msg)
        return out

    def _publish_result(self, header, keypoints: List[Keypoint]):
        result = KeypointResult()
        result.header = header
        result.keypoints = keypoints
        self.result_pub.publish(result)

    # ─────────────────────────────────────────────── debug viz

    def _start_debug_publish(self, img_bgr, state, bbox_xywh, kps, scores, header):
        with self._debug_lock:
            if self._debug_thread is not None and self._debug_thread.is_alive():
                return
            self._debug_thread = threading.Thread(
                target=self._publish_debug,
                args=(img_bgr.copy(), state, bbox_xywh, kps, scores, header),
                daemon=True,
            )
            self._debug_thread.start()

    def _publish_debug(self, vis, state, bbox_xywh, kps, scores, header):
        try:
            cv2.putText(
                vis,
                f"state: {state}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                STATE_COLOR,
                2,
            )

            if bbox_xywh is not None:
                x, y, bw, bh = bbox_xywh
                cv2.rectangle(
                    vis,
                    (int(x), int(y)),
                    (int(x + bw), int(y + bh)),
                    BBOX_COLOR,
                    2,
                )
                cv2.putText(
                    vis,
                    "VITTrack",
                    (int(x), int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    BBOX_COLOR,
                    1,
                )

            if kps is not None and scores is not None:
                n_kps = len(kps)
                if n_kps >= 11:
                    skeleton = SKELETON_11
                elif n_kps >= 10:
                    skeleton = SKELETON_10
                else:
                    skeleton = [(i, (i + 1) % 8) for i in range(min(n_kps, 8))]
                for a, b in skeleton:
                    if (
                        a < n_kps
                        and b < n_kps
                        and scores[a, 0] > self.conf_threshold
                        and scores[b, 0] > self.conf_threshold
                    ):
                        cv2.line(
                            vis,
                            (int(kps[a, 0]), int(kps[a, 1])),
                            (int(kps[b, 0]), int(kps[b, 1])),
                            SKELETON_COLOR,
                            2,
                        )

                for i in range(n_kps):
                    pt = (int(kps[i, 0]), int(kps[i, 1]))
                    conf = float(scores[i, 0])
                    if i == 8:
                        label = f"C ({conf:.2f})"  # face centre
                    elif i == 9:
                        label = f"A ({conf:.2f})"  # arrow tip
                    elif i == 10:
                        label = f"E ({conf:.2f})"  # handle end
                    else:
                        label = f"{i+1} ({conf:.2f})"
                    if conf > self.conf_threshold:
                        cv2.circle(vis, pt, 5, KP_COLOR, -1)
                        cv2.putText(
                            vis,
                            label,
                            (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            KP_COLOR,
                            1,
                        )
                    else:
                        cv2.circle(vis, pt, 4, LOW_CONF_COLOR, 1)
                        cv2.putText(
                            vis,
                            label,
                            (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            LOW_CONF_COLOR,
                            1,
                        )

            out = CompressedImage()
            out.header = header
            out.format = "jpeg"
            _, encoded = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            out.data = encoded.tobytes()
            self.image_pub.publish(out)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"debug image publish failed: {e}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        ValveKeypointNode().run()
    except rospy.ROSInterruptException:
        pass
