#!/usr/bin/env python3
"""Stub operation — the op-writing template, and proof the op seam works.

Logs a throttled one-line summary of every FrameData it receives and stamps
its name on the debug overlay. A real op (gate_pose, tetra_unfold, ...) is
this file plus actual work:

    - params: the op's `params:` dict from the object YAML.
    - ctx (OpContext): ctx.camera_frame, ctx.calibration() -> (K, D) | None,
      ctx.publish_tf(child_frame_id, xyz, stamp, rotation_quat=None) which
      routes through the object map TF server, ctx.publisher(topic, msg_type)
      for op-specific outputs.
    - process(frame): FrameData — ids/pixels/scores (all K keypoints, raw
      confidences: gate on them yourself), mask_probs (C, H, W) in [0, 1]
      with frame.binary_masks() at the calibrated threshold, keypoint_names,
      mask_classes, bbox, stamp.
    - draw(image_bgr, frame): optional overlay layer, only called while the
      debug topic has subscribers. Draw in place.
"""

import cv2
import numpy as np
import rospy


def create_op(params, ctx):
    return StubOp(params, ctx)


class StubOp:
    name = "stub"

    def __init__(self, params, ctx):
        self.ctx = ctx
        self.log_period = float(params.get("log_period", 2.0))
        rospy.loginfo(
            f"stub op ready for '{ctx.object_name}' "
            f"(camera_frame={ctx.camera_frame})"
        )

    def process(self, frame):
        calib = self.ctx.calibration()
        masks = frame.binary_masks()
        if masks is None:
            mask_summary = "none"
        else:
            total = masks[0].size
            mask_summary = " ".join(
                f"{name}={mask.sum() / total:.1%}"
                for name, mask in zip(frame.mask_classes, masks)
            )
        rospy.loginfo_throttle(
            self.log_period,
            f"[stub/{frame.object_name}] kps={len(frame.ids)} "
            f"mean_conf={float(np.mean(frame.scores)):.2f} "
            f"max_conf={float(np.max(frame.scores)):.2f} "
            f"mask_coverage: {mask_summary} "
            f"calib={'yes' if calib is not None else 'no'}",
        )

    def draw(self, image_bgr, frame):
        cv2.putText(
            image_bgr,
            f"op: {self.name}",
            (10, image_bgr.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1,
        )
