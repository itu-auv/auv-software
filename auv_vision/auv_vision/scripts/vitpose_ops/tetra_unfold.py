#!/usr/bin/env python3
"""tetra_unfold op — which letter is on which coloured face, filtered over time.

Consumes FrameData (3 letter keypoints + red/green/blue face masks) and
maintains a Bayesian filter over the 6 possible letter->colour permutations
(algorithm and rationale: utils/vitpose_utils.py SECTION 3). Two outputs:

  tetra/letter_colors          std_msgs/String, latched — the mission payload,
                               e.g. "A:red B:blue C:green p=0.97 state=LOCKED"
  tetra_unfold_image/compressed  the unfolded tetrahedron (a "triforce": three
                               coloured corner triangles around the white base)
                               with the current best-guess letters drawn on it.
                               Rendered ONLY while someone is subscribed.

Deliberately no draw() hook: the main vitpose debug overlay stays pure model
output (keypoints + masks), and the unfolding lives on its own topic.

Chirality (which way the colours run around the solid) is a config constant,
not something this op estimates — the object is seen before the competition.
It only decides which corner of the drawn net each colour occupies.
"""

import cv2
import numpy as np
import rospy

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

from utils.vitpose_utils import (
    TETRA_DEFAULTS,
    LetterFaceFilter,
    format_association,
    letter_face_membership,
    render_tetra_net,
)

_FILTER_KEYS = (
    "tau",
    "gain",
    "logp_clip",
    "lock_threshold",
    "min_evidence",
    "min_direct_evidence",
)
_MEMBERSHIP_KEYS = (
    "min_score",
    "inside_min",
    "max_mask_distance_px",
    "proximity_weight",
    "eps",
)


def create_op(params, ctx):
    return TetraUnfoldOp(params, ctx)


class TetraUnfoldOp:
    name = "tetra_unfold"

    def __init__(self, params, ctx):
        self.ctx = ctx
        self.chirality = str(params.get("chirality", "cw")).lower()
        if self.chirality not in ("cw", "ccw"):
            raise ValueError(
                f"tetra_unfold: chirality must be 'cw' or 'ccw', "
                f"got {self.chirality!r}"
            )
        unknown = (
            set(params)
            - set(TETRA_DEFAULTS)
            - {
                "chirality",
                "publish_period",
                "net_size",
                "face_colors",
                "jpeg_quality",
            }
        )
        if unknown:
            raise ValueError(f"tetra_unfold: unknown params {sorted(unknown)}")

        self.membership_cfg = {
            key: float(params[key]) for key in _MEMBERSHIP_KEYS if key in params
        }
        self.filter = LetterFaceFilter(
            **{key: float(params[key]) for key in _FILTER_KEYS if key in params}
        )
        self.publish_period = float(params.get("publish_period", 0.5))
        self.net_size = int(params.get("net_size", 480))
        self.face_colors = {
            name: tuple(int(c) for c in bgr)
            for name, bgr in (params.get("face_colors") or {}).items()
        }
        self.jpeg_quality = int(params.get("jpeg_quality", 80))

        self.result_pub = ctx.publisher("tetra/letter_colors", String, latch=True)
        self.image_pub = ctx.publisher("tetra_unfold_image/compressed", CompressedImage)
        self._reset_srv = rospy.Service("tetra_unfold/reset", Empty, self._on_reset)
        self._last_text = None
        self._last_publish = None

        rospy.loginfo(
            f"tetra_unfold op ready for '{ctx.object_name}': chirality="
            f"{self.chirality}, net on {self.image_pub.resolved_name}, "
            f"association on {self.result_pub.resolved_name}"
        )

    # ------------------------------------------------------------- service

    def _on_reset(self, _request):
        self.filter.reset()
        self._last_text = None
        rospy.loginfo("tetra_unfold: filter reset")
        return EmptyResponse()

    # ------------------------------------------------------------- process

    def process(self, frame):
        if frame.mask_probs is None or len(frame.mask_probs) != 3:
            rospy.logwarn_throttle(
                10.0,
                "tetra_unfold: needs 3 face masks, got "
                f"{0 if frame.mask_probs is None else len(frame.mask_probs)} "
                "— check the checkpoint/config mask_classes",
            )
            return

        # FrameData carries parallel id/pixel/score arrays; rebuild dense
        # per-letter arrays (ids 0=A, 1=B, 2=C), missing letters scored 0.
        pixels = np.zeros((3, 2), dtype=np.float64)
        scores = np.zeros(3, dtype=np.float64)
        for kp_id, pixel, score in zip(frame.ids, frame.pixels, frame.scores):
            if 0 <= kp_id < 3:
                pixels[kp_id] = pixel
                scores[kp_id] = score

        membership, weights, sources = letter_face_membership(
            pixels,
            scores,
            frame.mask_probs,
            mask_threshold=frame.mask_threshold,
            **self.membership_cfg,
        )
        now = frame.stamp.to_sec()
        self.filter.update(membership, weights, now)
        estimate = self.filter.result(now)

        letter_names = frame.keypoint_names or ["A", "B", "C"]
        mask_classes = frame.mask_classes or ["red", "green", "blue"]
        text = format_association(estimate, letter_names, mask_classes)
        if self._last_text != text or (
            self._last_publish is None
            or now - self._last_publish >= self.publish_period
        ):
            self.result_pub.publish(String(data=text))
            self._last_text = text
            self._last_publish = now
        rospy.logdebug_throttle(2.0, f"tetra_unfold: {text} sources={sources}")

        # The unfolded net — only rendered while someone is watching.
        if self.image_pub.get_num_connections() == 0:
            return
        net = render_tetra_net(
            estimate,
            letter_names,
            mask_classes,
            chirality=self.chirality,
            face_colors=self.face_colors,
            size=self.net_size,
            per_letter_scores=scores,
        )
        ok, encoded = cv2.imencode(
            ".jpg", net, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return
        message = CompressedImage()
        message.header.stamp = frame.stamp
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self.image_pub.publish(message)
