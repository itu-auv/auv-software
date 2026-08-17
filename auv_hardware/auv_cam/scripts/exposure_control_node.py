#!/usr/bin/env python3
"""RoI-metered exposure control for UVC cameras.

Drives V4L2 exposure (and gain) so the object inside a bounding box is well
exposed, instead of the camera's global auto-exposure metering the whole
frame (which crushes bright objects / glare underwater).

Two methods, switchable via ~method:
  percentile — servo the RoI's ~p90 intensity to a target just below
               saturation. Simple, robust spot-metering baseline.
  gradient   — maximize Shim's soft-log gradient metric inside the RoI via
               finite-difference momentum ascent (AAEC, arXiv:2404.12055
               style; no camera response calibration needed).

The RoI is a constant normalized [x, y, w, h] param, overridden by a
detector: ~detections_topics lists vision_msgs/Detection2DArray topics in
priority order (the ViTPose objectness boxes). A topic is "live" while it
published anything within ~source_timeout — an empty array is the
detectors' "looking, don't see it" heartbeat and keeps it live — and the
highest-priority live topic owns the RoI: its latest non-empty box, padded
by ~roi_pad, until it is older than ~roi_timeout, then the constant RoI.
So with [gate, tetra_front] on the front camera the gate box rules while
the gate pipeline runs (even when it sees nothing) and the tetra scan takes
over only once the gate pipeline is off.

V4L2 controls are set through a second fd on the same /dev node the
GStreamer capture holds — UVC allows that.
"""

import fcntl
import os
import struct

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Int32
from std_srvs.srv import SetBool, SetBoolResponse
from vision_msgs.msg import Detection2DArray

# --- V4L2 plumbing (structs from linux/videodev2.h) ---

V4L2_CID_EXPOSURE_AUTO = 0x009A0901
V4L2_CID_EXPOSURE_ABSOLUTE = 0x009A0902
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903  # 1 = cam may drop fps
V4L2_CID_GAIN = 0x00980913

V4L2_EXPOSURE_MANUAL = 1
V4L2_EXPOSURE_APERTURE_PRIORITY = 3  # UVC "auto"


def _iowr(nr, size):
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


VIDIOC_G_CTRL = _iowr(27, 8)  # struct v4l2_control: u32 id, s32 value
VIDIOC_S_CTRL = _iowr(28, 8)
VIDIOC_QUERYCTRL = _iowr(36, 68)  # struct v4l2_queryctrl


class V4l2Device:
    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDWR)

    def close(self):
        os.close(self.fd)

    def query(self, cid):
        """Return (min, max, step, default) or None if control missing."""
        buf = bytearray(struct.pack("=II32s4i3I", cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
        try:
            fcntl.ioctl(self.fd, VIDIOC_QUERYCTRL, buf)
        except OSError:
            return None
        _, _, _, mn, mx, step, default, flags, _, _ = struct.unpack("=II32s4i3I", buf)
        if flags & 0x1:  # V4L2_CTRL_FLAG_DISABLED
            return None
        return mn, mx, step, default

    def get(self, cid):
        buf = bytearray(struct.pack("=Ii", cid, 0))
        fcntl.ioctl(self.fd, VIDIOC_G_CTRL, buf)
        return struct.unpack("=Ii", buf)[1]

    def set(self, cid, value):
        buf = bytearray(struct.pack("=Ii", cid, int(value)))
        fcntl.ioctl(self.fd, VIDIOC_S_CTRL, buf)


# --- Metrics ---


def roi_percentile(gray, pct):
    return float(np.percentile(gray, pct))


def shim_gradient_metric(gray, delta, lam):
    """Shim soft-log gradient score in [0, 1]."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy) / (4.0 * 255.0 * np.sqrt(2.0))
    m = np.zeros_like(g)
    mask = g >= delta
    m[mask] = np.log(lam * (g[mask] - delta) + 1.0)
    return float(np.sum(m) / (g.size * np.log(lam * (1.0 - delta) + 1.0)))


class ExposureControlNode:
    def __init__(self):
        device = rospy.get_param("~device", rospy.get_param("device", ""))
        if not device:
            raise RuntimeError(
                "no ~device param and no camera 'device' param in namespace"
            )
        self.dev = V4l2Device(device)

        self.method = rospy.get_param("~method", "percentile")
        self.enabled = rospy.get_param("~enabled", True)

        # RoI: constant normalized [x, y, w, h]; detector boxes override it
        self.roi_norm = rospy.get_param("~roi", [0.25, 0.25, 0.5, 0.5])
        self.roi_timeout = rospy.get_param("~roi_timeout", 2.0)
        self.source_timeout = rospy.get_param("~source_timeout", 2.0)
        self.roi_pad = rospy.get_param("~roi_pad", 0.05)  # fraction of box size
        self.detections_topics = list(rospy.get_param("~detections_topics", []))
        # per topic: [last_msg_stamp, last_box_stamp, box (x0, y0, x1, y1)];
        # stamps None until the first message (Time(0) would look fresh)
        self.sources = [[None, None, None] for _ in self.detections_topics]

        self.settle_frames = rospy.get_param("~settle_frames", 3)
        self.update_period = rospy.get_param("~update_period", 0.15)
        self.max_dim = rospy.get_param("~metric_max_dim", 240)  # downsample cap

        # percentile servo
        self.percentile = rospy.get_param("~percentile", 90.0)
        self.target = rospy.get_param("~target", 210.0)
        self.deadband = rospy.get_param("~deadband", 12.0)
        self.kp = rospy.get_param("~kp", 1.2)

        # gradient ascent
        self.alpha = rospy.get_param("~alpha", 0.04)  # step, fraction of exposure range
        self.momentum = rospy.get_param("~momentum", 0.8)
        self.plateau_eps = rospy.get_param("~plateau_eps", 0.005)
        self.probe_period = rospy.get_param(
            "~probe_period", 5
        )  # updates between dithers at peak

        self.use_gain = rospy.get_param("~use_gain", True)
        self.gain_step = rospy.get_param("~gain_step", 8)
        self.restore_auto = rospy.get_param("~restore_auto_on_exit", True)

        # --- device control ranges ---
        q = self.dev.query(V4L2_CID_EXPOSURE_ABSOLUTE)
        if q is None:
            raise RuntimeError("%s has no exposure_absolute control" % device)
        dev_min, dev_max, self.exp_step, _ = q
        self.exp_step = max(self.exp_step, 1)
        self.exp_min = max(dev_min, rospy.get_param("~exposure_min", dev_min))
        # default cap ~30 ms (units of 100 us on UVC) so fps holds at 30
        self.exp_max = min(dev_max, rospy.get_param("~exposure_max", 300))

        gq = self.dev.query(V4L2_CID_GAIN)
        if gq is None:
            self.use_gain = False
            self.gain_min = self.gain_max = 0
        else:
            self.gain_min, self.gain_max = gq[0], gq[1]

        # --- state ---
        self.exposure = None
        self.gain = None
        self.frames_since_write = 1000
        self.last_update = rospy.Time(0)
        self.velocity = 0.0
        self.prev_sample = None  # (exposure, metric)
        self.probe_dir = 1
        self.hold_count = 0
        self.bridge = CvBridge()

        if self.enabled:
            self._go_manual()

        self.pub_metric = rospy.Publisher("~metric", Float32, queue_size=1)
        self.pub_exposure = rospy.Publisher("~exposure", Int32, queue_size=1)
        self.pub_gain = rospy.Publisher("~gain", Int32, queue_size=1)
        self.pub_debug = None
        if rospy.get_param("~debug_image", True):
            self.pub_debug = rospy.Publisher(
                "~debug_image/compressed", CompressedImage, queue_size=1
            )

        for i, topic in enumerate(self.detections_topics):
            rospy.Subscriber(
                topic,
                Detection2DArray,
                self.detections_cb,
                callback_args=i,
                queue_size=1,
            )
        rospy.Subscriber(
            rospy.get_param("~image_topic", "image_raw"),
            Image,
            self.image_cb,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Service("~set_enabled", SetBool, self.enable_cb)
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo(
            "exposure_control: %s method=%s exposure=[%d..%d] step=%d gain=%s",
            device,
            self.method,
            self.exp_min,
            self.exp_max,
            self.exp_step,
            "[%d..%d]" % (self.gain_min, self.gain_max) if self.use_gain else "n/a",
        )

    # --- device helpers ---

    def _go_manual(self):
        try:
            self.dev.set(V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL)
        except OSError as e:
            rospy.logwarn("set exposure_auto=manual failed: %s", e)
        try:  # keep the camera from dropping fps on long exposure
            self.dev.set(V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)
        except OSError:
            pass
        self.exposure = int(
            np.clip(
                self.dev.get(V4L2_CID_EXPOSURE_ABSOLUTE), self.exp_min, self.exp_max
            )
        )
        self._write_exposure(self.exposure)
        if self.use_gain:
            self.gain = self.dev.get(V4L2_CID_GAIN)

    def _write_exposure(self, value):
        value = int(np.clip(value, self.exp_min, self.exp_max))
        value -= (value - self.exp_min) % self.exp_step
        try:
            self.dev.set(V4L2_CID_EXPOSURE_ABSOLUTE, value)
        except OSError as e:
            rospy.logwarn_throttle(5.0, "set exposure failed: %s", e)
            return
        if value != self.exposure:
            self.frames_since_write = 0
        self.exposure = value

    def _write_gain(self, value):
        value = int(np.clip(value, self.gain_min, self.gain_max))
        try:
            self.dev.set(V4L2_CID_GAIN, value)
        except OSError as e:
            rospy.logwarn_throttle(5.0, "set gain failed: %s", e)
            return
        if value != self.gain:
            self.frames_since_write = 0
        self.gain = value

    # --- callbacks ---

    def enable_cb(self, req):
        if req.data and not self.enabled:
            self.enabled = True
            self._go_manual()
        elif not req.data and self.enabled:
            self.enabled = False
            if self.restore_auto:
                self._restore_auto()
        return SetBoolResponse(success=True, message="enabled=%s" % self.enabled)

    def detections_cb(self, msg, index):
        now = rospy.Time.now()
        src = self.sources[index]
        src[0] = now
        if not msg.detections:
            return  # heartbeat: source alive, nothing seen
        det = max(
            msg.detections,
            key=lambda d: d.results[0].score if d.results else 0.0,
        )
        b = det.bbox
        px, py = self.roi_pad * b.size_x, self.roi_pad * b.size_y
        src[1] = now
        src[2] = (
            int(b.center.x - b.size_x * 0.5 - px),
            int(b.center.y - b.size_y * 0.5 - py),
            int(b.center.x + b.size_x * 0.5 + px),
            int(b.center.y + b.size_y * 0.5 + py),
        )

    def detector_roi(self):
        """Box of the highest-priority live source, or None."""
        now = rospy.Time.now()
        for last_msg, last_box, box in self.sources:
            if last_msg is None or (now - last_msg).to_sec() >= self.source_timeout:
                continue
            if box is not None and (now - last_box).to_sec() < self.roi_timeout:
                return box
            return None  # live source, but nothing (fresh) seen
        return None

    def current_roi(self, w, h):
        """Pixel-space (x0, y0, x1, y1), detector box wins while fresh."""
        box = self.detector_roi()
        if box is not None:
            x0, y0, x1, y1 = box
        else:
            rx, ry, rw, rh = self.roi_norm
            x0, y0 = int(rx * w), int(ry * h)
            x1, y1 = int((rx + rw) * w), int((ry + rh) * h)
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        return x0, y0, x1, y1

    def image_cb(self, msg):
        self.frames_since_write += 1
        if not self.enabled:
            return

        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        h, w = gray.shape
        x0, y0, x1, y1 = self.current_roi(w, h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            rospy.logwarn_throttle(5.0, "RoI too small/off-frame, skipping")
            return
        crop = gray[y0:y1, x0:x1]
        scale = max(crop.shape) / float(self.max_dim)
        if scale > 1.0:
            crop = cv2.resize(
                crop, (int(crop.shape[1] / scale), int(crop.shape[0] / scale))
            )

        pct = roi_percentile(crop, self.percentile)
        metric = (
            shim_gradient_metric(crop, delta=0.06, lam=1000.0)
            if self.method == "gradient"
            else pct
        )

        self.pub_metric.publish(metric)
        self.pub_exposure.publish(self.exposure)
        if self.use_gain:
            self.pub_gain.publish(self.gain)
        if self.pub_debug is not None and self.pub_debug.get_num_connections() > 0:
            self.publish_debug(gray, (x0, y0, x1, y1), pct, metric, msg.header)

        # measurement only valid once the new exposure has latched
        if self.frames_since_write <= self.settle_frames:
            return
        now = rospy.Time.now()
        if (now - self.last_update).to_sec() < self.update_period:
            return
        self.last_update = now

        if self.method == "gradient":
            self.step_gradient(metric, pct)
        else:
            self.step_percentile(pct)

    # --- controllers ---

    def _gain_step_for(self, err):
        # scale with error so big errors unwind gain about as fast as the
        # exposure servo moves, small ones stay gentle near the deadband
        return self.gain_step * int(np.clip(abs(err) / self.deadband, 1, 4))

    def step_percentile(self, pct):
        err = self.target - pct
        if abs(err) <= self.deadband:
            return
        if err < 0 and self.use_gain and self.gain > self.gain_min:
            # too bright: gain is a last resort in both directions — shed it
            # to the floor before trading away exposure (best SNR at rest)
            self._write_gain(self.gain - self._gain_step_for(err))
            return
        if err < 0 and self.exposure <= self.exp_min:
            return  # true hardware floor: gain 0, exposure at min
        if err > 0 and self.exposure >= self.exp_max:
            # too dark at ceiling exposure: add gain
            if self.use_gain and self.gain < self.gain_max:
                self._write_gain(self.gain + self._gain_step_for(err))
            return
        factor = 1.0 + self.kp * err / 255.0
        new_e = self.exposure * factor
        if abs(new_e - self.exposure) < self.exp_step:
            new_e = self.exposure + np.sign(err) * self.exp_step
        self._write_exposure(new_e)

    def step_gradient(self, metric, pct):
        exp_range = float(self.exp_max - self.exp_min)

        # zero-metric plateau (saturated, black, or textureless-dim) carries no
        # gradient signal: spot-meter with the percentile servo until texture appears
        if metric < self.plateau_eps:
            self.velocity = 0.0
            self.prev_sample = None
            self.step_percentile(pct)
            return

        if self.prev_sample is not None and self.prev_sample[0] != self.exposure:
            e_prev, m_prev = self.prev_sample
            dmde = (metric - m_prev) / (self.exposure - e_prev)
            # relative metric change over full exposure range, clipped to [-1, 1]
            g = float(np.clip(dmde * exp_range / max(metric, 1e-6), -1.0, 1.0))
            if (
                g * self.velocity < 0
            ):  # reversal: kill momentum to damp ringing at the peak
                self.velocity = 0.0
            self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * g
        self.prev_sample = (self.exposure, metric)

        d_e = int(round(self.alpha * self.velocity * exp_range))
        if d_e == 0:
            # at/near peak: mostly hold, dither occasionally so finite
            # differences keep flowing without visible exposure flicker
            self.hold_count += 1
            if self.hold_count < self.probe_period:
                return
            self.hold_count = 0
            d_e = self.probe_dir * max(self.exp_step, int(round(0.02 * exp_range)))
            self.probe_dir = -self.probe_dir
        new_e = int(np.clip(self.exposure + d_e, self.exp_min, self.exp_max))
        if new_e == self.exposure:  # pinned at a bound: bounce
            self.velocity = 0.0
            new_e = self.exposure - np.sign(d_e) * self.exp_step
        self._write_exposure(new_e)

    # --- misc ---

    def publish_debug(self, gray, box, pct, metric, header):
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        x0, y0, x1, y1 = box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        txt = "E=%s G=%s p%d=%.0f M=%.4f" % (
            self.exposure,
            self.gain if self.use_gain else "-",
            int(self.percentile),
            pct,
            metric,
        )
        cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        out = CompressedImage()
        out.header = header
        out.format = "jpeg"
        out.data = encoded.tobytes()
        self.pub_debug.publish(out)

    def _restore_auto(self):
        try:
            self.dev.set(V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_APERTURE_PRIORITY)
            rospy.loginfo("exposure_control: restored auto exposure")
        except OSError as e:
            rospy.logwarn("restore auto exposure failed: %s", e)

    def on_shutdown(self):
        if self.enabled and self.restore_auto:
            self._restore_auto()
        self.dev.close()


if __name__ == "__main__":
    rospy.init_node("exposure_control")
    ExposureControlNode()
    rospy.spin()
