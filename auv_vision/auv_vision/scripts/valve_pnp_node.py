#!/usr/bin/env python3
"""
ROS Noetic node -- valve pose pipeline using GT bbox from simulation.

Ground-truth bbox (from Gazebo model_states) -> ViTPose (8 bolt-hole keypoints)
-> PnP RANSAC -> TF via object_transform_updates topic.

Instead of running YOLO for valve detection, this node projects the known 3D
bolt-hole positions into the camera image using both the valve's and the AUV's
Gazebo poses (from /gazebo/model_states), composed with the static URDF chain
to the camera optical frame.  The tight bounding box around visible projected
keypoints is used as the ViTPose input bbox.

Subscribes:
  ~image_topic           (sensor_msgs/Image)       -- camera feed
  ~camera_info_topic     (sensor_msgs/CameraInfo)  -- camera intrinsics
  /gazebo/model_states   (gazebo_msgs/ModelStates) -- valve pose in world

Publishes:
  ~visualization         (sensor_msgs/Image)       -- annotated image (bbox + kps + axes)
  object_transform_updates (geometry_msgs/TransformStamped) -- valve pose in odom frame

Parameters:
  ~image_topic            (str,   default /taluy/cameras/cam_front/image_raw)
  ~camera_info_topic      (str,   default /taluy/cameras/cam_front/camera_info)
  ~vitpose_model_path     (str,   default auv_detection/models/best.pth)
  ~conf_threshold         (float, default 0.5)   keypoint confidence for PnP inclusion
  ~device                 (str,   default cpu)
  ~frame_id               (str,   default tac/valve)
  ~camera_optical_frame   (str,   default taluy/base_link/front_camera_optical_link)
  ~valve_model_name       (str,   default tac_valve)
  ~bbox_pad               (float, default 0.2)   fractional padding on GT bbox
"""

import sys
import os
import threading
from collections import deque

import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge
from tf.transformations import quaternion_matrix, euler_matrix
import tf2_ros
import tf2_geometry_msgs

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from vitpose_inference import ValvePose, SKELETON

# Bolt hole 3D model points (meters, valve face-center frame)
BOLT_HOLES_3D = np.array([
    [0.0, -0.072832, -0.072832],   # kp 0  -135deg
    [0.0,  0.000000, -0.103000],   # kp 1   -90deg
    [0.0,  0.072832, -0.072832],   # kp 2   -45deg
    [0.0,  0.103000,  0.000000],   # kp 3     0deg
    [0.0,  0.072832,  0.072832],   # kp 4    45deg
    [0.0,  0.000000,  0.103000],   # kp 5    90deg
    [0.0, -0.072832,  0.072832],   # kp 6   135deg
    [0.0, -0.103000,  0.000000],   # kp 7   180deg
], dtype=np.float64)

# Bolt hole positions in STL frame (with X = 0.020 flange offset)
BOLT_HOLES_STL_M = np.array([
    [0.020, -0.072832, -0.072832],
    [0.020,  0.000000, -0.103000],
    [0.020,  0.072832, -0.072832],
    [0.020,  0.103000,  0.000000],
    [0.020,  0.072832,  0.072832],
    [0.020,  0.000000,  0.103000],
    [0.020, -0.072832,  0.072832],
    [0.020, -0.103000,  0.000000],
], dtype=np.float64)

# valve_right link offset within the tac_valve model
VALVE_RIGHT = {"pos": [0.58, 0.555, 1.4205], "rpy": [0.0, np.pi, 0.0]}

# Static URDF chain: base_link -> front_camera_optical_link
# From taluy_description: base_link -> front_camera_link  xyz="0.40125 0 0" rpy="0 0 0"
# From logitech_c920:     front_camera_link -> optical    xyz="0.01 0 0"    rpy="-pi/2 0 -pi/2"
def _build_T_optical_baselink():
    """Precompute T_{optical<-base}: transforms base_link points into optical frame."""
    # URDF joint base_link -> camera_link:  T_{base<-cam} (child-to-parent)
    #   xyz="0.40125 0 0"  rpy="0 0 0"
    T_base_from_cam = np.eye(4)
    T_base_from_cam[0, 3] = 0.40125

    # URDF joint camera_link -> optical_link:  T_{cam<-opt} (child-to-parent)
    #   xyz="0.01 0 0"  rpy="-pi/2 0 -pi/2"
    T_cam_from_opt = np.eye(4)
    T_cam_from_opt[:3, :3] = euler_matrix(-np.pi / 2, 0.0, -np.pi / 2, axes="sxyz")[:3, :3]
    T_cam_from_opt[0, 3] = 0.01

    # Invert both to get parent-to-child direction, then compose:
    # T_{optical<-base} = T_{optical<-cam} @ T_{cam<-base}
    T_cam_from_base = np.linalg.inv(T_base_from_cam)
    T_opt_from_cam  = np.linalg.inv(T_cam_from_opt)

    return T_opt_from_cam @ T_cam_from_base

T_OPTICAL_BASELINK = _build_T_optical_baselink()

SKELETON_COLOR = (0, 255, 0)
KP_COLOR       = (0, 80, 255)
BBOX_COLOR     = (0, 200, 255)
AXES_LENGTH    = 0.08   # metres -- length of projected axes for visualisation


def _get_valve_keypoints_world(valve_pose):
    """Transform 8 bolt holes: STL frame -> valve_right link frame -> world frame."""
    q = valve_pose.orientation
    R_model = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    t_model = np.array([valve_pose.position.x, valve_pose.position.y,
                        valve_pose.position.z])

    roll, pitch, yaw = VALVE_RIGHT["rpy"]
    R_link = euler_matrix(roll, pitch, yaw, axes="sxyz")[:3, :3]
    t_link = np.array(VALVE_RIGHT["pos"])

    p_link  = (R_link @ BOLT_HOLES_STL_M.T).T + t_link
    p_world = (R_model @ p_link.T).T + t_model
    return p_world


def _project_world_to_image(world_pts, T_cam_world, K):
    """Project world-frame 3D points into camera image pixels.

    T_cam_world: (4,4) homogeneous transform  world -> camera optical frame
    K:           (3,3) camera intrinsic matrix

    Returns:
        pts_2d: (N, 2) pixel coords
        behind: (N,)   bool mask -- True if point is behind camera
    """
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    # World -> camera optical frame:  p_cam = R @ p_world + t
    P_cam = (R @ world_pts.T).T + t
    behind = P_cam[:, 2] <= 0.05
    # Pinhole projection
    pts_2d = np.zeros((len(world_pts), 2), dtype=np.float64)
    valid = ~behind
    if valid.any():
        pts_2d[valid, 0] = K[0, 0] * P_cam[valid, 0] / P_cam[valid, 2] + K[0, 2]
        pts_2d[valid, 1] = K[1, 1] * P_cam[valid, 1] / P_cam[valid, 2] + K[1, 2]
    return pts_2d, behind


def rvec_tvec_to_transform_stamped(rvec, tvec, parent_frame, child_frame, stamp):
    """Convert OpenCV PnP output to a ROS TransformStamped."""
    R, _ = cv2.Rodrigues(rvec)

    # Rotation matrix -> quaternion (w, x, y, z via trace method).
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    ts = TransformStamped()
    ts.header.stamp    = stamp
    ts.header.frame_id = parent_frame
    ts.child_frame_id  = child_frame
    ts.transform.translation.x = float(tvec[0])
    ts.transform.translation.y = float(tvec[1])
    ts.transform.translation.z = float(tvec[2])
    ts.transform.rotation.x = float(x)
    ts.transform.rotation.y = float(y)
    ts.transform.rotation.z = float(z)
    ts.transform.rotation.w = float(w)
    return ts


class ValvePnPNode:
    def __init__(self):
        rospy.init_node('valve_pnp', anonymous=False)

        # params
        image_topic   = rospy.get_param('~image_topic',
                                         '/taluy/cameras/cam_front/image_raw')
        info_topic    = rospy.get_param('~camera_info_topic',
                                         '/taluy/cameras/cam_front/camera_info')
        import rospkg
        _models_dir   = os.path.join(
            rospkg.RosPack().get_path('auv_detection'), 'models')
        vp_path       = rospy.get_param('~vitpose_model_path',
                                         os.path.join(_models_dir, 'best.pth'))
        self.conf_thr = rospy.get_param('~conf_threshold', 0.5)
        device_str    = rospy.get_param('~device',         'cpu')
        self.frame_id = rospy.get_param('~frame_id',       'tac/valve')
        self.cam_opt_frame = rospy.get_param(
            '~camera_optical_frame',
            'taluy/base_link/front_camera_optical_link')
        self.valve_model_name = rospy.get_param('~valve_model_name', 'tac_valve')
        self.auv_model_name   = rospy.get_param('~auv_model_name',   'taluy')
        self.bbox_pad = rospy.get_param('~bbox_pad', 0.2)

        # load ViTPose
        rospy.loginfo(f"Loading ViTPose from {vp_path} ...")
        self.vp = ValvePose(vp_path, device=device_str)
        rospy.loginfo("ViTPose ready.")

        # camera intrinsics
        self.K            = None
        self.dist_coeffs  = np.zeros((4, 1), dtype=np.float64)
        self._info_lock   = threading.Lock()

        # Gazebo pose buffer: deque of (sim_time_sec, valve_pose, auv_pose).
        # model_states has no header timestamp and arrives asynchronously from
        # camera images. Without buffering, the GT bbox would use the latest
        # Gazebo pose (often >1s ahead of the image in sim time due to camera
        # rendering/transport latency), causing the bbox crop to drift during
        # movement and degrading ViTPose + PnP accuracy. The buffer lets us
        # look up the pose closest to the image capture timestamp.
        self._gz_buf  = deque(maxlen=3000)
        self._gz_lock = threading.Lock()

        # state
        self.bridge = CvBridge()
        self._prev_rvec = None
        self._prev_tvec = None
        self._prev_stamp = None
        self._max_rotation_jump = rospy.get_param('~max_rotation_jump_deg', 30.0)

        # publishers
        self.pub_vis = rospy.Publisher('~visualization', Image, queue_size=1)

        # TF2 for timestamp-correct transforms to odom
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # publisher (same topic as camera_detection_pose_estimator)
        object_transform_topic = rospy.get_param(
            '~object_transform_topic', '/taluy/map/object_transform_updates')
        self.object_transform_pub = rospy.Publisher(
            object_transform_topic, TransformStamped, queue_size=10)

        # subscribers
        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self.model_states_cb, queue_size=1)
        rospy.Subscriber(info_topic,  CameraInfo, self.info_cb,  queue_size=1)
        rospy.Subscriber(image_topic, Image,      self.image_cb, queue_size=1,
                         buff_size=2**24)

        rospy.loginfo("valve_pnp node started (GT bbox mode).")

    # callbacks

    def model_states_cb(self, msg):
        valve_pose = None
        auv_pose   = None
        for i, name in enumerate(msg.name):
            if name == self.valve_model_name:
                valve_pose = msg.pose[i]
            elif name == self.auv_model_name:
                auv_pose = msg.pose[i]
        if valve_pose is not None and auv_pose is not None:
            stamp = rospy.Time.now().to_sec()
            with self._gz_lock:
                self._gz_buf.append((stamp, valve_pose, auv_pose))

    def _lookup_gz_pose(self, target_sec):
        """Find the buffered Gazebo pose closest to target_sec.

        Returns (valve_pose, auv_pose, dt) or (None, None, None) if buffer
        is empty. dt is target_sec - matched_sec (negative means match is
        newer than target).
        """
        with self._gz_lock:
            if not self._gz_buf:
                return None, None, None
            # Binary search for closest timestamp
            buf = self._gz_buf
            lo, hi = 0, len(buf) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if buf[mid][0] < target_sec:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1 for closest
            best = lo
            if lo > 0:
                if abs(buf[lo - 1][0] - target_sec) < abs(buf[lo][0] - target_sec):
                    best = lo - 1
            stamp, valve_pose, auv_pose = buf[best]
            return valve_pose, auv_pose, target_sec - stamp

    def info_cb(self, msg):
        with self._info_lock:
            if self.K is None:
                self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
                if any(msg.D):
                    self.dist_coeffs = np.array(msg.D, dtype=np.float64)
                rospy.loginfo(f"Camera intrinsics received: "
                              f"fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} "
                              f"cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}")

    def _get_T_cam_world(self, auv_pose):
        """Build the 4x4 transform  world -> camera optical frame from Gazebo AUV pose.

        Composes:  T_optical_baselink (static URDF) @ inv(T_baselink_world)
        where T_baselink_world comes from the AUV's Gazebo model_states pose.
        """
        q = auv_pose.orientation
        T_base_world = quaternion_matrix([q.x, q.y, q.z, q.w])
        T_base_world[0, 3] = auv_pose.position.x
        T_base_world[1, 3] = auv_pose.position.y
        T_base_world[2, 3] = auv_pose.position.z

        # world -> optical = optical<-base @ inv(base<-world)
        #                  = T_OPTICAL_BASELINK @ inv(T_base_world)
        T_world_base = np.linalg.inv(T_base_world)
        return T_OPTICAL_BASELINK @ T_world_base

    def _compute_gt_bbox(self, K, img_h, img_w, valve_pose, auv_pose):
        """Compute ground-truth bounding box by projecting valve bolt holes.

        Returns (x, y, w, h) in pixels or None if not visible.
        """
        T_cam_world = self._get_T_cam_world(auv_pose)

        world_kps = _get_valve_keypoints_world(valve_pose)
        pts_2d, behind = _project_world_to_image(
            world_kps, T_cam_world, K)

        # Only use points in front of camera and within image bounds
        visible = ~behind
        for i in range(len(pts_2d)):
            if visible[i]:
                u, v = pts_2d[i]
                if not (0 <= u < img_w and 0 <= v < img_h):
                    visible[i] = False

        n_vis = int(visible.sum())
        if n_vis < 3:
            rospy.logdebug(f"GT bbox: only {n_vis} keypoints visible.")
            return None

        vis_pts = pts_2d[visible]
        x_min, y_min = vis_pts.min(axis=0)
        x_max, y_max = vis_pts.max(axis=0)

        # Add padding
        bw = x_max - x_min
        bh = y_max - y_min
        pad_x = bw * self.bbox_pad
        pad_y = bh * self.bbox_pad
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(img_w, x_max + pad_x)
        y_max = min(img_h, y_max + pad_y)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def image_cb(self, msg):
        with self._info_lock:
            K = self.K.copy() if self.K is not None else None
            dist = self.dist_coeffs.copy()

        if K is None:
            rospy.logwarn_throttle(5.0, "No camera_info yet -- skipping frame.")
            return

        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"cv_bridge: {e}")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vis     = img_bgr.copy()
        img_h, img_w = img_bgr.shape[:2]

        # Look up buffered Gazebo pose closest to image capture time
        img_sec = msg.header.stamp.to_sec()
        valve_pose, auv_pose, gz_dt = self._lookup_gz_pose(img_sec)

        if valve_pose is None:
            rospy.logwarn_throttle(5.0, "No buffered Gazebo poses yet.")
            self._publish_vis(vis, msg.header)
            return

        # GT bounding box from simulation
        bbox_xywh = self._compute_gt_bbox(K, img_h, img_w, valve_pose, auv_pose)

        if bbox_xywh is None:
            rospy.logdebug("GT bbox: valve not visible.")
            self._publish_vis(vis, msg.header)
            return

        x, y, bw, bh = bbox_xywh
        cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)), BBOX_COLOR, 2)
        cv2.putText(vis, "GT", (int(x), int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, BBOX_COLOR, 1)

        # ViTPose keypoints
        kps, scores = self.vp.predict(img_rgb, bbox_xywh)   # (8,2), (8,1)

        # Draw skeleton on visualisation.
        for a, b in SKELETON:
            if scores[a, 0] > self.conf_thr and scores[b, 0] > self.conf_thr:
                cv2.line(vis,
                         (int(kps[a, 0]), int(kps[a, 1])),
                         (int(kps[b, 0]), int(kps[b, 1])),
                         SKELETON_COLOR, 2)
        # Draw all keypoints -- confident ones bold, low-confidence ones dimmer.
        for i in range(8):
            pt = (int(kps[i, 0]), int(kps[i, 1]))
            conf = scores[i, 0]
            if conf > self.conf_thr:
                cv2.circle(vis, pt, 5, KP_COLOR, -1)
                cv2.putText(vis, f"{i+1} ({conf:.2f})", (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, KP_COLOR, 1)
            else:
                cv2.circle(vis, pt, 4, (100, 100, 100), 1)
                cv2.putText(vis, f"{i+1} ({conf:.2f})", (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        # Build PnP correspondence arrays
        confident_mask = (scores[:, 0] > self.conf_thr)
        n_confident = int(confident_mask.sum())

        if n_confident < 4:
            rospy.logwarn_throttle(
                2.0, f"Only {n_confident}/8 confident keypoints -- need >=4 for PnP.")
            self._publish_vis(vis, msg.header)
            return

        pts3d = BOLT_HOLES_3D[confident_mask].reshape(-1, 1, 3)
        pts2d = kps[confident_mask].reshape(-1, 1, 2).astype(np.float64)

        # Scale reprojection error threshold with bbox size so it doesn't
        # become too strict at close range where the valve fills the image.
        bbox_diag = np.sqrt(bw**2 + bh**2)
        reproj_err = max(4.0, bbox_diag * 0.02)

        # PnP pipeline: P3P RANSAC -> IPPE -> VVS.
        # P3P is the proper minimal solver for RANSAC (3 points per hypothesis).
        # IPPE (Infinitesimal Plane-based Pose Estimation) exploits the fact
        # that all bolt holes are coplanar and returns both ambiguous solutions,
        # letting us pick the temporally consistent one instead of randomly
        # flipping. VVS iteratively refines for sub-pixel accuracy.
        success, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, K, dist,
            iterationsCount=200,
            reprojectionError=reproj_err,
            confidence=0.99,
            flags=cv2.SOLVEPNP_P3P,
        )

        n_inliers = len(inliers) if inliers is not None else 0
        if not success or n_inliers < 6:
            rospy.logwarn_throttle(
                2.0, f"PnP RANSAC failed (inliers={n_inliers}/{n_confident}).")
            self._publish_vis(vis, msg.header)
            return

        rospy.logdebug(f"PnP: inliers={n_inliers}/{n_confident}")

        # IPPE on inliers: coplanar solver that returns both ambiguous solutions
        inlier_3d = pts3d[inliers.flatten()]
        inlier_2d = pts2d[inliers.flatten()]
        n_solutions, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
            inlier_3d, inlier_2d, K, dist,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if n_solutions == 0:
            rospy.logwarn_throttle(2.0, "IPPE returned no solutions.")
            self._publish_vis(vis, msg.header)
            return

        # Pick the solution closest to previous frame's orientation,
        # falling back to lowest reprojection error for the first frame
        if self._prev_rvec is not None and n_solutions > 1:
            R_prev, _ = cv2.Rodrigues(self._prev_rvec)
            best_idx = 0
            best_angle = float('inf')
            for i in range(n_solutions):
                R_i, _ = cv2.Rodrigues(rvecs[i])
                R_delta = R_i @ R_prev.T
                angle = np.degrees(np.arccos(np.clip(
                    (np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))
                if angle < best_angle:
                    best_angle = angle
                    best_idx = i
            if best_angle > self._max_rotation_jump:
                rospy.logwarn(
                    f"PnP ambiguity: both solutions too far from previous "
                    f"({best_angle:.1f}deg > {self._max_rotation_jump:.0f}deg)")
                self._publish_vis(vis, msg.header)
                return
        else:
            best_idx = 0  # IPPE sorts by reprojection error, first is best

        rvec = rvecs[best_idx]
        tvec = tvecs[best_idx]

        # VVS refinement on the IPPE solution for sub-pixel accuracy
        rvec, tvec = cv2.solvePnPRefineVVS(
            inlier_3d, inlier_2d, K, dist,
            rvec.reshape(3, 1), tvec.reshape(3, 1),
        )

        rvec = rvec.flatten()
        tvec = tvec.flatten()
        self._prev_rvec = rvec.copy()

        # Consistency check: compare PnP measurement against prediction from
        # previous PnP + EKF-reported camera motion. Uses TF time-travel to get
        # the relative camera transform between frames. For a static target,
        # any discrepancy is real error (PnP noise + EKF frame-to-frame error).
        dist_m = float(np.linalg.norm(tvec))
        consistency_str = ""
        if self._prev_tvec is not None and self._prev_stamp is not None:
            try:
                # Camera motion between frames: cam@now <- odom <- cam@prev
                T = self.tf_buffer.lookup_transform_full(
                    self.cam_opt_frame, msg.header.stamp,
                    self.cam_opt_frame, self._prev_stamp,
                    'odom', rospy.Duration(1.0))
                q = T.transform.rotation
                R_rel = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
                t_rel = np.array([T.transform.translation.x,
                                  T.transform.translation.y,
                                  T.transform.translation.z])
                tvec_predicted = R_rel @ self._prev_tvec + t_rel
                err = tvec - tvec_predicted
                consistency_str = f"  err={np.linalg.norm(err):.4f}m"
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                pass
        self._prev_tvec = tvec.copy()
        self._prev_stamp = msg.header.stamp
        rospy.loginfo(f"PnP: d={dist_m:.3f}m{consistency_str}")

        # Transform PnP result to odom frame using the image timestamp.
        # Uses tf_buffer.transform() which looks up odom->camera at the
        # correct image capture time, avoiding the stale-pose issue that
        # would occur if the object map TF server did the transform with
        # ros::Time(0) (latest) instead of the image stamp.
        ts = rvec_tvec_to_transform_stamped(
            rvec, tvec,
            parent_frame=self.cam_opt_frame,
            child_frame=self.frame_id,
            stamp=msg.header.stamp,
        )
        try:
            pose_stamped = PoseStamped()
            pose_stamped.header = ts.header
            pose_stamped.pose.position.x = ts.transform.translation.x
            pose_stamped.pose.position.y = ts.transform.translation.y
            pose_stamped.pose.position.z = ts.transform.translation.z
            pose_stamped.pose.orientation = ts.transform.rotation

            transformed = self.tf_buffer.transform(
                pose_stamped, 'odom', rospy.Duration(1.0))

            odom_ts = TransformStamped()
            odom_ts.header = transformed.header
            odom_ts.child_frame_id = self.frame_id
            odom_ts.transform.translation.x = transformed.pose.position.x
            odom_ts.transform.translation.y = transformed.pose.position.y
            odom_ts.transform.translation.z = transformed.pose.position.z
            odom_ts.transform.rotation = transformed.pose.orientation

            self.object_transform_pub.publish(odom_ts)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr_throttle(5.0, f"TF transform to odom failed: {e}")

        # Project axes for visualisation
        axes_3d = np.float32([
            [0,           0,           0          ],   # origin
            [AXES_LENGTH, 0,           0          ],   # X (face normal)
            [0,           AXES_LENGTH, 0          ],   # Y
            [0,           0,           AXES_LENGTH],   # Z
        ]).reshape(-1, 1, 3)
        axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
        axes_2d = axes_2d.reshape(-1, 2).astype(int)
        o = tuple(axes_2d[0])
        cv2.arrowedLine(vis, o, tuple(axes_2d[1]), (0,   0, 255), 2, tipLength=0.2)  # X red
        cv2.arrowedLine(vis, o, tuple(axes_2d[2]), (0, 255,   0), 2, tipLength=0.2)  # Y green
        cv2.arrowedLine(vis, o, tuple(axes_2d[3]), (255, 0,   0), 2, tipLength=0.2)  # Z blue

        dist_m = float(np.linalg.norm(tvec))
        cv2.putText(vis,
                    f"valve  d={dist_m:.2f}m  inliers={len(inliers)}/{n_confident}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        self._publish_vis(vis, msg.header)

    def _publish_vis(self, img, header):
        try:
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
        except Exception as e:
            rospy.logerr_throttle(5.0, f"vis publish failed: {e}")


if __name__ == '__main__':
    node = ValvePnPNode()
    rospy.spin()
