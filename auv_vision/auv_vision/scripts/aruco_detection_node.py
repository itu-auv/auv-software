#!/usr/bin/env python3

import os
import sys
import time
import yaml
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetScanEnabled, SetScanEnabledResponse
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from dynamic_reconfigure.server import Server
from auv_vision.cfg import ArucoConfig

scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from utils.detection_utils import CameraCalibration
from utils.aruco_utils import (
    get_aruco_dictionary,
    preprocess_image,
    build_board_object_points,
    build_marker_object_points,
    solve_board_pose,
    solve_marker_pose,
    rvec_tvec_to_quaternion,
    publish_pose_to_odom,
)


class BoardTarget:
    def __init__(
        self,
        name,
        tf_frame,
        obj_points_by_id,
        aruco_ids,
        force_floor_orientation,
        cv_board,
    ):
        self.name = name
        self.tf_frame = tf_frame
        self.obj_points_by_id = obj_points_by_id
        self.aruco_ids = set(aruco_ids)
        self.force_floor_orientation = force_floor_orientation
        self.cv_board = cv_board


class MarkerTarget:
    def __init__(self, name, tf_frame, aruco_id, obj_points, force_floor_orientation):
        self.name = name
        self.tf_frame = tf_frame
        self.aruco_id = aruco_id
        self.obj_points = obj_points
        self.force_floor_orientation = force_floor_orientation


class ScanLogger:
    """Logs confirmed ArUco marker discoveries to YAML and saves cropped images."""

    def __init__(
        self, log_dir, confirm_count, confirm_window, image_throttle, dictionary_name
    ):
        self.log_dir = log_dir
        self.confirm_count = confirm_count
        self.confirm_window = confirm_window
        self.image_throttle = image_throttle
        self.dictionary_name = dictionary_name
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.confirmed_ids = set()
        self.detection_history = {}  # {marker_id: deque of monotonic timestamps}
        self.discovery_order = 0
        self.markers_data = []
        self.last_image_time = {}  # {marker_id: monotonic time of last save}
        self.image_counters = {}  # {marker_id: int}

        os.makedirs(log_dir, exist_ok=True)
        self.yaml_path = os.path.join(log_dir, "scan_log.yaml")
        self._write_yaml()
        rospy.loginfo(f"[Scan] Logging to {log_dir}")

    def process_detections(self, ids, corners, cv_image):
        """Process detected markers: confirm new ones, save throttled images."""
        if ids is None or len(ids) == 0:
            return

        now = time.monotonic()

        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            marker_corners = corners[i]

            if marker_id not in self.detection_history:
                self.detection_history[marker_id] = deque()

            history = self.detection_history[marker_id]
            history.append(now)

            # Trim entries outside the confirmation window
            while history and (now - history[0]) > self.confirm_window:
                history.popleft()

            if marker_id not in self.confirmed_ids:
                if len(history) >= self.confirm_count:
                    self._confirm_marker(marker_id, marker_corners, cv_image, now)
            else:
                self._save_throttled_image(marker_id, marker_corners, cv_image, now)

    def _confirm_marker(self, marker_id, corners, cv_image, now):
        self.confirmed_ids.add(marker_id)
        self.discovery_order += 1
        ordinal = self._ordinal(self.discovery_order)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        corners_list = corners.reshape(-1, 2).tolist()
        corners_rounded = [[round(x, 1), round(y, 1)] for x, y in corners_list]

        self.markers_data.append(
            {
                "id": marker_id,
                "discovered": ordinal,
                "confirmed_at": timestamp,
                "corners_px": corners_rounded,
            }
        )

        self._write_yaml()
        self._save_marker_image(marker_id, corners, cv_image)
        self.last_image_time[marker_id] = now

        rospy.loginfo(f"[Scan] Confirmed marker ID {marker_id} ({ordinal})")

    def _save_throttled_image(self, marker_id, corners, cv_image, now):
        last = self.last_image_time.get(marker_id, 0)
        if (now - last) >= self.image_throttle:
            self._save_marker_image(marker_id, corners, cv_image)
            self.last_image_time[marker_id] = now

    def _save_marker_image(self, marker_id, corners, cv_image):
        id_dir = os.path.join(self.log_dir, f"id_{marker_id}")
        os.makedirs(id_dir, exist_ok=True)

        if marker_id not in self.image_counters:
            self.image_counters[marker_id] = 0
        self.image_counters[marker_id] += 1

        filename = f"{self.image_counters[marker_id]:04d}.jpg"
        filepath = os.path.join(id_dir, filename)

        # Crop to marker bounding box with 30% padding
        pts = corners.reshape(-1, 2)
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)

        h, w = cv_image.shape[:2]
        pad = int(max(x_max - x_min, y_max - y_min) * 0.3)
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        crop = cv_image[y_min:y_max, x_min:x_max].copy()

        # Draw marker outline on the crop
        shifted_pts = (pts - np.array([x_min, y_min])).astype(np.int32)
        cv2.polylines(
            crop, [shifted_pts], isClosed=True, color=(0, 255, 0), thickness=2
        )

        cv2.imwrite(filepath, crop)

    def _write_yaml(self):
        with open(self.yaml_path, "w") as f:
            f.write("# ArUco Scan Log\n")
            f.write(f"# Started: {self.start_time}\n")
            f.write(f"# Dictionary: {self.dictionary_name}\n")
            f.write(
                f"# Confirm threshold: {self.confirm_count} detections"
                f" in {self.confirm_window}s\n"
            )
            f.write("\n")
            if not self.markers_data:
                f.write("markers: []\n")
            else:
                f.write("markers:\n")
                for m in self.markers_data:
                    f.write(f"  - id: {m['id']}\n")
                    f.write(f"    discovered: {m['discovered']}\n")
                    f.write(f"    confirmed_at: \"{m['confirmed_at']}\"\n")
                    corners_str = ", ".join(f"[{x}, {y}]" for x, y in m["corners_px"])
                    f.write(f"    corners_px: [{corners_str}]\n")
                    f.write("\n")

    @staticmethod
    def _ordinal(n):
        if 11 <= (n % 100) <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"


class ArucoCamera:
    def __init__(
        self,
        name,
        camera_frame,
        camera_matrix,
        dist_coeffs,
        boards,
        markers,
        tf_buffer,
        transform_pub,
        debug_pubs,
        scan_mode=False,
        scan_id_min=1,
        scan_id_max=99,
    ):
        self.name = name
        self.camera_frame = camera_frame
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.boards = boards
        self.markers = markers
        self.tf_buffer = tf_buffer
        self.transform_pub = transform_pub
        self.debug_pubs = debug_pubs
        self.scan_mode = scan_mode
        # Scan-mode ID window (inclusive). Markers outside [min, max] are
        # dropped right after detection, so they never reach the TF/pose path
        # nor the debug overlay.
        self.scan_id_min = scan_id_min
        self.scan_id_max = scan_id_max

        self.enabled = True
        self.last_published_stamp = rospy.Time(0)

    def _filter_scan_ids(self, corners, ids):
        """Keep only detections whose ArUco ID is within [scan_id_min,
        scan_id_max] (inclusive). Returns (corners, ids) shaped exactly as
        detectMarkers would (a list of corner arrays and an Nx1 id array, or
        ([], None) when nothing survives)."""
        if ids is None or len(ids) == 0:
            return corners, ids
        ids_flat = ids.flatten()
        keep = [
            i
            for i, mid in enumerate(ids_flat)
            if self.scan_id_min <= int(mid) <= self.scan_id_max
        ]
        if len(keep) == len(ids_flat):
            return corners, ids
        if not keep:
            return [], None
        filtered_corners = [corners[i] for i in keep]
        filtered_ids = ids[keep]
        return filtered_corners, filtered_ids

    def process(self, cv_image, stamp, detector, params):
        """Detect ArUco markers and estimate poses for boards and standalone markers.

        Returns (gray, corners, ids, board_debug_info) for debug rendering,
        or None if the frame was skipped.
        """
        if not self.enabled:
            return None

        gray = preprocess_image(
            cv_image,
            params["clahe_clip_limit"],
            params["clahe_tile_size"],
        )

        orig_corners, orig_ids, orig_rejected = detector.detectMarkers(gray)

        if self.scan_mode:
            # Detection-only: no pose estimation, no TF output. Restrict to the
            # configured ID window so out-of-range markers are invisible to
            # both downstream consumers and the debug view.
            orig_corners, orig_ids = self._filter_scan_ids(orig_corners, orig_ids)
            return gray, orig_corners, orig_ids, []

        any_detected = False
        board_debug_info = []
        claimed_ids = set()

        if orig_ids is not None and len(orig_ids) > 0:
            # Process boards (with per-board marker refinement)
            for board in self.boards:
                ref_corners, ref_ids, ref_rejected, _ = detector.refineDetectedMarkers(
                    gray,
                    board.cv_board,
                    [c.copy() for c in orig_corners],
                    orig_ids.copy(),
                    [r.copy() for r in orig_rejected],
                    self.camera_matrix,
                    self.dist_coeffs,
                )

                success, rvec, tvec, matched_ids, inliers = solve_board_pose(
                    board.obj_points_by_id,
                    ref_ids,
                    ref_corners,
                    self.camera_matrix,
                    self.dist_coeffs,
                    params["ransac_iterations"],
                    params["ransac_reproj_error"],
                )

                if success:
                    any_detected = True
                    claimed_ids.update(matched_ids)
                    quat = rvec_tvec_to_quaternion(rvec, tvec)

                    if stamp != self.last_published_stamp:
                        result = publish_pose_to_odom(
                            self.camera_frame,
                            board.tf_frame,
                            tvec.flatten(),
                            quat,
                            stamp,
                            self.tf_buffer,
                            self.transform_pub,
                            force_floor_orientation=board.force_floor_orientation,
                        )
                        if result:
                            pos = result.pose.position
                            rospy.loginfo_throttle(
                                0.5,
                                f"{board.name} in odom - "
                                f"x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}",
                            )

                board_debug_info.append(
                    (board, rvec, tvec, matched_ids, inliers, ref_corners, ref_ids)
                )

            # Process standalone markers (using original detections)
            for marker in self.markers:
                if marker.aruco_id in claimed_ids:
                    continue

                ids_flat = orig_ids.flatten()
                idx_matches = np.where(ids_flat == marker.aruco_id)[0]
                if len(idx_matches) == 0:
                    continue

                marker_corners = orig_corners[idx_matches[0]]
                success, rvec, tvec = solve_marker_pose(
                    marker.obj_points,
                    marker_corners,
                    self.camera_matrix,
                    self.dist_coeffs,
                )

                if success:
                    any_detected = True
                    quat = rvec_tvec_to_quaternion(rvec, tvec)
                    if stamp != self.last_published_stamp:
                        publish_pose_to_odom(
                            self.camera_frame,
                            marker.tf_frame,
                            tvec.flatten(),
                            quat,
                            stamp,
                            self.tf_buffer,
                            self.transform_pub,
                            force_floor_orientation=marker.force_floor_orientation,
                        )

        if any_detected:
            self.last_published_stamp = stamp

        return gray, orig_corners, orig_ids, board_debug_info


class ArucoDetectionNode:
    def __init__(self):
        rospy.init_node("aruco_detection", anonymous=True)
        rospy.loginfo("ArUco detection node started")

        # Mode selection: full pose estimation (yaml-driven) vs scan-only (param-driven).
        self.scan_mode = rospy.get_param("~scan_mode", False)

        # Initial enabled state for all cameras. Set false to launch the node
        # dormant and have it switched on later via the set_enabled service
        # (e.g. the inspect SMACH enables it for the duration of inspection).
        self.start_enabled = rospy.get_param("~start_enabled", True)

        config_file = rospy.get_param("~config_file", "")
        scan_dict_name = rospy.get_param("~aruco_dictionary", "")
        scan_image_topic = rospy.get_param("~image_topic", "")
        scan_camera_frame = rospy.get_param("~camera_frame", "")
        # Scan-mode ID window (inclusive on both ends). Defaults restrict the
        # debug/detection view to IDs 1..99, hiding 0 and anything >= 100.
        self.scan_id_min = rospy.get_param("~scan_id_min", 1)
        self.scan_id_max = rospy.get_param("~scan_id_max", 99)

        scan_params = (
            ("~aruco_dictionary", scan_dict_name),
            ("~image_topic", scan_image_topic),
            ("~camera_frame", scan_camera_frame),
        )

        if self.scan_mode:
            if config_file:
                raise rospy.ROSInitException(
                    "~config_file is not allowed when ~scan_mode is true"
                )
            missing = [name for name, value in scan_params if not value]
            if missing:
                raise rospy.ROSInitException(
                    f"~scan_mode requires: {', '.join(missing)}"
                )
            self.config = {}
            rospy.loginfo("ArUco scan mode: detection-only, no pose estimation.")
        else:
            set_scan_params = [name for name, value in scan_params if value]
            if set_scan_params:
                raise rospy.ROSInitException(
                    f"These params are only allowed with ~scan_mode=true: "
                    f"{', '.join(set_scan_params)}"
                )
            if not config_file:
                raise rospy.ROSInitException(
                    "~config_file parameter is required (path to ArUco objects YAML)"
                )
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)

        # Parameter persistence
        try:
            pkg_path = rospkg.RosPack().get_path("auv_vision")
            config_dir = os.path.join(pkg_path, "config")
            os.makedirs(config_dir, exist_ok=True)
            self.param_save_file = os.path.join(config_dir, "aruco_debug_params.yaml")
        except rospkg.common.ResourceNotFound:
            rospy.logerr(
                "auv_vision package not found. Parameter persistence disabled."
            )
            self.param_save_file = ""

        # ArUco detector
        if self.scan_mode:
            dict_name = scan_dict_name
        else:
            dict_name = self.config.get("aruco_dictionary", "DICT_ARUCO_ORIGINAL")
        self.aruco_dict = get_aruco_dictionary(dict_name)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.refine_params = cv2.aruco.RefineParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params, self.refine_params
        )

        # Detection parameters (updated by dynamic reconfigure)
        self.params = {
            "clahe_clip_limit": 2.0,
            "clahe_tile_size": 8,
            "adaptive_thresh_win_size_min": 7,
            "adaptive_thresh_win_size_max": 31,
            "adaptive_thresh_win_size_step": 10,
            "adaptive_thresh_constant": 5.0,
            "debug_thresh_win_size": 41,
            "min_marker_perimeter_rate": 0.02,
            "max_marker_perimeter_rate": 4.0,
            "polygonal_approx_accuracy_rate": 0.05,
            "error_correction_rate": 0.3,
            "corner_refinement_win_size": 7,
            "corner_refinement_max_iterations": 30,
            "corner_refinement_min_accuracy": 0.05,
            "ransac_iterations": 150,
            "ransac_reproj_error": 3.0,
        }
        self._apply_detector_params()

        # Shared resources
        self.bridge = CvBridge()
        if self.scan_mode:
            self.tf_buffer = None
            self.tf_listener = None
            self.transform_pub = None
        else:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            self.transform_pub = rospy.Publisher(
                "object_transform_updates", TransformStamped, queue_size=10
            )

        # Build targets from config
        markers_config = self.config.get("markers", {})
        boards_config = self.config.get("boards", {})
        standalone_config = self.config.get("standalone_markers", {})

        self.all_boards = {}
        for board_name, board_cfg in boards_config.items():
            obj_points = build_board_object_points(board_cfg, markers_config)
            aruco_ids = list(obj_points.keys())

            obj_points_list = [obj_points[aid] for aid in aruco_ids]
            ids_array = np.array(aruco_ids, dtype=np.int32)
            cv_board = cv2.aruco.Board(obj_points_list, self.aruco_dict, ids_array)

            self.all_boards[board_name] = BoardTarget(
                name=board_name,
                tf_frame=board_cfg["tf_frame"],
                obj_points_by_id=obj_points,
                aruco_ids=aruco_ids,
                force_floor_orientation=board_cfg.get("force_floor_orientation", False),
                cv_board=cv_board,
            )
            rospy.loginfo(
                f"Board '{board_name}' -> '{board_cfg['tf_frame']}' "
                f"with {len(aruco_ids)} markers (IDs: {sorted(aruco_ids)})"
            )

        self.all_markers = {}
        for marker_name, marker_cfg in (standalone_config or {}).items():
            ref = marker_cfg["marker_ref"]
            marker_def = markers_config[ref]
            obj_points = build_marker_object_points(marker_def["size"])

            self.all_markers[marker_name] = MarkerTarget(
                name=marker_name,
                tf_frame=marker_cfg["tf_frame"],
                aruco_id=marker_def["aruco_id"],
                obj_points=obj_points,
                force_floor_orientation=marker_cfg.get(
                    "force_floor_orientation", False
                ),
            )
            rospy.loginfo(
                f"Marker '{marker_name}' (ID {marker_def['aruco_id']}) -> "
                f"'{marker_cfg['tf_frame']}'"
            )

        # Scan logger is created per enable (see _set_global_enabled_cb), each
        # in its own folder named after the enable request's `name`. No logger
        # exists until the node is enabled; the lock guards the swap against the
        # image callbacks that read it.
        self._scan_log_lock = threading.Lock()
        self.scan_logger = None
        if self.scan_mode:
            self.scan_base_dir = os.path.expanduser(
                rospy.get_param("~scan_log_dir", "~/aruco_scans")
            )
            self.scan_dict_name = dict_name
            self.scan_confirm_count = rospy.get_param("~scan_confirm_count", 3)
            self.scan_confirm_window = rospy.get_param("~scan_confirm_window", 2.0)
            self.scan_image_throttle = rospy.get_param("~scan_image_throttle", 2.0)

        # Create cameras
        self.cameras = {}

        if self.scan_mode:
            cam_key = "camera"
            debug_pubs = {
                "debug": rospy.Publisher(
                    f"~{cam_key}/debug/compressed",
                    CompressedImage,
                    queue_size=1,
                ),
                "processed": rospy.Publisher(
                    f"~{cam_key}/debug_processed/compressed",
                    CompressedImage,
                    queue_size=1,
                ),
                "threshold": rospy.Publisher(
                    f"~{cam_key}/debug_threshold/compressed",
                    CompressedImage,
                    queue_size=1,
                ),
            }

            camera = ArucoCamera(
                name=cam_key,
                camera_frame=scan_camera_frame,
                camera_matrix=None,
                dist_coeffs=None,
                boards=[],
                markers=[],
                tf_buffer=None,
                transform_pub=None,
                debug_pubs=debug_pubs,
                scan_mode=True,
                scan_id_min=self.scan_id_min,
                scan_id_max=self.scan_id_max,
            )
            self.cameras[cam_key] = camera

            rospy.Subscriber(
                scan_image_topic,
                Image,
                lambda msg, k=cam_key: self._image_callback(msg, k),
                queue_size=1,
                buff_size=2**24,
            )

            rospy.loginfo(
                f"Scan mode camera '{cam_key}' on {scan_image_topic} "
                f"(frame: {scan_camera_frame}, dict: {dict_name}, "
                f"ID window: {self.scan_id_min}..{self.scan_id_max} inclusive)"
            )

        for cam_key, cam_cfg in self.config.get("cameras", {}).items():
            try:
                calib = CameraCalibration(cam_cfg["ns"])
                camera_matrix = np.array(calib.calibration.K).reshape(3, 3)
                dist_coeffs = np.array(calib.calibration.D)

                cam_boards = [
                    self.all_boards[b]
                    for b in cam_cfg.get("detect_boards", [])
                    if b in self.all_boards
                ]
                cam_markers = [
                    self.all_markers[m]
                    for m in cam_cfg.get("detect_markers", [])
                    if m in self.all_markers
                ]

                debug_pubs = {
                    "debug": rospy.Publisher(
                        f"~{cam_key}/debug/compressed",
                        CompressedImage,
                        queue_size=1,
                    ),
                    "processed": rospy.Publisher(
                        f"~{cam_key}/debug_processed/compressed",
                        CompressedImage,
                        queue_size=1,
                    ),
                    "threshold": rospy.Publisher(
                        f"~{cam_key}/debug_threshold/compressed",
                        CompressedImage,
                        queue_size=1,
                    ),
                }

                camera = ArucoCamera(
                    name=cam_key,
                    camera_frame=cam_cfg["frame"],
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    boards=cam_boards,
                    markers=cam_markers,
                    tf_buffer=self.tf_buffer,
                    transform_pub=self.transform_pub,
                    debug_pubs=debug_pubs,
                )
                self.cameras[cam_key] = camera

                rospy.Subscriber(
                    cam_cfg["image_topic"],
                    Image,
                    lambda msg, k=cam_key: self._image_callback(msg, k),
                    queue_size=1,
                    buff_size=2**24,
                )

                rospy.loginfo(
                    f"Camera '{cam_key}': {len(cam_boards)} board(s), "
                    f"{len(cam_markers)} marker(s) on {cam_cfg['image_topic']}"
                )
            except Exception as e:
                rospy.logerr(f"Failed to initialize camera '{cam_key}': {e}. Skipping.")

        # Apply the configured initial enabled state to every camera.
        for cam in self.cameras.values():
            cam.enabled = self.start_enabled
        if not self.start_enabled:
            rospy.loginfo("ArUco cameras start disabled (~start_enabled=false)")
        elif self.scan_mode:
            # Enabled at launch in scan mode: open a scan folder immediately so
            # detections are logged without waiting for a service call.
            self._start_new_scan(rospy.get_param("~scan_name", "scan"))

        # Enable/disable services
        for cam_key in self.cameras:
            rospy.Service(
                f"~{cam_key}/set_enabled",
                SetBool,
                lambda req, k=cam_key: self._set_camera_enabled_cb(req, k),
            )
        rospy.Service("~set_enabled", SetScanEnabled, self._set_global_enabled_cb)

        # Debug image publishing thread
        self._debug_lock = threading.Lock()
        self._debug_queue = {}
        self._debug_ready = threading.Event()
        self._shutdown = False
        self._debug_thread = threading.Thread(
            target=self._debug_publisher_loop, daemon=True
        )
        self._debug_thread.start()

        # Load saved params and start dynamic reconfigure
        saved = self._load_saved_params()
        for k, v in saved.items():
            rospy.set_param(f"~{k}", v)
        self.dyn_reconf_server = Server(ArucoConfig, self._reconfigure_cb)

        rospy.loginfo("ArUco detection node ready!")

    def _apply_detector_params(self):
        p = self.params
        self.detector_params.cornerRefinementWinSize = p["corner_refinement_win_size"]
        self.detector_params.cornerRefinementMaxIterations = p[
            "corner_refinement_max_iterations"
        ]
        self.detector_params.cornerRefinementMinAccuracy = p[
            "corner_refinement_min_accuracy"
        ]
        self.detector_params.adaptiveThreshWinSizeMin = p[
            "adaptive_thresh_win_size_min"
        ]
        self.detector_params.adaptiveThreshWinSizeMax = p[
            "adaptive_thresh_win_size_max"
        ]
        self.detector_params.adaptiveThreshWinSizeStep = p[
            "adaptive_thresh_win_size_step"
        ]
        self.detector_params.adaptiveThreshConstant = p["adaptive_thresh_constant"]
        self.detector_params.minMarkerPerimeterRate = p["min_marker_perimeter_rate"]
        self.detector_params.maxMarkerPerimeterRate = p["max_marker_perimeter_rate"]
        self.detector_params.polygonalApproxAccuracyRate = p[
            "polygonal_approx_accuracy_rate"
        ]
        self.detector_params.errorCorrectionRate = p["error_correction_rate"]

        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params, self.refine_params
        )

    # Params that must be odd (OpenCV kernel sizes)
    _ODD_PARAMS = {"debug_thresh_win_size"}

    def _reconfigure_cb(self, config, level):
        rospy.loginfo("ArUco reconfigure request")

        for key in self.params:
            if hasattr(config, key):
                self.params[key] = getattr(config, key)

        for key in self._ODD_PARAMS:
            if self.params[key] % 2 == 0:
                self.params[key] += 1
                setattr(config, key, self.params[key])

        self._apply_detector_params()
        self._save_params()

        return config

    def _load_saved_params(self):
        if not self.param_save_file or not os.path.exists(self.param_save_file):
            rospy.loginfo("No saved ArUco parameters found, using defaults.")
            return {}
        try:
            with open(self.param_save_file, "r") as f:
                params = yaml.safe_load(f) or {}
            rospy.loginfo(f"Loaded saved parameters from: {self.param_save_file}")
            return params
        except Exception as e:
            rospy.logwarn(f"Failed to load saved parameters: {e}")
            return {}

    def _save_params(self):
        if not self.param_save_file:
            return
        try:
            with open(self.param_save_file, "w") as f:
                yaml.dump(dict(self.params), f, default_flow_style=False)
        except IOError as e:
            rospy.logerr(f"Failed to save parameters: {e}")

    def _set_camera_enabled_cb(self, req, cam_key):
        cam = self.cameras[cam_key]
        cam.enabled = req.data
        state = "enabled" if req.data else "disabled"
        msg = f"ArUco camera '{cam_key}' {state}"
        rospy.loginfo(msg)
        return SetBoolResponse(success=True, message=msg)

    def _unique_log_dir(self, name):
        """Return base_dir/name, appending 2, 3, ... until the path is free.

        e.g. 'Auto' -> 'Auto' if absent, else 'Auto2', 'Auto3', ...
        """
        candidate = os.path.join(self.scan_base_dir, name)
        if not os.path.exists(candidate):
            return candidate
        i = 2
        while True:
            candidate = os.path.join(self.scan_base_dir, f"{name}{i}")
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def _start_new_scan(self, name):
        """Create a fresh ScanLogger in a uniquely-named folder for `name`."""
        log_dir = self._unique_log_dir(name)
        logger = ScanLogger(
            log_dir=log_dir,
            confirm_count=self.scan_confirm_count,
            confirm_window=self.scan_confirm_window,
            image_throttle=self.scan_image_throttle,
            dictionary_name=self.scan_dict_name,
        )
        with self._scan_log_lock:
            self.scan_logger = logger
        return log_dir

    def _set_global_enabled_cb(self, req):
        if req.enabled and self.scan_mode:
            # Each enable opens a new scan folder named after req.name.
            if not req.name:
                msg = "Enable request requires a non-empty 'name' in scan mode"
                rospy.logerr(msg)
                return SetScanEnabledResponse(success=False, message=msg)
            log_dir = self._start_new_scan(req.name)
            scan_msg = f", logging to {log_dir}"
        else:
            scan_msg = ""

        for cam in self.cameras.values():
            cam.enabled = req.enabled
        state = "enabled" if req.enabled else "disabled"
        msg = f"All ArUco cameras {state}{scan_msg}"
        rospy.loginfo(msg)
        return SetScanEnabledResponse(success=True, message=msg)

    def _image_callback(self, msg, cam_key):
        camera = self.cameras.get(cam_key)
        if camera is None or not camera.enabled:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        result = camera.process(
            cv_image, msg.header.stamp, self.aruco_detector, self.params
        )

        with self._scan_log_lock:
            scan_logger = self.scan_logger
        if scan_logger is not None and result is not None:
            _, corners, ids, _ = result
            scan_logger.process_detections(ids, corners, cv_image)

        if result is not None and self._has_debug_subscribers(cam_key):
            gray, corners, ids, board_debug_info = result
            with self._debug_lock:
                self._debug_queue[cam_key] = (
                    cv_image.copy(),
                    gray,
                    corners,
                    ids,
                    board_debug_info,
                    camera,
                )
            self._debug_ready.set()

    def _has_debug_subscribers(self, cam_key):
        pubs = self.cameras[cam_key].debug_pubs
        return any(p.get_num_connections() > 0 for p in pubs.values())

    def _debug_publisher_loop(self):
        while not rospy.is_shutdown() and not self._shutdown:
            if not self._debug_ready.wait(timeout=0.1):
                continue
            self._debug_ready.clear()

            with self._debug_lock:
                queue = dict(self._debug_queue)
                self._debug_queue.clear()

            for cam_key, (
                cv_image,
                gray,
                corners,
                ids,
                board_debug_info,
                camera,
            ) in queue.items():
                pubs = camera.debug_pubs
                try:
                    if rospy.is_shutdown():
                        break

                    need_annotated = (
                        pubs["debug"].get_num_connections() > 0
                        or pubs["processed"].get_num_connections() > 0
                    )

                    if need_annotated:
                        debug_img = cv_image
                        debug_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        debug_imgs = [debug_img, debug_gray]

                        if ids is not None and len(ids) > 0:
                            for img in debug_imgs:
                                cv2.aruco.drawDetectedMarkers(img, corners, ids)

                            for (
                                board,
                                rvec,
                                tvec,
                                matched_ids,
                                inliers,
                                ref_corners,
                                ref_ids,
                            ) in board_debug_info:
                                if rvec is not None and inliers is not None:
                                    img_pts_list = []
                                    for i, mid in enumerate(ref_ids.flatten()):
                                        if mid in board.obj_points_by_id:
                                            img_pts_list.append(
                                                ref_corners[i].reshape(4, 2)
                                            )
                                    if img_pts_list:
                                        img_points = np.vstack(img_pts_list).astype(
                                            np.float32
                                        )
                                        inlier_set = set(inliers.flatten())
                                        for idx, pt in enumerate(img_points):
                                            pt_int = tuple(pt.astype(int))
                                            for img in debug_imgs:
                                                cv2.circle(
                                                    img, pt_int, 15, (0, 0, 0), -1
                                                )
                                                if idx in inlier_set:
                                                    cv2.circle(
                                                        img, pt_int, 10, (0, 255, 0), -1
                                                    )
                                                else:
                                                    cv2.circle(
                                                        img, pt_int, 12, (0, 0, 255), 4
                                                    )

                                    for img in debug_imgs:
                                        cv2.drawFrameAxes(
                                            img,
                                            camera.camera_matrix,
                                            camera.dist_coeffs,
                                            rvec,
                                            tvec,
                                            0.15,
                                        )

                                    dist = np.linalg.norm(tvec)
                                    n_inliers = len(inliers.flatten())
                                    info_text = (
                                        f"{board.name} ({len(matched_ids)} mkrs, "
                                        f"{n_inliers} inliers) {dist:.2f}m"
                                    )
                                    for img in debug_imgs:
                                        cv2.putText(
                                            img,
                                            info_text,
                                            (20, 70),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.2,
                                            (0, 255, 0),
                                            3,
                                        )
                                        cv2.putText(
                                            img,
                                            f"IDs: {sorted(matched_ids)}",
                                            (20, 120),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.5,
                                            (255, 255, 0),
                                            3,
                                        )
                                elif matched_ids:
                                    for img in debug_imgs:
                                        cv2.putText(
                                            img,
                                            f"RANSAC failed ({len(matched_ids)} mkrs)",
                                            (20, 70),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.5,
                                            (0, 0, 255),
                                            3,
                                        )
                        else:
                            for img in debug_imgs:
                                cv2.putText(
                                    img,
                                    "SEARCHING...",
                                    (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    2.0,
                                    (0, 0, 255),
                                    4,
                                )

                        if pubs["debug"].get_num_connections() > 0:
                            pubs["debug"].publish(self._encode_compressed(debug_img))
                        if pubs["processed"].get_num_connections() > 0:
                            pubs["processed"].publish(
                                self._encode_compressed(debug_gray)
                            )

                    if pubs["threshold"].get_num_connections() > 0:
                        win_size = self.params["debug_thresh_win_size"]
                        thresh_img = cv2.adaptiveThreshold(
                            gray,
                            255,
                            cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY,
                            win_size,
                            int(self.params["adaptive_thresh_constant"]),
                        )
                        pubs["threshold"].publish(self._encode_compressed(thresh_img))

                except Exception as e:
                    rospy.logwarn_throttle(5.0, f"Debug publish failed: {e}")

    def _encode_compressed(self, cv_image):
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        _, compressed = cv2.imencode(".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        msg.data = compressed.tobytes()
        return msg

    def run(self):
        rospy.spin()
        self._shutdown = True
        self._debug_ready.set()
        self._debug_thread.join(timeout=1.0)


if __name__ == "__main__":
    try:
        node = ArucoDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
