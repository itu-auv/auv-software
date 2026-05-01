#!/usr/bin/env python3

import os
import sys
import yaml
import threading

import cv2
import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import Image, CompressedImage

from std_srvs.srv import SetBool, SetBoolResponse
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from dynamic_reconfigure.server import Server
from auv_vision.cfg import ArucoConfig

scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
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

        self.enabled = True
        self.last_published_stamp = rospy.Time(0)

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
                    quat = rvec_tvec_to_quaternion(
                        rvec, tvec, board.force_floor_orientation
                    )

                    if stamp != self.last_published_stamp:
                        result = publish_pose_to_odom(
                            self.camera_frame,
                            board.tf_frame,
                            tvec.flatten(),
                            quat,
                            stamp,
                            self.tf_buffer,
                            self.transform_pub,
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
                    quat = rvec_tvec_to_quaternion(
                        rvec, tvec, marker.force_floor_orientation
                    )
                    if stamp != self.last_published_stamp:
                        publish_pose_to_odom(
                            self.camera_frame,
                            marker.tf_frame,
                            tvec.flatten(),
                            quat,
                            stamp,
                            self.tf_buffer,
                            self.transform_pub,
                        )

        if any_detected:
            self.last_published_stamp = stamp

        return gray, orig_corners, orig_ids, board_debug_info


class ArucoDetectionNode:
    def __init__(self):
        rospy.init_node("aruco_detection")
        rospy.loginfo("ArUco detection node started")

        # Load config
        config_file = rospy.get_param("~config_file", "")
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
            "adaptive_thresh_win_size_max": 75,
            "adaptive_thresh_win_size_step": 8,
            "adaptive_thresh_constant": 5.0,
            "debug_thresh_win_size": 41,
            "min_marker_perimeter_rate": 0.02,
            "max_marker_perimeter_rate": 4.0,
            "polygonal_approx_accuracy_rate": 0.05,
            "error_correction_rate": 0.6,
            "corner_refinement_win_size": 7,
            "corner_refinement_max_iterations": 30,
            "corner_refinement_min_accuracy": 0.05,
            "ransac_iterations": 200,
            "ransac_reproj_error": 20.0,
        }
        self._apply_detector_params()
        self.initial_enabled = rospy.get_param("~enabled", False)

        # Shared resources
        self.bridge = CvBridge()
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

        # Create cameras
        self.cameras = {}
        for cam_key, cam_cfg in self.config.get("cameras", {}).items():
            try:
                cam_info = CameraCalibrationFetcher(cam_cfg["ns"]).get_camera_info()
                camera_matrix = np.array(cam_info.K).reshape(3, 3)
                dist_coeffs = np.array(cam_info.D)

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
                camera.enabled = self.initial_enabled
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

        # Per-camera enable/disable services
        for cam_key in self.cameras:
            rospy.Service(
                f"~{cam_key}/set_enabled",
                SetBool,
                lambda req, k=cam_key: self._set_camera_enabled_cb(req, k),
            )

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

        if result is not None:
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
                                            cv2.circle(img, pt_int, 15, (0, 0, 0), -1)
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

                    pubs["debug"].publish(self._encode_compressed(debug_img))
                    pubs["processed"].publish(self._encode_compressed(debug_gray))

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
