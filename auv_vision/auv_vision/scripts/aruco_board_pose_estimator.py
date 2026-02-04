#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import yaml
import rospkg
import threading
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import SetBool, SetBoolResponse
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3, Quaternion
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf2_geometry_msgs
import tf.transformations
import auv_common_lib.vision.camera_calibrations as camera_calibrations
from dynamic_reconfigure.server import Server
from auv_vision.cfg import ArucoConfig


class ArucoBoardEstimator:
    """
    Estimates the pose of an ArUco marker board using OpenCV's ArUco detection.

    Board Layout (dimensions in meters, origin at center):
        ┌─────────────────────────────────────────┐
        │  [ID:28]                       [ID:7]   │
        │  top-left                     top-right │
        │                                         │
        │               (0,0) ORIGIN              │  1200mm
        │                                         │
        │  [ID:19]                       [ID:96]  │
        │  bottom-left                bottom-right│
        └─────────────────────────────────────────┘
                          800mm
    """

    BOARD_WIDTH = 0.800
    BOARD_HEIGHT = 1.200
    MARKER_SIZE = 0.145
    EDGE_OFFSET = 0.048
    MARKER_IDS = [28, 7, 19, 96]

    def __init__(self):
        rospy.init_node("aruco_board_estimator", anonymous=True)

        try:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("auv_vision")
            config_dir = os.path.join(pkg_path, "config")
            os.makedirs(config_dir, exist_ok=True)
            self.config_file = os.path.join(config_dir, "aruco_debug_params.yaml")
        except rospkg.common.ResourceNotFound:
            rospy.logerr(
                "auv_vision package not found. Parameter persistence disabled."
            )
            self.config_file = ""

        self.camera_ns = rospy.get_param("~camera_ns", "taluy/cameras/cam_bottom")
        self.camera_frame = rospy.get_param(
            "~camera_frame", "taluy/base_link/bottom_camera_optical_link"
        )

        rospy.loginfo(f"Fetching camera calibration for: {self.camera_ns}")
        calibration_fetcher = camera_calibrations.CameraCalibrationFetcher(
            self.camera_ns, True
        )
        self.calibration = calibration_fetcher.get_camera_info()

        self.camera_matrix = np.array(self.calibration.K).reshape(3, 3)
        self.dist_coeffs = np.array(self.calibration.D)

        rospy.loginfo(f"Camera calibration loaded! Using frame: {self.camera_frame}")
        rospy.loginfo(f"Camera matrix:\n{self.camera_matrix}")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_ARUCO_ORIGINAL
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.refine_params = cv2.aruco.RefineParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params, self.refine_params
        )

        self.clahe_clip_limit = 2.0
        self.clahe_tile_size = 8
        self.blur_strength = 3
        self.adaptive_thresh_win_size_min = 7
        self.adaptive_thresh_win_size_max = 75
        self.adaptive_thresh_win_size_step = 8
        self.adaptive_thresh_constant = 5.0
        self.min_marker_perimeter_rate = 0.02
        self.max_marker_perimeter_rate = 4.0
        self.polygonal_approx_accuracy_rate = 0.05
        self.error_correction_rate = 0.6
        self.corner_refinement_win_size = 7
        self.corner_refinement_max_iterations = 30
        self.corner_refinement_min_accuracy = 0.05
        self.ransac_iterations = 200
        self.ransac_reproj_error = 20.0

        self._apply_detector_params()
        self.marker_obj_points = self._create_competition_board_points()
        obj_points_list = [self.marker_obj_points[mid] for mid in self.MARKER_IDS]
        ids_array = np.array(self.MARKER_IDS, dtype=np.int32)
        self.board = cv2.aruco.Board(obj_points_list, self.aruco_dict, ids_array)

        rospy.loginfo(
            f"Custom ArUco board configured with {len(self.MARKER_IDS)} markers"
        )

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.object_transform_pub = rospy.Publisher(
            "/taluy/map/object_transform_updates", TransformStamped, queue_size=10
        )
        self.pose_pub = rospy.Publisher("~detected_pose", PoseStamped, queue_size=10)
        self.debug_pub = rospy.Publisher(
            "~debug_image/compressed", CompressedImage, queue_size=1
        )
        self.debug_processed_pub = rospy.Publisher(
            "~debug_image_processed/compressed", CompressedImage, queue_size=1
        )
        self.debug_threshold_pub = rospy.Publisher(
            "~debug_image_threshold/compressed", CompressedImage, queue_size=1
        )
        self.board_detected_pub = rospy.Publisher("~board_detected", Bool, queue_size=1)

        self._debug_images_lock = threading.Lock()
        self._debug_images = None
        self._debug_images_ready = threading.Event()
        self._shutdown = False
        self._debug_thread = threading.Thread(
            target=self._debug_publisher_loop, daemon=True
        )
        self._debug_thread.start()

        image_topic = f"/{self.camera_ns}/image_raw"
        rospy.loginfo(f"Subscribing to: {image_topic}")
        self.img_sub = rospy.Subscriber(image_topic, Image, self.image_cb)

        self.last_published_stamp = rospy.Time(0)
        self.enabled = True
        self.set_enabled_srv = rospy.Service(
            "~set_enabled", SetBool, self._set_enabled_cb
        )

        saved_params = self._load_saved_parameters()
        for param_name, value in saved_params.items():
            rospy.set_param(f"~{param_name}", value)

        self.dyn_reconf_server = Server(ArucoConfig, self._reconfigure_callback)

        rospy.loginfo("ArUco Board Estimator started!")
        rospy.loginfo(f"Markers: {self.MARKER_SIZE*100:.0f}cm, IDs: {self.MARKER_IDS}")

    def _load_saved_parameters(self):
        """Load parameters from the saved YAML file, return empty dict if not found."""
        if not self.config_file or not os.path.exists(self.config_file):
            rospy.loginfo("No saved parameters found, using defaults.")
            return {}

        try:
            with open(self.config_file, "r") as f:
                params = yaml.safe_load(f) or {}
            rospy.loginfo(f"Loaded saved parameters from: {self.config_file}")
            return params
        except Exception as e:
            rospy.logwarn(f"Failed to load saved parameters: {e}")
            return {}

    def _save_parameters(self):
        """Save current parameters to the YAML file."""
        if not self.config_file:
            return

        params = {
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_size": self.clahe_tile_size,
            "blur_strength": self.blur_strength,
            "adaptive_thresh_win_size_min": self.adaptive_thresh_win_size_min,
            "adaptive_thresh_win_size_max": self.adaptive_thresh_win_size_max,
            "adaptive_thresh_win_size_step": self.adaptive_thresh_win_size_step,
            "adaptive_thresh_constant": self.adaptive_thresh_constant,
            "min_marker_perimeter_rate": self.min_marker_perimeter_rate,
            "max_marker_perimeter_rate": self.max_marker_perimeter_rate,
            "polygonal_approx_accuracy_rate": self.polygonal_approx_accuracy_rate,
            "error_correction_rate": self.error_correction_rate,
            "corner_refinement_win_size": self.corner_refinement_win_size,
            "corner_refinement_max_iterations": self.corner_refinement_max_iterations,
            "corner_refinement_min_accuracy": self.corner_refinement_min_accuracy,
            "ransac_iterations": self.ransac_iterations,
            "ransac_reproj_error": self.ransac_reproj_error,
        }

        try:
            with open(self.config_file, "w") as f:
                yaml.dump(params, f, default_flow_style=False)
            rospy.logdebug(f"Parameters saved to: {self.config_file}")
        except IOError as e:
            rospy.logerr(f"Failed to save parameters: {e}")

    def _apply_detector_params(self):
        """Apply current parameter values to the ArUco detector."""
        self.detector_params.cornerRefinementWinSize = self.corner_refinement_win_size
        self.detector_params.cornerRefinementMaxIterations = (
            self.corner_refinement_max_iterations
        )
        self.detector_params.cornerRefinementMinAccuracy = (
            self.corner_refinement_min_accuracy
        )
        self.detector_params.adaptiveThreshWinSizeMin = (
            self.adaptive_thresh_win_size_min
        )
        self.detector_params.adaptiveThreshWinSizeMax = (
            self.adaptive_thresh_win_size_max
        )
        self.detector_params.adaptiveThreshWinSizeStep = (
            self.adaptive_thresh_win_size_step
        )
        self.detector_params.adaptiveThreshConstant = self.adaptive_thresh_constant
        self.detector_params.minMarkerPerimeterRate = self.min_marker_perimeter_rate
        self.detector_params.maxMarkerPerimeterRate = self.max_marker_perimeter_rate
        self.detector_params.polygonalApproxAccuracyRate = (
            self.polygonal_approx_accuracy_rate
        )
        self.detector_params.errorCorrectionRate = self.error_correction_rate

        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params, self.refine_params
        )

    def _reconfigure_callback(self, config, level):
        """Dynamic reconfigure callback - updates parameters at runtime."""
        rospy.loginfo("Reconfigure request received")

        self.clahe_clip_limit = config.clahe_clip_limit
        self.clahe_tile_size = config.clahe_tile_size
        self.blur_strength = config.blur_strength
        if self.blur_strength % 2 == 0:
            self.blur_strength += 1
            config.blur_strength = self.blur_strength

        self.adaptive_thresh_win_size_min = config.adaptive_thresh_win_size_min
        self.adaptive_thresh_win_size_max = config.adaptive_thresh_win_size_max
        self.adaptive_thresh_win_size_step = config.adaptive_thresh_win_size_step
        self.adaptive_thresh_constant = config.adaptive_thresh_constant

        self.min_marker_perimeter_rate = config.min_marker_perimeter_rate
        self.max_marker_perimeter_rate = config.max_marker_perimeter_rate
        self.polygonal_approx_accuracy_rate = config.polygonal_approx_accuracy_rate
        self.error_correction_rate = config.error_correction_rate

        self.corner_refinement_win_size = config.corner_refinement_win_size
        self.corner_refinement_max_iterations = config.corner_refinement_max_iterations
        self.corner_refinement_min_accuracy = config.corner_refinement_min_accuracy

        self.ransac_iterations = config.ransac_iterations
        self.ransac_reproj_error = config.ransac_reproj_error

        self._apply_detector_params()
        self._save_parameters()

        return config

    def _set_enabled_cb(self, req):
        """Service callback to enable/disable pose estimation."""
        self.enabled = req.data
        state = "enabled" if self.enabled else "disabled"
        rospy.loginfo(f"ArUco Board Estimator {state}")
        return SetBoolResponse(success=True, message=f"Estimator {state}")

    def _debug_publisher_loop(self):
        while not rospy.is_shutdown() and not self._shutdown:
            if not self._debug_images_ready.wait(timeout=0.1):
                continue

            self._debug_images_ready.clear()

            with self._debug_images_lock:
                if self._debug_images is None:
                    continue
                debug_img, debug_gray, thresh_img = self._debug_images
                self._debug_images = None

            try:
                if not rospy.is_shutdown():
                    self.debug_pub.publish(self._encode_compressed(debug_img, "bgr8"))
                    self.debug_processed_pub.publish(
                        self._encode_compressed(debug_gray, "bgr8")
                    )
                    self.debug_threshold_pub.publish(
                        self._encode_compressed(thresh_img, "mono8")
                    )
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"Debug image publish failed: {e}")

    def _encode_compressed(self, cv_image, encoding):
        """Encode a CV image as a CompressedImage message (JPEG)."""
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        _, compressed = cv2.imencode(".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        msg.data = compressed.tobytes()
        return msg

    def _queue_debug_images(self, debug_img, debug_gray, thresh_img):
        with self._debug_images_lock:
            self._debug_images = (
                debug_img.copy(),
                debug_gray.copy(),
                thresh_img.copy(),
            )
        self._debug_images_ready.set()

    def _create_competition_board_points(self):
        """
        Create marker object points for the board. Coordinate system has origin at
        board center, X-right, Y-down, Z-out. Returns dict of marker ID to 4x3 corner points.
        """
        half_w = self.BOARD_WIDTH / 2.0
        half_h = self.BOARD_HEIGHT / 2.0

        marker_positions = {
            28: (-half_w + self.EDGE_OFFSET, -half_h + self.EDGE_OFFSET),
            7: (
                half_w - self.EDGE_OFFSET - self.MARKER_SIZE,
                -half_h + self.EDGE_OFFSET,
            ),
            19: (
                -half_w + self.EDGE_OFFSET,
                half_h - self.EDGE_OFFSET - self.MARKER_SIZE,
            ),
            96: (
                half_w - self.EDGE_OFFSET - self.MARKER_SIZE,
                half_h - self.EDGE_OFFSET - self.MARKER_SIZE,
            ),
        }

        marker_obj_points = {}
        for marker_id in self.MARKER_IDS:
            x, y = marker_positions[marker_id]
            corners = np.array(
                [
                    [x, y, 0],
                    [x + self.MARKER_SIZE, y, 0],
                    [x + self.MARKER_SIZE, y + self.MARKER_SIZE, 0],
                    [x, y + self.MARKER_SIZE, 0],
                ],
                dtype=np.float32,
            )
            marker_obj_points[marker_id] = corners

        rospy.loginfo("COMPETITION BOARD marker positions (relative to center):")
        for marker_id, corners in marker_obj_points.items():
            center = corners.mean(axis=0)
            rospy.loginfo(
                f"  Marker {marker_id}: center at ({center[0]*1000:.0f}, {center[1]*1000:.0f}) mm"
            )

        return marker_obj_points

    def image_cb(self, msg):
        """Process incoming images and estimate board pose using solvePnP."""
        if not self.enabled or rospy.is_shutdown():
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        tile_size = (self.clahe_tile_size, self.clahe_tile_size)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=tile_size)
        gray = clahe.apply(l_channel)
        gray = cv2.medianBlur(gray, self.blur_strength)

        thresh_win_size = (
            self.adaptive_thresh_win_size_min + self.adaptive_thresh_win_size_max
        ) // 2
        if thresh_win_size % 2 == 0:
            thresh_win_size += 1
        thresh_img = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            thresh_win_size,
            int(self.adaptive_thresh_constant),
        )

        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        corners, ids, rejected, _ = self.aruco_detector.refineDetectedMarkers(
            gray,
            self.board,
            corners,
            ids,
            rejected,
            self.camera_matrix,
            self.dist_coeffs,
        )

        debug_img = cv_image.copy()
        debug_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
            cv2.aruco.drawDetectedMarkers(debug_gray, corners, ids)

            obj_points_list = []
            img_points_list = []
            matched_ids = []

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.marker_obj_points:
                    obj_points_list.append(self.marker_obj_points[marker_id])
                    img_points_list.append(corners[i].reshape(4, 2))
                    matched_ids.append(marker_id)

            if len(matched_ids) > 0:
                obj_points = np.vstack(obj_points_list).astype(np.float32)
                img_points = np.vstack(img_points_list).astype(np.float32)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_points,
                    img_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_SQPNP,
                    iterationsCount=self.ransac_iterations,
                    reprojectionError=self.ransac_reproj_error,
                    confidence=0.99,
                )

                if (
                    not success
                    or rvec is None
                    or tvec is None
                    or inliers is None
                    or len(inliers) == 0
                ):
                    rospy.logwarn_throttle(
                        1.0,
                        "solvePnPRansac failed or returned no inliers, skipping pose",
                    )
                    cv2.putText(
                        debug_img,
                        f"RANSAC failed ({len(matched_ids)} markers)",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 0, 255),
                        4,
                    )
                    cv2.putText(
                        debug_gray,
                        f"RANSAC failed ({len(matched_ids)} markers)",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 0, 255),
                        4,
                    )
                else:
                    inlier_indices = inliers.flatten()
                    inlier_obj_points = obj_points[inlier_indices]
                    inlier_img_points = img_points[inlier_indices]
                    rvec, tvec = cv2.solvePnPRefineLM(
                        inlier_obj_points,
                        inlier_img_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                    )

                    inlier_set = set(inlier_indices)
                    for idx, pt in enumerate(img_points):
                        pt_int = tuple(pt.astype(int))
                        cv2.circle(debug_img, pt_int, 15, (0, 0, 0), -1)
                        cv2.circle(debug_gray, pt_int, 15, (0, 0, 0), -1)
                        if idx in inlier_set:
                            cv2.circle(debug_img, pt_int, 10, (0, 255, 0), -1)
                            cv2.circle(debug_gray, pt_int, 10, (0, 255, 0), -1)
                        else:
                            cv2.circle(debug_img, pt_int, 12, (0, 0, 255), 4)
                            cv2.circle(debug_gray, pt_int, 12, (0, 0, 255), 4)

                    self.publish_board_transform(rvec, tvec, msg.header)
                    self.board_detected_pub.publish(Bool(data=True))

                    prev_log_level = cv2.getLogLevel()
                    cv2.setLogLevel(2)
                    cv2.drawFrameAxes(
                        debug_img,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                        0.15,
                    )
                    cv2.drawFrameAxes(
                        debug_gray,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                        0.15,
                    )
                    cv2.setLogLevel(prev_log_level)

                    dist = np.linalg.norm(tvec)
                    info_text = f"BOARD ({len(matched_ids)} markers, {len(inlier_indices)}/{len(img_points)} corners) {dist:.2f}m"
                    cv2.putText(
                        debug_img,
                        info_text,
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        4,
                    )
                    cv2.putText(
                        debug_gray,
                        info_text,
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        4,
                    )
                    cv2.putText(
                        debug_img,
                        f"IDs: {sorted(matched_ids)}",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8,
                        (255, 255, 0),
                        4,
                    )
                    cv2.putText(
                        debug_gray,
                        f"IDs: {sorted(matched_ids)}",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8,
                        (255, 255, 0),
                        4,
                    )
            else:
                detected_ids = sorted(ids.flatten().tolist())
                cv2.putText(
                    debug_img,
                    f"Unknown markers: {detected_ids}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 165, 255),
                    4,
                )
                cv2.putText(
                    debug_gray,
                    f"Unknown markers: {detected_ids}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 165, 255),
                    4,
                )
        else:
            cv2.putText(
                debug_img,
                "SEARCHING FOR BOARD...",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 255),
                4,
            )
            cv2.putText(
                debug_gray,
                "SEARCHING FOR BOARD...",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 255),
                4,
            )

        self._queue_debug_images(debug_img, debug_gray, thresh_img)

    def publish_board_transform(self, rvec, tvec, header):
        """Publish the board pose as a transform relative to the camera."""
        rot_mat, _ = cv2.Rodrigues(rvec)
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rot_mat
        transform_matrix[0:3, 3] = tvec.flatten()

        rot_x_180 = tf.transformations.rotation_matrix(np.pi, [1, 0, 0])
        transform_matrix = transform_matrix @ rot_x_180

        quat = tf.transformations.quaternion_from_matrix(transform_matrix)

        # Force horizontal: board is flat on pool floor, keep only yaw
        euler = tf.transformations.euler_from_quaternion(quat)
        yaw = euler[2]
        quat = tf.transformations.quaternion_from_euler(np.pi, 0.0, yaw)

        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = header.stamp
        transform_stamped.header.frame_id = self.camera_frame
        transform_stamped.child_frame_id = "docking_station"

        transform_stamped.transform.translation = Vector3(
            tvec[0][0], tvec[1][0], tvec[2][0]
        )
        transform_stamped.transform.rotation = Quaternion(
            quat[0], quat[1], quat[2], quat[3]
        )

        if rospy.is_shutdown():
            return

        if header.stamp == self.last_published_stamp:
            return
        self.last_published_stamp = header.stamp

        pose_stamped = PoseStamped()
        pose_stamped.header = transform_stamped.header
        pose_stamped.pose.position = transform_stamped.transform.translation
        pose_stamped.pose.orientation = transform_stamped.transform.rotation
        self.pose_pub.publish(pose_stamped)

        try:
            transformed_pose = self.tf_buffer.transform(
                pose_stamped, "odom", rospy.Duration(1.0)
            )

            odom_transform = TransformStamped()
            odom_transform.header = transformed_pose.header
            odom_transform.child_frame_id = "docking_station"
            odom_transform.transform.translation = transformed_pose.pose.position
            odom_transform.transform.rotation = transformed_pose.pose.orientation

            self.object_transform_pub.publish(odom_transform)

            pos = transformed_pose.pose.position
            rospy.loginfo_throttle(
                0.5,
                f"Board in odom frame - x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}",
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5, f"Transform to odom failed: {e}")

    def run(self):
        rospy.spin()
        self._shutdown = True
        self._debug_images_ready.set()
        self._debug_thread.join(timeout=1.0)


if __name__ == "__main__":
    try:
        node = ArucoBoardEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass
