#!/usr/bin/env python3
"""
ArUco Board Pose Estimation Node

Uses a custom ArUco board for more accurate pose estimation than single markers.
Dictionary: DICT_6X6_250

Currently configured for TEST BOARD (2 markers).
Competition board code is commented out below.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3, Quaternion
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf2_geometry_msgs
import tf.transformations
import auv_common_lib.vision.camera_calibrations as camera_calibrations


class ArucoBoardEstimator:
    """
    ArUco Board-based pose estimator for docking station detection.
    Uses multiple markers on a known planar layout for improved accuracy.
    """
    
    # =========================================================================
    # TEST BOARD CONFIGURATION (2 markers, 33.5 x 83.5 cm)
    # =========================================================================
    # Board Layout (as seen from camera, origin at center):
    #
    #     ┌───────────────────────────────────────────────────────────────────┐
    #     │                                                                   │
    #     │      [ID:28]              28cm gap              [ID:7]            │  33.5cm
    #     │      (left)                                     (right)           │
    #     │                                                                   │
    #     └───────────────────────────────────────────────────────────────────┘
    #                                   83.5cm
    #
    # - Markers: 19x19 cm
    # - ID 28 (left):  8.5cm from left edge, 7.5cm from bottom edge
    # - ID 7 (right):  6.5cm from right edge, 7.5cm from bottom edge
    # =========================================================================
    
    BOARD_WIDTH = 0.835      # 83.5cm (horizontal, long edge)
    BOARD_HEIGHT = 0.335     # 33.5cm (vertical, short edge)
    MARKER_SIZE = 0.190      # 19cm
    
    # Marker IDs
    MARKER_IDS = [28, 7]     # left, right
    
    # Marker positions (distance from edges in meters)
    # ID 28 (left marker): 8.5cm from left edge, 7.5cm from bottom edge
    # ID 7 (right marker): 6.5cm from right edge, 7.5cm from bottom edge
    LEFT_MARKER_X_OFFSET = 0.085   # 8.5cm from left edge
    RIGHT_MARKER_X_OFFSET = 0.065  # 6.5cm from right edge
    MARKER_Y_OFFSET = 0.075        # 7.5cm from bottom edge (both markers)
    
    # =========================================================================
    # COMPETITION BOARD CONFIGURATION (4 markers, 800 x 1200 mm)
    # Uncomment this section and comment out TEST BOARD section above for competition
    # =========================================================================
    # """
    # Board Layout (dimensions in meters, origin at center):
    #     ┌─────────────────────────────────────────┐
    #     │  [ID:0]                         [ID:1]  │
    #     │  top-left                     top-right │
    #     │                                         │
    #     │               (0,0) ORIGIN              │  1200mm
    #     │                                         │
    #     │  [ID:2]                         [ID:3]  │
    #     │  bottom-left                bottom-right│
    #     └─────────────────────────────────────────┘
    #                       800mm
    #
    # Marker size: 150mm (0.15m)
    # Edge offset: 35mm from each edge
    # """
    # BOARD_WIDTH = 0.800      # 800mm
    # BOARD_HEIGHT = 1.200     # 1200mm
    # MARKER_SIZE = 0.150      # 150mm
    # EDGE_OFFSET = 0.035      # 35mm from edge
    # MARKER_IDS = [0, 1, 2, 3]  # top-left, top-right, bottom-left, bottom-right
    # =========================================================================
    
    def __init__(self):
        rospy.init_node('aruco_board_estimator', anonymous=True)
        
        # --- Parameters ---
        self.camera_ns = rospy.get_param('~camera_ns', 'taluy/cameras/cam_front')
        self.camera_frame = rospy.get_param(
            '~camera_frame', 
            'taluy/base_link/front_camera_optical_link'
        )
        
        # Fetch camera calibration
        rospy.loginfo(f"Fetching camera calibration for: {self.camera_ns}")
        calibration_fetcher = camera_calibrations.CameraCalibrationFetcher(
            self.camera_ns, True
        )
        self.calibration = calibration_fetcher.get_camera_info()
        
        # Extract camera matrix and distortion coefficients
        self.camera_matrix = np.array(self.calibration.K).reshape(3, 3)
        self.dist_coeffs = np.array(self.calibration.D)
        
        rospy.loginfo(f"Camera calibration loaded! Using frame: {self.camera_frame}")
        rospy.loginfo(f"Camera matrix:\n{self.camera_matrix}")
        
        # --- ArUco Setup (OpenCV 4.7-4.12) ---
        # Use 6x6 dictionary as specified
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # Create ArucoDetector for marker detection
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # Create the marker object points dictionary (OpenCV 4.7 Board constructor has Python binding issues)
        # We use solvePnP directly with marker corner positions instead
        self.marker_obj_points = self._create_test_board_points()
        # self.marker_obj_points = self._create_competition_board_points()  # Uncomment for competition
        
        rospy.loginfo(f"Custom ArUco board configured with {len(self.MARKER_IDS)} markers")
        
        # --- ROS Setup ---
        self.bridge = CvBridge()
        
        # TF2 buffer and listener for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Publishers
        self.object_transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )
        self.pose_pub = rospy.Publisher('~detected_pose', PoseStamped, queue_size=10)
        self.debug_pub = rospy.Publisher('~debug_image', Image, queue_size=1)
        
        # Image subscriber
        image_topic = f'/{self.camera_ns}/image_raw'
        rospy.loginfo(f"Subscribing to: {image_topic}")
        self.img_sub = rospy.Subscriber(image_topic, Image, self.image_cb)
        
        rospy.loginfo("ArUco Board Estimator started!")
        rospy.loginfo(f"Board: {self.BOARD_WIDTH*100:.1f}x{self.BOARD_HEIGHT*100:.1f}cm, "
                     f"Markers: {self.MARKER_SIZE*100:.0f}cm, IDs: {self.MARKER_IDS}")
    
    def _create_test_board_points(self):
        """
        Create marker object points for the TEST board with 2 markers.
        
        Coordinate system:
        - Origin at center of board
        - X axis: right (along 83.5cm edge)
        - Y axis: down (along 33.5cm edge)
        - Z axis: out of the board (towards camera)
        
        Returns:
            dict: Marker ID -> numpy array of 4 corner 3D points (4x3)
        """
        half_w = self.BOARD_WIDTH / 2.0   # 0.4175m
        half_h = self.BOARD_HEIGHT / 2.0  # 0.1675m
        
        # Calculate marker top-left corners (origin at board center)
        # Y is measured from bottom edge, so we need to flip for "Y down" convention
        # Bottom of board is at +half_h, top is at -half_h
        
        # ID 28 (left marker): 8.5cm from left edge, 7.5cm from bottom edge
        # Top-left corner of marker:
        id28_x = -half_w + self.LEFT_MARKER_X_OFFSET
        id28_y = half_h - self.MARKER_Y_OFFSET - self.MARKER_SIZE  # from bottom, marker extends up
        
        # ID 7 (right marker): 6.5cm from right edge, 7.5cm from bottom edge
        id7_x = half_w - self.RIGHT_MARKER_X_OFFSET - self.MARKER_SIZE
        id7_y = half_h - self.MARKER_Y_OFFSET - self.MARKER_SIZE  # same Y as left marker
        
        marker_positions = {
            28: (id28_x, id28_y),  # left marker
            7: (id7_x, id7_y),     # right marker
        }
        
        # Build objPoints dict: marker ID -> 4 corner 3D coordinates
        # Corner order: top-left, top-right, bottom-right, bottom-left (clockwise)
        marker_obj_points = {}
        for marker_id in self.MARKER_IDS:
            x, y = marker_positions[marker_id]
            corners = np.array([
                [x, y, 0],                                        # top-left
                [x + self.MARKER_SIZE, y, 0],                     # top-right
                [x + self.MARKER_SIZE, y + self.MARKER_SIZE, 0],  # bottom-right
                [x, y + self.MARKER_SIZE, 0],                     # bottom-left
            ], dtype=np.float32)
            marker_obj_points[marker_id] = corners
        
        # Log marker positions for debugging
        rospy.loginfo("TEST BOARD marker positions (relative to center):")
        for marker_id, corners in marker_obj_points.items():
            center = corners.mean(axis=0)
            rospy.loginfo(f"  Marker {marker_id}: center at ({center[0]*100:.1f}, {center[1]*100:.1f}) cm")
        
        return marker_obj_points
    
    def _create_competition_board_points(self):
        """
        Create marker object points for the COMPETITION board with 4 corner markers (800x1200mm panel).
        
        Coordinate system:
        - Origin at center of board
        - X axis: right (along 800mm edge)
        - Y axis: down (along 1200mm edge)
        - Z axis: out of the board (towards camera)
        
        Returns:
            dict: Marker ID -> numpy array of 4 corner 3D points (4x3)
        """
        # Competition board constants (redefined here for clarity when this method is used)
        board_width = 0.800      # 800mm
        board_height = 1.200     # 1200mm
        marker_size = 0.150      # 150mm
        edge_offset = 0.035      # 35mm from edge
        marker_ids = [0, 1, 2, 3]  # top-left, top-right, bottom-left, bottom-right
        
        half_w = board_width / 2.0   # 0.400m
        half_h = board_height / 2.0  # 0.600m
        
        # Marker top-left corner positions (relative to board center)
        marker_positions = {
            0: (-half_w + edge_offset, -half_h + edge_offset),  # top-left
            1: (half_w - edge_offset - marker_size, -half_h + edge_offset),  # top-right
            2: (-half_w + edge_offset, half_h - edge_offset - marker_size),  # bottom-left
            3: (half_w - edge_offset - marker_size, half_h - edge_offset - marker_size),  # bottom-right
        }
        
        # Build objPoints dict: marker ID -> 4 corner 3D coordinates
        # Corner order: top-left, top-right, bottom-right, bottom-left (clockwise)
        marker_obj_points = {}
        for marker_id in marker_ids:
            x, y = marker_positions[marker_id]
            corners = np.array([
                [x, y, 0],                                        # top-left
                [x + marker_size, y, 0],                          # top-right
                [x + marker_size, y + marker_size, 0],            # bottom-right
                [x, y + marker_size, 0],                          # bottom-left
            ], dtype=np.float32)
            marker_obj_points[marker_id] = corners
        
        # Log marker positions for debugging
        rospy.loginfo("COMPETITION BOARD marker positions (relative to center):")
        for marker_id, corners in marker_obj_points.items():
            center = corners.mean(axis=0)
            rospy.loginfo(f"  Marker {marker_id}: center at ({center[0]*1000:.0f}, {center[1]*1000:.0f}) mm")
        
        return marker_obj_points
    
    def image_cb(self, msg):
        """Process incoming images and estimate board pose using solvePnP."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers using ArucoDetector (OpenCV 4.7+ API)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        debug_img = cv_image.copy()
        
        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
            
            # Collect 3D object points and 2D image points for known board markers
            obj_points_list = []
            img_points_list = []
            matched_ids = []
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.marker_obj_points:
                    # This marker is part of our board
                    obj_points_list.append(self.marker_obj_points[marker_id])
                    img_points_list.append(corners[i].reshape(4, 2))
                    matched_ids.append(marker_id)
            
            if len(matched_ids) > 0:
                # Combine all points into single arrays for solvePnP
                obj_points = np.vstack(obj_points_list).astype(np.float32)
                img_points = np.vstack(img_points_list).astype(np.float32)
                
                # Estimate board pose using solvePnP
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Draw board coordinate frame
                    cv2.drawFrameAxes(
                        debug_img, self.camera_matrix, self.dist_coeffs,
                        rvec, tvec, 0.15  # 15cm axis length
                    )
                    
                    # Publish transform
                    self.publish_board_transform(rvec, tvec, msg.header)
                    
                    # Display info
                    dist = np.linalg.norm(tvec)
                    cv2.putText(
                        debug_img, 
                        f"BOARD DETECTED ({len(matched_ids)} markers) Dist: {dist:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    
                    # Show which marker IDs were detected
                    cv2.putText(
                        debug_img,
                        f"IDs: {sorted(matched_ids)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )
                else:
                    cv2.putText(
                        debug_img,
                        f"Board markers found but pose failed",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )
            else:
                # Detected markers but none are part of our board
                detected_ids = sorted([id for id in ids.flatten()])
                cv2.putText(
                    debug_img,
                    f"Unknown markers: {detected_ids}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                )
        else:
            cv2.putText(
                debug_img, "SEARCHING FOR BOARD...",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        
        # Publish debug image
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
    
    def publish_board_transform(self, rvec, tvec, header):
        """
        Publish the board pose as a transform.
        The pose represents the board center relative to the camera.
        """
        # Convert rotation vector to quaternion
        rot_mat, _ = cv2.Rodrigues(rvec)
        
        # Create 4x4 homogeneous matrix
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rot_mat
        transform_matrix[0:3, 3] = tvec.flatten()
        
        # Get quaternion
        quat = tf.transformations.quaternion_from_matrix(transform_matrix)
        
        # Create TransformStamped message
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
        
        # Broadcast TF
        self.tf_broadcaster.sendTransform(transform_stamped)
        
        # Also publish pose
        pose_stamped = PoseStamped()
        pose_stamped.header = transform_stamped.header
        pose_stamped.pose.position = transform_stamped.transform.translation
        pose_stamped.pose.orientation = transform_stamped.transform.rotation
        self.pose_pub.publish(pose_stamped)
        
        # Transform to odom frame and publish
        try:
            transformed_pose = self.tf_buffer.transform(
                pose_stamped, "odom", rospy.Duration(1.0)
            )
            
            odom_transform = TransformStamped()
            odom_transform.header = transformed_pose.header
            odom_transform.child_frame_id = "docking_station"
            odom_transform.transform.translation = transformed_pose.pose.position
            odom_transform.transform.rotation = transform_stamped.transform.rotation
            
            self.object_transform_pub.publish(odom_transform)
            
            rospy.loginfo_once("Board pose publishing to odom frame")
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"Transform to odom failed: {e}")
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ArucoBoardEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass