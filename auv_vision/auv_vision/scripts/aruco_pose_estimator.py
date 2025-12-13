#!/usr/bin/env python3
"""
ArUco Auto-Detect & Pose Estimation Node
Automatically detects the correct ArUco dictionary and performs pose estimation.
Compatible with OpenCV 4.2 (Jetson/Ubuntu 18/20).
Transforms to odom frame.
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

class ArucoAutoEstimator:
    def __init__(self):
        rospy.init_node('aruco_auto_estimator', anonymous=True)
        
        # --- params ---
        self.marker_size = rospy.get_param('~marker_size', 0.1)
        self.camera_ns = rospy.get_param('~camera_ns', 'taluy/cameras/cam_front')
        
        # Camera optical frame for transform
        self.camera_frame = "taluy/base_link/front_camera_optical_link"
        
        # Fetch camera calibration using CameraCalibrationFetcher (like camera_detection_pose_estimator.py)
        rospy.loginfo(f"Fetching camera calibration for: {self.camera_ns}")
        calibration_fetcher = camera_calibrations.CameraCalibrationFetcher(self.camera_ns, True)
        self.calibration = calibration_fetcher.get_camera_info()
        
        # Extract camera matrix and distortion coefficients
        self.camera_matrix = np.array(self.calibration.K).reshape(3, 3)
        self.dist_coeffs = np.array(self.calibration.D)
        
        rospy.loginfo(f"Camera calibration loaded! Using frame: {self.camera_frame}")
        rospy.loginfo(f"Camera matrix:\n{self.camera_matrix}")
        
        # OpenCV 3.x and 4.x (pre-4.7)
        self.ARUCO_DICTS = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }
        
        self.current_dict_name = None
        self.current_dict_obj = None
        
        # --- ROS SETUP ---
        self.bridge = CvBridge()
        
        # TF2 buffer and listener for transformations (like camera_detection_pose_estimator.py)
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
        
        rospy.loginfo("ArUco Auto-Detector started. Searching for markers...")
        rospy.loginfo("Publishing to: object_transform_updates (odom frame)")

    def image_cb(self, msg):
        rospy.loginfo_once("Image callback - Calibration loaded from CameraCalibrationFetcher")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Use the calibration loaded at startup
        cam_mat = self.camera_matrix
        dist_coef = self.dist_coeffs

        # --- ARUCO SCANNING LOOP ---
        found_ids = None
        found_corners = None
        
        # Dictionaries to try
        dictionaries_to_try = []
        if self.current_dict_obj is not None:
            dictionaries_to_try.append((self.current_dict_name, self.current_dict_obj))
        else:
            for name, enum_val in self.ARUCO_DICTS.items():
                dictionaries_to_try.append((name, cv2.aruco.Dictionary_get(enum_val)))

        detector_params = cv2.aruco.DetectorParameters_create()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        for name, dict_obj in dictionaries_to_try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, dict_obj, parameters=detector_params
            )
            
            if ids is not None and len(ids) > 0:
                found_ids = ids
                found_corners = corners
                
                if self.current_dict_name != name:
                    self.current_dict_name = name
                    self.current_dict_obj = dict_obj
                    rospy.loginfo(f"----------------------------------------")
                    rospy.loginfo(f"SUCCESS! Marker found. Dictionary: {name}")
                    rospy.loginfo(f"----------------------------------------")
                break
        
        # --- VISUALIZATION AND PUBLISHING ---
        debug_img = cv_image.copy()
        
        if found_ids is not None:
            cv2.aruco.drawDetectedMarkers(debug_img, found_corners, found_ids)
            
            # Pose Estimate
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                found_corners, self.marker_size, cam_mat, dist_coef
            )
            
            for i in range(len(found_ids)):
                cv2.drawFrameAxes(
                    debug_img, cam_mat, dist_coef, 
                    rvecs[i], tvecs[i], self.marker_size * 0.5
                )
                
                # Transform to odom and publish
                self.publish_transform_to_odom(
                    found_ids[i][0], 
                    rvecs[i][0], 
                    tvecs[i][0], 
                    msg.header
                )
                
                # Display distance on screen
                dist = np.linalg.norm(tvecs[i][0])
                cv2.putText(debug_img, f"ID:{found_ids[i][0]} Dist:{dist:.2f}m", 
                           (10, 70 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(debug_img, f"DICT: {self.current_dict_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            self.current_dict_name = None
            cv2.putText(debug_img, "SEARCHING...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Publish debug image
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

    def publish_transform_to_odom(self, marker_id, rvec, tvec, header):
        """
        Transforms marker position from camera frame to odom frame and publishes.
        """
        # Rotation Vector -> Matrix -> Quaternion
        rot_mat, _ = cv2.Rodrigues(rvec)
        
        # Create homogeneous matrix
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rot_mat
        transform_matrix[0:3, 3] = tvec
        
        # Quaternion calculation
        quat = tf.transformations.quaternion_from_matrix(transform_matrix)
        
        child_frame_id = f"aruco_{marker_id}"
        
        # Create TransformStamped message in camera frame
        transform_stamped_msg = TransformStamped()
        transform_stamped_msg.header.stamp = header.stamp
        transform_stamped_msg.header.frame_id = self.camera_frame
        transform_stamped_msg.child_frame_id = child_frame_id
        
        transform_stamped_msg.transform.translation = Vector3(tvec[0], tvec[1], tvec[2])
        transform_stamped_msg.transform.rotation = Quaternion(quat[0], quat[1], quat[2], quat[3])
        
        # TF broadcast (in camera frame)
        self.tf_broadcaster.sendTransform(transform_stamped_msg)
        
        try:
            # Create PoseStamped (in camera frame)
            pose_stamped = PoseStamped()
            pose_stamped.header = transform_stamped_msg.header
            pose_stamped.pose.position = transform_stamped_msg.transform.translation
            pose_stamped.pose.orientation = transform_stamped_msg.transform.rotation
            
            # Transform to odom frame
            transformed_pose_stamped = self.tf_buffer.transform(
                pose_stamped, "odom", rospy.Duration(1.0)
            )
            
            # Create final TransformStamped in odom frame
            final_transform_stamped = TransformStamped()
            final_transform_stamped.header = transformed_pose_stamped.header
            final_transform_stamped.child_frame_id = child_frame_id
            final_transform_stamped.transform.translation = transformed_pose_stamped.pose.position
            final_transform_stamped.transform.rotation = transform_stamped_msg.transform.rotation
            
            # Publish to object_transform_updates topic
            self.object_transform_pub.publish(final_transform_stamped)
            
            rospy.loginfo_once("Publishing frame to odom")
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr_throttle(5, f"Transform error for {child_frame_id}: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ArucoAutoEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass